import numpy as np

from reid.config import AssociationCfg, CameraSet
from reid.identity import constraints as C
from tests.synth import D_FACE, unit


def sp(i, cam, a, b, gid=None):
    return C.Span(i, cam, a, b, gid)


def test_same_camera_overlap_cannot_link():
    cfg = AssociationCfg()
    assert C.violates(sp(1, "c1", 0, 10), sp(2, "c1", 5, 15), cfg)
    assert not C.violates(sp(1, "c1", 0, 10), sp(2, "c1", 10.5, 15), cfg)  # sequential
    assert not C.violates(sp(1, "c1", 0, 10), sp(2, "c2", 5, 15), cfg)  # other camera
    tol = AssociationCfg(overlap_tolerance_s=2.0)
    assert not C.violates(sp(1, "c1", 0, 10), sp(2, "c1", 9, 15), tol)  # 1 s overlap tolerated
    assert C.violates(sp(1, "c1", 0, 10), sp(2, "c1", 5, 15), tol)


def test_topology_transit_time():
    cams = CameraSet.model_validate(
        {
            "cameras": [{"id": c, "video_path": f"{c}.mp4"} for c in ("a", "b", "c")],
            "topology": [{"from": "a", "to": "b", "min_transit_s": 30}],
        }
    )
    cfg = AssociationCfg()
    assert C.violates(sp(1, "a", 0, 10), sp(2, "b", 20, 25), cfg, cams)  # gap 10 < 30
    assert C.violates(sp(2, "b", 20, 25), sp(1, "a", 0, 10), cfg, cams)  # symmetric
    assert C.violates(sp(1, "a", 0, 10), sp(2, "b", 5, 25), cfg, cams)  # simultaneous
    assert not C.violates(sp(1, "a", 0, 10), sp(2, "b", 45, 50), cfg, cams)  # gap 35
    assert not C.violates(sp(1, "a", 0, 10), sp(2, "c", 11, 12), cfg, cams)  # unconstrained pair
    off = AssociationCfg(use_topology=False)
    assert not C.violates(sp(1, "a", 0, 10), sp(2, "b", 20, 25), off, cams)


def test_blocked_gids_and_matrix():
    cfg = AssociationCfg()
    spans = [sp(1, "c1", 0, 10, 7), sp(2, "c1", 20, 30, 8), sp(3, "c2", 0, 10, 9)]
    assert C.blocked_gids(sp(4, "c1", 5, 8), spans, cfg) == {7}
    m = C.violation_matrix([*spans, sp(4, "c1", 5, 8)], cfg)
    assert m[0, 3] and m[3, 0] and not m[1, 3] and not m[2, 3] and not m.diagonal().any()
    gids = [7, 8, 9]
    cl = C.gid_cannot_links([*spans, sp(4, "c1", 5, 8, 8)], gids, cfg)
    assert cl[0, 1] and cl[1, 0] and not cl[0, 2]  # gid 8 now has a tracklet overlapping gid 7


def test_face_conflict_blocks_gid():
    cfg = AssociationCfg()
    rng = np.random.default_rng(0)
    f1, f2 = unit(rng.normal(size=(2, D_FACE)))
    assert C.face_conflict(f1, f2, 0.9) and not C.face_conflict(f1, f1, 0.3)
    assert not C.face_conflict(None, f2, 0.9)
    assert float(f1 @ f2) < cfg.face_conflict  # fixed seed: two unrelated faces
    b = C.blocked_gids(sp(1, "c1", 0, 1), [], cfg, None, f1, {5: f2, 6: f1})
    assert b == {5}


def faces(*cosines):
    """Face centroids with prescribed cosines to a reference vector f0."""
    d = 8
    f0 = np.zeros(d)
    f0[0] = 1
    out = {0: f0}
    for k, c in enumerate(cosines, start=1):
        v = np.zeros(d)
        v[0], v[k] = c, np.sqrt(1 - c * c)
        out[k] = v
    return out


def test_must_link_needs_margin():
    cfg = AssociationCfg(face_match=0.55, face_margin=0.05)
    # gid 1 is the only face near gid 0 -> must-link
    assert C.must_links(faces(0.7, 0.0), cfg) == {(0, 1)}
    # gid 2 competes (0.68 vs 0.70, within the margin) and looks different from gid 1
    f = faces(0.70, 0.68)
    assert C.must_links(f, cfg) == set()
    # a competitor that is itself the same face as gid 1 is just fragmentation: still linked
    f = {0: f[0], 1: f[1], 2: f[1] * 0.9999 + 1e-4}
    f[2] = f[2] / np.linalg.norm(f[2])
    assert (0, 1) in C.must_links(f, cfg)
    # below face_match: no link
    assert C.must_links(faces(0.4), cfg) == set()


def test_face_ambiguous_helper():
    cfg = AssociationCfg()
    assert not C.face_ambiguous(1, {1: 0.7, 2: 0.4}, {}, cfg)
    assert C.face_ambiguous(1, {1: 0.7, 2: 0.68}, {}, cfg)
    assert not C.face_ambiguous(1, {1: 0.7, 2: 0.68}, {(1, 2): 0.9}, cfg)


def test_time_rules_apply_only_within_one_run():
    """Re-processing the same videos (same cameras, same timestamps) in a new run must not
    forbid the people of the first run (regression: every person got a second identity)."""
    from reid.config import AssociationCfg
    from reid.identity.constraints import Span, violates, violation_matrix

    cfg = AssociationCfg()
    run1 = Span(1, "cam_a", 0.0, 30.0, gid=1, run_id=1)
    run2 = Span(2, "cam_a", 0.0, 30.0, gid=None, run_id=2)
    same_run = Span(3, "cam_a", 10.0, 20.0, gid=None, run_id=1)
    assert not violates(run1, run2, cfg)
    assert violates(run1, same_run, cfg)
    m = violation_matrix([run1, run2, same_run], cfg)
    assert not m[0, 1] and m[0, 2] and not m[1, 2]
    legacy = Span(4, "cam_a", 0.0, 30.0)  # no run id: old behaviour (time rules apply)
    assert violates(run1, legacy, cfg)
