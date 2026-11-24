import numpy as np

from reid.identity.associate import IdentityEngine
from reid.storage import db
from tests.synth import D_FACE, centers, make_env, make_tracklet, unit


def gid_of(env, t):
    return db.get_tracklet(env.conn, t.id).gid


def test_same_body_later_matches_and_other_person_mints(tmp_path):
    rng = np.random.default_rng(0)
    c = centers(2, seed=1)
    env = make_env(tmp_path)
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    a1 = make_tracklet(env, "c1", 0, 10, c[0], rng)
    a2 = make_tracklet(env, "c2", 100, 110, c[0], rng)
    b1 = make_tracklet(env, "c1", 200, 210, c[1], rng)
    g_a1, g_a2, g_b = (eng.on_tracklet(t) for t in (a1, a2, b1))
    assert g_a1 is not None and g_a1 == g_a2 != g_b
    assert db.get_tracklet(env.conn, a2.id).assign_source == "body"
    assert eng.on_tracklet(a2) == g_a2  # idempotent: no double processing


def test_simultaneous_same_camera_tracklets_never_share_a_gid(tmp_path):
    rng = np.random.default_rng(1)
    c = centers(1, seed=2)
    env = make_env(tmp_path)
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    # two different people who look identical to the model, both on screen at the same time
    t1 = make_tracklet(env, "c1", 0, 20, c[0], rng)
    t2 = make_tracklet(env, "c1", 5, 25, c[0], rng)
    g1, g2 = eng.on_tracklet(t1), eng.on_tracklet(t2)
    assert g1 is not None and g2 is not None and g1 != g2
    # a later sequential sighting may join either, but consolidation must never fuse the two
    t3 = make_tracklet(env, "c1", 100, 110, c[0], rng)
    eng.on_tracklet(t3)
    report = eng.end_run(env.run_id)
    assert gid_of(env, t1) != gid_of(env, t2)
    assert report.n_identified == 3


def test_topology_blocks_impossible_transit(tmp_path):
    from reid.config import CameraSet

    rng = np.random.default_rng(2)
    c = centers(1, seed=3)
    cams = CameraSet.model_validate(
        {
            "cameras": [{"id": x, "video_path": f"{x}.mp4"} for x in ("c1", "c2", "c3")],
            "topology": [{"from": "c1", "to": "c2", "min_transit_s": 60}],
        }
    )
    env = make_env(tmp_path)
    eng = IdentityEngine(env.cfg, env.conn, env.vectors, cams)
    t1 = make_tracklet(env, "c1", 0, 10, c[0], rng)
    t2 = make_tracklet(env, "c2", 20, 30, c[0], rng)  # 10 s later, needs 60: other person
    t3 = make_tracklet(env, "c3", 20, 30, c[0], rng)  # unconstrained pair: same person
    g1, g2, g3 = (eng.on_tracklet(t) for t in (t1, t2, t3))
    assert g2 != g1 and g3 == g1


def _two_cameras_with_gap(env, rng, c, f, f_other=None, n_faces=3):
    """Person A seen on c1 (body view 1) and on c2 (a very different body view)."""
    shift = unit(np.random.default_rng(99).normal(size=c.shape[-1]))
    view2 = unit(0.3 * c[0] + shift)  # cos with the first view is only ~0.3
    t1 = make_tracklet(env, "c1", 0, 10, c[0], rng, face_center=f[0])
    t2 = make_tracklet(env, "c2", 100, 110, view2, rng, face_center=f[0], n_faces=n_faces)
    return t1, t2


def test_face_first_assigns_across_a_body_gap(tmp_path):
    rng = np.random.default_rng(3)
    c = centers(2, seed=4)
    f = centers(3, d=D_FACE, seed=5, shared=0.0)
    env = make_env(tmp_path)
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    t1, t2 = _two_cameras_with_gap(env, rng, c, f)
    g1, g2 = eng.on_tracklet(t1), eng.on_tracklet(t2)
    assert g1 == g2 and db.get_tracklet(env.conn, t2.id).assign_source == "face"


def test_face_without_enough_observations_is_not_used(tmp_path):
    rng = np.random.default_rng(4)
    c = centers(2, seed=6)
    f = centers(3, d=D_FACE, seed=7, shared=0.0)
    env = make_env(tmp_path)
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    t1, t2 = _two_cameras_with_gap(env, rng, c, f, n_faces=1)  # a single face < face_min_obs
    g1, g2 = eng.on_tracklet(t1), eng.on_tracklet(t2)
    assert g1 != g2 and db.get_tracklet(env.conn, t2.id).assign_source != "face"


def test_face_without_margin_does_not_link(tmp_path):
    rng = np.random.default_rng(5)
    c = centers(3, seed=8)
    f = centers(2, d=D_FACE, seed=11, shared=0.0)  # two unrelated faces
    env = make_env(tmp_path, **{"association.face_match": 0.5, "association.face_margin": 0.2})
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    ta = make_tracklet(env, "c1", 0, 10, c[0], rng, face_center=f[0], face_noise=0.02)
    tb = make_tracklet(env, "c1", 20, 30, c[1], rng, face_center=f[1], face_noise=0.02)
    ga, gb = eng.on_tracklet(ta), eng.on_tracklet(tb)
    assert ga != gb
    # a query face exactly between the two: both clear face_match, neither leads by the margin
    q = make_tracklet(
        env, "c2", 100, 110, c[2], rng, face_center=unit(f[0] + f[1]), face_noise=0.02
    )
    gq = eng.on_tracklet(q)
    assert db.get_tracklet(env.conn, q.id).assign_source != "face" and gq not in (ga, gb)
    # a clear query (person A's face with a decisive lead) is linked by face
    q2 = make_tracklet(env, "c2", 200, 210, c[2], rng, face_center=f[0], face_noise=0.02)
    assert eng.on_tracklet(q2) == ga
    assert db.get_tracklet(env.conn, q2.id).assign_source == "face"


def test_body_contradicting_face_records_must_link_and_skips_gallery(tmp_path):
    rng = np.random.default_rng(6)
    c = centers(2, seed=9)
    f = centers(2, d=D_FACE, seed=10, shared=0.0)
    env = make_env(tmp_path)
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    ta = make_tracklet(env, "c1", 0, 10, c[0], rng, face_center=f[0])
    tb = make_tracklet(env, "c1", 20, 30, c[1], rng)  # no face observed yet
    ga, gb = eng.on_tracklet(ta), eng.on_tracklet(tb)
    # the body of person B but the face of person A
    t = make_tracklet(env, "c2", 100, 110, c[1], rng, face_center=f[0])
    g = eng.on_tracklet(t)
    assert g == ga and db.get_tracklet(env.conn, t.id).assign_source == "face"
    assert (min(ga, gb), max(ga, gb)) in eng.must_links
    assert not any(r["in_gallery"] for r in db.get_samples(env.conn, tracklet_id=t.id))
