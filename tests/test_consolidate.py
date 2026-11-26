import numpy as np

from reid.identity import consolidate as K
from reid.identity.associate import IdentityEngine
from reid.storage import db
from tests.synth import D_FACE, centers, make_env, make_tracklet, unit


def second_view(c, cos=0.7, seed=99):
    """A different appearance of the same person (e.g. another camera): cosine ~cos to c."""
    r = np.random.default_rng(seed).normal(size=c.shape[-1])
    r = unit(r - (r @ c) * c)
    return unit(cos * c + np.sqrt(1 - cos**2) * r)


def gid_of(env, t):
    return db.get_tracklet(env.conn, t.id).gid


def build_fragmented(env, rng, c):
    """Person X: gid fragments (bodies of the two views are below min_cos), Y and Z are others."""
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    x1 = make_tracklet(env, "c1", 0, 10, c[0], rng)
    y1 = make_tracklet(env, "c1", 2, 12, c[1], rng)  # overlaps x1 on c1 (cannot-link)
    x2 = make_tracklet(env, "c2", 100, 110, second_view(c[0]), rng)
    z1 = make_tracklet(env, "c2", 105, 115, c[2], rng)
    for t in (x1, y1, x2, z1):
        eng.on_tracklet(t)
    return eng, x1, y1, x2, z1


def test_consolidation_merges_fragments_keeps_oldest_and_records_history(tmp_path):
    rng = np.random.default_rng(0)
    c = centers(3, seed=1, shared=0.2)
    env = make_env(
        tmp_path, **{"consolidation.merge_threshold": 0.45, "association.gallery_budget": 4}
    )
    eng, x1, y1, x2, z1 = build_fragmented(env, rng, c)
    assert gid_of(env, x1) != gid_of(env, x2)  # fragmented online (cos below min_cos)
    old_x2 = gid_of(env, x2)
    report = eng.end_run(env.run_id)
    assert gid_of(env, x1) == gid_of(env, x2)
    assert len({gid_of(env, t) for t in (x1, y1, z1)}) == 3  # different people stay apart
    assert [(m.kept, m.absorbed) for m in report.merges] == [(gid_of(env, x1), old_x2)]
    assert gid_of(env, x1) < old_x2  # oldest gid kept
    # reversible: the history records the old gid and the consolidate source
    h = db.get_gid_history(env.conn, x2.id)
    assert h[-1]["old_gid"] == old_x2 and h[-1]["new_source"] == "consolidate"
    assert db.get_tracklet(env.conn, x2.id).assign_source == "consolidate"
    # absorbed identity removed, galleries within the budget, prototypes follow the labels
    assert db.get_identity(env.conn, old_x2) is None
    assert all(len(g) <= 4 for g in eng.store.bank.galleries.values())
    labels = {int(x) for x in env.vectors.get("body_proto").labels}
    assert old_x2 not in labels and gid_of(env, x1) in labels
    in_gallery = db.get_samples(env.conn, in_gallery=True)
    assert all(r["gid"] is not None and r["gid"] != old_x2 for r in in_gallery)


def test_cannot_link_blocks_consolidation(tmp_path):
    rng = np.random.default_rng(1)
    c = centers(1, seed=2)
    env = make_env(tmp_path, **{"consolidation.merge_threshold": 1.0})  # would merge anything
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    a = make_tracklet(env, "c1", 0, 10, c[0], rng)
    b = make_tracklet(env, "c1", 5, 15, c[0], rng)  # simultaneous, same appearance
    eng.on_tracklet(a), eng.on_tracklet(b)
    eng.end_run(env.run_id)
    assert gid_of(env, a) != gid_of(env, b)


def test_face_must_link_merges_despite_body_distance(tmp_path):
    rng = np.random.default_rng(2)
    c = centers(3, seed=3, shared=0.0)
    f = centers(3, d=D_FACE, seed=4, shared=0.0)
    env = make_env(tmp_path, **{"consolidation.merge_threshold": 0.05})  # body alone: no merge
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    a = make_tracklet(env, "c1", 0, 10, c[0], rng)
    b = make_tracklet(env, "c2", 100, 110, c[1], rng)
    eng.on_tracklet(a), eng.on_tracklet(b)
    ga, gb = gid_of(env, a), gid_of(env, b)
    assert ga != gb
    # the same face is later observed with each gid (faces attached after the fact)
    for t, g in ((a, ga), (b, gb)):
        env_f = f[0] + 0.0
        eng.store.face_tracklet.add([t.id], unit(env_f + 0.05 * rng.normal(size=D_FACE))[None], [g])
    res = K.consolidate(eng.store)
    assert [(m.kept, m.absorbed, m.reason) for m in res.merges] == [(ga, gb, "must_link")]
    assert gid_of(env, b) == ga


def test_face_without_margin_does_not_merge(tmp_path):
    rng = np.random.default_rng(3)
    c = centers(3, seed=5, shared=0.0)
    env = make_env(tmp_path, **{"consolidation.merge_threshold": 0.05})
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    ts = [make_tracklet(env, "c1", 100 * i, 100 * i + 10, c[i], rng) for i in range(3)]
    for t in ts:
        eng.on_tracklet(t)
    g = [gid_of(env, t) for t in ts]

    # gid 0's face is similar to gid 1 (0.70) and, within the margin, to gid 2 (0.68), but
    # gids 1 and 2 are different faces (cos 0.49): linking either would be a coin flip
    def vec(*xs):
        v = np.zeros(D_FACE)
        v[: len(xs)] = xs
        return v

    s1, s2 = np.sqrt(1 - 0.70**2), np.sqrt(1 - 0.68**2)
    faces = [vec(1), vec(0.70, s1), vec(0.68, 0, s2)]
    for t, gg, fc in zip(ts, g, faces, strict=True):
        eng.store.face_tracklet.add([t.id], unit(fc)[None], [gg])
    res = K.consolidate(eng.store)
    assert res.merges == [] and (g[0], g[1]) not in res.must_links


def test_manual_ops_merge_split_rename(tmp_path):
    rng = np.random.default_rng(4)
    c = centers(2, seed=7, shared=0.0)
    env = make_env(tmp_path, **{"consolidation.merge_threshold": 0.05})
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    a1 = make_tracklet(env, "c1", 0, 10, c[0], rng)
    b1 = make_tracklet(env, "c1", 20, 30, c[1], rng)
    a2 = make_tracklet(env, "c2", 0, 10, c[0], rng)
    for t in (a1, b1, a2):
        eng.on_tracklet(t)
    eng.end_run(env.run_id)
    ga, gb = gid_of(env, a1), gid_of(env, b1)
    assert gid_of(env, a2) == ga
    K.rename_identity(env.conn, ga, "Alice")
    assert db.get_identity(env.conn, ga).label == "Alice"
    kept = K.merge_identities(env.conn, env.vectors, env.cfg, gb, ga)
    assert kept == ga and gid_of(env, b1) == ga and db.get_identity(env.conn, gb) is None
    new = K.split_tracklet(env.conn, env.vectors, env.cfg, b1.id)  # undo for b1
    assert gid_of(env, b1) == new and db.get_tracklet(env.conn, b1.id).assign_source == "manual"
    assert gid_of(env, a1) == ga == gid_of(env, a2)
    # the persisted cannot-link keeps consolidation from fusing them again
    cfg2 = env.cfg.model_copy(deep=True)
    cfg2.consolidation.merge_threshold = 2.0
    res = K.recluster(env.conn, env.vectors, cfg2)
    assert res.merges == [] and gid_of(env, b1) == new


def _attach_same_face(eng, rng, tracklets_gids, face):
    for t, g in tracklets_gids:
        eng.store.face_tracklet.add([t.id], unit(face + 0.05 * rng.normal(size=D_FACE))[None], [g])


def test_cannot_link_beats_face_must_link(tmp_path):
    """Same face on two gids (must-link) but the tracklets overlap on one camera: never merged."""
    rng = np.random.default_rng(6)
    c = centers(3, seed=7, shared=0.0)
    f = centers(3, d=D_FACE, seed=8, shared=0.0)
    env = make_env(tmp_path, **{"consolidation.merge_threshold": 0.05})
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    a = make_tracklet(env, "c1", 0, 10, c[0], rng)
    b = make_tracklet(env, "c1", 5, 15, c[1], rng)  # simultaneous on the same camera
    eng.on_tracklet(a), eng.on_tracklet(b)
    ga, gb = gid_of(env, a), gid_of(env, b)
    assert ga != gb
    _attach_same_face(eng, rng, [(a, ga), (b, gb)], f[0])
    res = K.consolidate(eng.store)
    assert res.merges == []
    assert res.skipped_must_links == [(min(ga, gb), max(ga, gb))]
    assert gid_of(env, a) == ga and gid_of(env, b) == gb


def test_cannot_link_beats_must_link_chain(tmp_path):
    """A~B and B~C by face, but A and C overlap: B joins one of them, never both."""
    rng = np.random.default_rng(9)
    c = centers(3, seed=10, shared=0.0)
    f = centers(3, d=D_FACE, seed=11, shared=0.0)
    env = make_env(tmp_path, **{"consolidation.merge_threshold": 0.05})
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    a = make_tracklet(env, "c1", 0, 10, c[0], rng)
    cc = make_tracklet(env, "c1", 5, 15, c[1], rng)  # overlaps a on c1
    b = make_tracklet(env, "c2", 100, 110, c[2], rng)  # far in time, other camera
    for t in (a, cc, b):
        eng.on_tracklet(t)
    ga, gc, gb = gid_of(env, a), gid_of(env, cc), gid_of(env, b)
    assert len({ga, gb, gc}) == 3
    _attach_same_face(eng, rng, [(a, ga), (cc, gc), (b, gb)], f[0])
    K.consolidate(eng.store)
    assert gid_of(env, a) != gid_of(env, cc)
