import numpy as np

from reid.config import AssociationCfg
from reid.identity import prototypes as P
from reid.identity.associate import IdentityEngine
from tests.synth import centers, jitter, make_env, make_tracklet


def three_modes(rng, n=30):
    c = centers(3, seed=11, shared=0.0)
    e = np.concatenate([jitter(ci, n, 0.15, rng) for ci in c])
    return e.astype(np.float32), c


def distinct(rows, tol=1e-4):
    return all(
        np.linalg.norm(rows[i] - rows[j]) > tol
        for i in range(len(rows))
        for j in range(i + 1, len(rows))
    )


def test_alpha_fps_returns_k_distinct_covering_modes():
    rng = np.random.default_rng(0)
    e, c = three_modes(rng)
    q = np.full(len(e), 0.7)  # constant quality: normalized term is inert, diversity decides
    sel = P.alpha_fps_select(e, q, k=3, alpha=0.5)
    assert len(sel) == 3 == len(set(sel))
    assert {int(np.argmax(c @ e[i])) for i in sel} == {0, 1, 2}  # one prototype per mode


def test_alpha_fps_quality_still_matters_and_small_n():
    rng = np.random.default_rng(1)
    e, _ = three_modes(rng, n=10)
    q = np.random.default_rng(2).uniform(0.4, 0.9, len(e))
    assert P.alpha_fps_select(e, q, 3, 0.5)[0] == int(np.argmax(q))  # seeded by best quality
    assert P.alpha_fps_select(e[:2], q[:2], 3, 0.5) == [0, 1]  # fewer samples than K


def test_gcp_returns_k_distinct_normalized_prototypes():
    rng = np.random.default_rng(3)
    e, c = three_modes(rng)
    p = P.gcp_prototypes(e, k=3, alpha=0.5)
    assert p.shape == (3, e.shape[1]) and distinct(p)
    np.testing.assert_allclose(np.linalg.norm(p, axis=1), 1.0, atol=1e-5)
    # first prototype is the centroid, the others were pulled toward it by alpha
    cen = e.mean(0) / np.linalg.norm(e.mean(0))
    np.testing.assert_allclose(p[0], cen, atol=1e-5)
    p0 = P.gcp_prototypes(e, k=3, alpha=0.0)  # alpha=0: plain farthest points
    assert (p0[1:] @ cen).mean() < (p[1:] @ cen).mean()


def test_compute_prototypes_dispatch():
    rng = np.random.default_rng(4)
    e, _ = three_modes(rng, n=10)
    q = np.full(len(e), 0.8, np.float32)
    for m in ("alpha_fps", "gcp"):
        p = P.compute_prototypes(e, q, AssociationCfg(proto_method=m, proto_k=3))
        assert len(p) == 3 and distinct(p)


def test_rebuild_prototypes_idempotent_and_both_methods(tmp_path):
    rng = np.random.default_rng(5)
    c = centers(2, seed=12)
    for method in ("alpha_fps", "gcp"):
        env = make_env(tmp_path / method, **{"association.proto_method": method})
        eng = IdentityEngine(env.cfg, env.conn, env.vectors)
        for i in range(3):
            eng.on_tracklet(make_tracklet(env, "c1", 20 * i, 20 * i + 10, c[0], rng))
        eng.on_tracklet(make_tracklet(env, "c1", 100, 110, c[1], rng))
        eng.end_run(env.run_id)
        col = env.vectors.get("body_proto")
        labels = col.labels
        assert sorted(set(labels.tolist())) == [1, 2] and (labels == 1).sum() <= 3
        before = col.vectors().copy()
        assert P.rebuild_prototypes(env.conn, env.vectors, env.cfg) == 2
        np.testing.assert_allclose(env.vectors.get("body_proto").vectors(), before, atol=1e-6)
