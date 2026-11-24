import numpy as np

from reid.config import AssociationCfg
from reid.identity.associate import IdentityEngine
from reid.identity.constraints import Span
from reid.identity.gallery import GalleryBank
from reid.identity.udm import PoolItem, UnknownDataManager
from reid.storage import db
from tests.synth import centers, jitter, make_env, make_tracklet


def item(tid, cam, a, b, c, rng, n=10):
    e = jitter(c, n, 0.3, rng).astype(np.float32)
    return PoolItem(
        Span(tid, cam, a, b),
        e.mean(0) / np.linalg.norm(e.mean(0)),
        e,
        np.ones(n, np.float32),
        list(range(tid * 100, tid * 100 + n)),
    )


def test_cluster_test_new_vs_merge():
    rng = np.random.default_rng(0)
    c = centers(3, seed=3)
    cfg = AssociationCfg()
    bank = GalleryBank()
    for i, e in enumerate(jitter(c[0], 8, 0.3, rng)):
        bank.insert(1, e, 1.0, i)
    udm = UnknownDataManager(cfg)
    same = [item(1, "c1", 0, 1, c[0], rng)]
    other = [item(2, "c1", 0, 1, c[2], rng)]
    gid, d = udm.cluster_test(same, bank, set())
    assert gid == 1 and d <= cfg.udm_new_dist
    gid, d = udm.cluster_test(other, bank, set())
    assert gid is None and d > cfg.udm_new_dist  # farther than alpha: new identity
    assert udm.cluster_test(same, bank, {1})[0] is None  # blocked gid is never a target
    assert udm.cluster_test(same, GalleryBank(), set())[0] is None  # empty gallery


def test_clusters_respect_cannot_links_and_size_threshold():
    rng = np.random.default_rng(1)
    c = centers(2, seed=4)
    udm = UnknownDataManager(AssociationCfg(udm_min_samples=20))
    # two simultaneous tracklets of near-identical appearance in one camera + one later
    for it in (
        item(1, "c1", 0, 10, c[0], rng),
        item(2, "c1", 5, 15, c[0], rng),
        item(3, "c1", 30, 40, c[0], rng),
    ):
        udm.add(it)

    def violates(a, b):
        return a.camera_id == b.camera_id and max(a.start_ts, b.start_ts) < min(a.end_ts, b.end_ts)

    groups = udm.clusters(violates)
    ids = sorted(sorted(p.span.id for p in g) for g in groups)
    assert [1, 2] not in ids and all(not ({1, 2} <= set(g)) for g in ids)
    res = udm.process(GalleryBank(), violates, lambda p: set())
    # groups of 20 samples (two tracklets) are evaluated, a lone 10-sample tracklet is not
    assert [r.n_samples for r in res] == [20] and res[0].gid is None  # mint (empty gallery)
    assert {p.span.id for p in res[0].items} in ({1, 3}, {2, 3})
    assert len(udm) == 1 and udm.pool[0].span.id in (1, 2)  # the 10-sample leftover waits


def test_udm_minting_with_flush_nobody_vanishes(tmp_path):
    rng = np.random.default_rng(2)
    c = centers(4, seed=5)
    env = make_env(tmp_path, **{"association.bootstrap_q": 0, "consolidation.enabled": False})
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    ts = []
    # persons 0..2 each appear twice (2 x 10 samples = 20 -> evaluated online), person 3 once
    for p in range(3):
        ts.append(make_tracklet(env, "c1", 100 * p, 100 * p + 20, c[p], rng))
        ts.append(make_tracklet(env, "c1", 100 * p + 50, 100 * p + 70, c[p], rng))
    ts.append(make_tracklet(env, "c1", 500, 520, c[3], rng))
    for t in ts:
        assert eng.on_tracklet(t) is None  # empty gallery + no bootstrap: everything is pooled
    assert len(eng.udm) == 7
    eng.end_camera("c1")
    gids = [db.get_tracklet(env.conn, t.id).gid for t in ts]
    assert len(set(gids[:6])) == 3 and gids[0] == gids[1] and gids[2] == gids[3]
    assert gids[6] is None and len(eng.udm) == 1  # 10 samples < l = 20: still waiting
    report = eng.end_run(env.run_id)
    gids = [db.get_tracklet(env.conn, t.id).gid for t in ts]
    assert None not in gids and len(set(gids)) == 4  # flush: the lone tracklet got its own gid
    assert report.n_unidentified == 0 and report.n_identities == 4
