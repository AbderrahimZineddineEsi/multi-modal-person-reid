"""End-to-end synthetic scenario: 6 people, 3 cameras, time overlaps, faces for some."""

import shutil
from collections import Counter

import numpy as np
import pytest

from reid import analytics
from reid.identity.associate import IdentityEngine
from reid.storage import db
from reid.storage.vectors import VectorStore
from tests.synth import D_FACE, Env, centers, make_env, make_tracklet, unit

N_PEOPLE = 6
FACE_PEOPLE = {0, 1, 2}
# (person, camera, start_s, end_s, n_fragments)
VISITS = [
    (0, "c1", 0, 30, 2),
    (1, "c1", 10, 40, 1),  # overlap on c1
    (2, "c2", 5, 35, 1),
    (3, "c2", 20, 50, 1),  # overlap on c2
    (4, "c3", 0, 30, 1),
    (5, "c3", 10, 40, 1),  # overlap on c3
    (3, "c3", 60, 90, 2),
    (1, "c3", 120, 150, 1),
    (0, "c2", 100, 130, 1),
    (5, "c2", 160, 190, 1),
    (2, "c1", 150, 180, 1),
    (0, "c3", 200, 230, 1),
    (4, "c1", 220, 250, 1),
]


def view(c, cam, person, cos=0.9):
    """Camera-specific appearance of a person (domain gap): cosine ~cos to the base centre."""
    seed = 1000 + 10 * person + int(cam[1])
    r = np.random.default_rng(seed).normal(size=c.shape[-1])
    r = unit(r - (r @ c) * c)
    return c if cam == "c1" else unit(cos * c + np.sqrt(1 - cos**2) * r)


def make_visits(env: Env, cams: tuple[str, ...]):
    """Deterministic tracklets of the given cameras; returns [(tracklet, person)]."""
    c = centers(N_PEOPLE, seed=5, shared=0.3)
    f = centers(N_PEOPLE, d=D_FACE, seed=6, shared=0.0)
    out = []
    for k, (p, cam, a, b, nfrag) in enumerate(VISITS):
        if cam not in cams:
            continue
        rng = np.random.default_rng(k)
        edges = np.linspace(a, b, nfrag + 1)
        for j in range(nfrag):
            lo, hi = edges[j], edges[j + 1] - (1.0 if nfrag > 1 and j < nfrag - 1 else 0.0)
            t = make_tracklet(
                env,
                cam,
                float(lo),
                float(hi),
                view(c[p], cam, p),
                rng,
                noise=0.35,
                face_center=f[p] if (p in FACE_PEOPLE and cam in ("c1", "c2")) else None,
            )
            out.append((t, p))
    return out


def run_cameras(eng: IdentityEngine, items):
    for cam in sorted({t.camera_id for t, _ in items}):
        for t, _ in sorted((x for x in items if x[0].camera_id == cam), key=lambda x: x[0].end_ts):
            eng.on_tracklet(t)
        eng.end_camera(cam)


def metrics(env: Env, items):
    gids = {t.id: db.get_tracklet(env.conn, t.id).gid for t, _ in items}
    per_gid: dict[int, Counter] = {}
    for t, p in items:
        per_gid.setdefault(gids[t.id], Counter())[p] += 1
    ident = {g: c for g, c in per_gid.items() if g is not None}
    n_id = sum(sum(c.values()) for c in ident.values())
    purity = sum(max(c.values()) for c in ident.values()) / n_id
    gids_per_person = {p: len({gids[t.id] for t, q in items if q == p}) for p in range(N_PEOPLE)}
    return {
        "purity": purity,
        "n_gids": len(ident),
        "fragmentation": len(ident) / N_PEOPLE,
        "max_gids_per_person": max(gids_per_person.values()),
        "unidentified": sum(1 for g in gids.values() if g is None),
        "n_tracklets": len(items),
    }


@pytest.mark.parametrize("over", [{}, {"association.bootstrap_q": 0}], ids=["bootstrap", "udm"])
def test_multicamera_scenario_purity_and_fragmentation(tmp_path, capsys, over):
    env = make_env(tmp_path, **over)
    items = make_visits(env, ("c1", "c2", "c3"))
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    run_cameras(eng, items)
    report = eng.end_run(env.run_id)
    m = metrics(env, items)
    with capsys.disabled():
        print(f"\nSCENARIO METRICS {m} sources={report.by_source} merges={len(report.merges)}")
    assert m["unidentified"] == 0
    assert m["purity"] == 1.0  # no two people share a gid
    assert N_PEOPLE <= m["n_gids"] <= N_PEOPLE + 2  # few fragments
    assert m["max_gids_per_person"] <= 2
    # simultaneous same-camera tracklets never share a gid
    by_gid = {}
    for t, _ in items:
        by_gid.setdefault(db.get_tracklet(env.conn, t.id).gid, []).append(t)
    for ts in by_gid.values():
        for i, a in enumerate(ts):
            for b in ts[i + 1 :]:
                assert not (
                    a.camera_id == b.camera_id and a.start_ts < b.end_ts and b.start_ts < a.end_ts
                )
    s = analytics.run_summary(env.conn, env.run_id)
    assert s["identified_pct"] == 100.0 and s["n_people"] == m["n_gids"]
    assert s["face_confirmed_pct"] > 0
    # faces were used: some tracklets were assigned by face evidence
    assert report.by_source.get("face", 0) + report.by_source.get("consolidate", 0) > 0
    if over:  # UDM path: pooled tracklets were resolved by clustering / flush, not bootstrap
        assert {d.source for d in report.decisions} & {"udm", "flush"}


def test_reload_from_data_dir_gives_identical_decisions(tmp_path):
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    env_a = make_env(a_dir)
    eng_a = IdentityEngine(env_a.cfg, env_a.conn, env_a.vectors)
    run_cameras(eng_a, make_visits(env_a, ("c1", "c2")))
    eng_a.end_run(env_a.run_id)
    env_a.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    shutil.copytree(a_dir, b_dir)  # snapshot of the data dir after "session 1"

    # session 2 on the snapshot: a brand new engine that only has data_dir to go on
    conn_b = db.connect(b_dir / "reid.db")
    env_b = Env(
        env_a.cfg.model_copy(deep=True), conn_b, VectorStore(b_dir / "vectors"), env_a.run_id
    )
    env_b.cfg.paths.data_dir = str(b_dir)
    eng_b = IdentityEngine(env_b.cfg, conn_b, env_b.vectors)
    assert eng_b.store.bank.gids() == eng_a.store.bank.gids()
    for g in eng_a.store.bank.gids():  # identical galleries
        ga, gb = eng_a.store.bank.galleries[g], eng_b.store.bank.galleries[g]
        assert ga.ids == gb.ids
        np.testing.assert_array_equal(ga.embs, gb.embs)

    n1 = len(eng_a.decisions)
    items_a = make_visits(env_a, ("c3",))
    items_b = make_visits(env_b, ("c3",))
    run_cameras(eng_a, items_a)
    run_cameras(eng_b, items_b)
    eng_a.end_run(env_a.run_id)
    eng_b.end_run(env_b.run_id)
    ga = [db.get_tracklet(env_a.conn, t.id).gid for t, _ in items_a]
    gb = [db.get_tracklet(conn_b, t.id).gid for t, _ in items_b]
    assert ga == gb and None not in ga
    assert eng_a.decisions[n1:] == eng_b.decisions  # same gid, source, score, note per tracklet
    pa, pb = env_a.vectors.get("body_proto"), env_b.vectors.get("body_proto")
    np.testing.assert_allclose(pa.vectors(), pb.vectors(), atol=1e-6)
    # a run accumulates: identities from session 1 are re-used in session 2 (same people on c3)
    assert len(db.list_identities(conn_b)) <= N_PEOPLE + 2
