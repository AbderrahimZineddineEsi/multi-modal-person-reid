import numpy as np

from reid import analytics as A
from reid.identity.associate import IdentityEngine
from reid.storage import db
from tests.synth import centers, make_env, make_tracklet


def test_analytics(tmp_path):
    rng = np.random.default_rng(0)
    c = centers(3, seed=31, shared=0.0)
    env = make_env(tmp_path)
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    # person 0: c1 [0,10] -> c2 [40,55] -> c1 [100,104]; person 1: c1 [5,20]; one 1-sample blip
    spec = [("c1", 0, 10, 0), ("c2", 40, 55, 0), ("c1", 100, 104, 0), ("c1", 5, 20, 1)]
    tl = [make_tracklet(env, cam, a, b, c[p], rng) for cam, a, b, p in spec]
    blip = make_tracklet(env, "c3", 0, 1, c[2], rng, n_samples=1)
    for t in (*tl, blip):
        eng.on_tracklet(t)
    eng.end_run(env.run_id)
    g0 = db.get_tracklet(env.conn, tl[0].id).gid
    g1 = db.get_tracklet(env.conn, tl[3].id).gid
    steps = A.journey(env.conn, g0)
    assert [(s.camera_id, s.start_ts) for s in steps] == [("c1", 0), ("c2", 40), ("c1", 100)]
    assert [s.gap_s for s in steps] == [None, 30, 45]
    assert [round(s.duration_s) for s in steps] == [10, 15, 4]
    cams, mat = A.transitions(env.conn)
    i = {c_: k for k, c_ in enumerate(cams)}
    assert mat[i["c1"], i["c2"]] == 1 and mat[i["c2"], i["c1"]] == 1 and mat.sum() == 2
    dw = A.dwell_stats(env.conn)
    assert dw["c1"]["n"] == 3 and dw["c1"]["max_s"] == 15 and "c3" not in dw  # blip unidentified
    assert A.dwell_stats(env.conn, gid=g1)["c1"]["total_s"] == 15
    starts, counts = A.occupancy(env.conn, "c1", bin_s=10)
    assert starts[0] == 0 and counts.tolist()[:3] == [2, 2, 1] and counts[-1] == 1
    s = A.run_summary(env.conn, env.run_id)
    assert s["n_tracklets"] == 5 and s["n_people"] == 2
    assert s["identified_pct"] == 80.0 and s["unidentified_tracklets"] == [blip.id]
    assert s["tracklets_per_identity"] == {g0: 3, g1: 1} and s["mean_tracklets_per_identity"] == 2.0
    assert s["face_confirmed_pct"] == 0.0
