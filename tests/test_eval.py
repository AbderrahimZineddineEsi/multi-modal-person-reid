"""Evaluation metrics on synthetic track points (CPU, no models, no video)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("motmetrics")

from reid.config import Config  # noqa: E402
from reid.eval import evaluate, mot, to_markdown  # noqa: E402
from reid.eval.hypothesis import processed_frame_mask  # noqa: E402
from reid.eval.retrieval import _ap, sample_queries  # noqa: E402
from reid.storage import db  # noqa: E402
from reid.types import CameraInfo, Tracklet, TrackPoint  # noqa: E402

CAMS = ["c1", "c2"]


def _box(person: int, f: int) -> tuple[float, float, float, float]:
    x = 100.0 * person + f  # people far apart, moving 1 px / frame
    return x, 50.0, x + 40.0, 170.0


def _gt(frames=range(0, 40), people_by_cam=None) -> pd.DataFrame:
    people_by_cam = people_by_cam or {"c1": [1, 2], "c2": [1, 2]}
    rows = [
        (cam, f, p, *_box(p, f)) for cam, ps in people_by_cam.items() for f in frames for p in ps
    ]
    return pd.DataFrame(rows, columns=["camera", "frame", "gid", "x1", "y1", "x2", "y2"])


def _hyp(mapping: dict[tuple[str, int], int], frames=range(0, 40)) -> pd.DataFrame:
    """mapping[(camera, person)] -> hypothesis id (the same box as the GT person)."""
    rows = [(c, f, h, *_box(p, f)) for (c, p), h in mapping.items() for f in frames]
    df = pd.DataFrame(rows, columns=["camera", "frame", "hid", "x1", "y1", "x2", "y2"])
    df["tid"] = df["hid"]
    return df


def _combined(gt, hyp, iou=0.5):
    return mot.summarize(mot.build_accumulator(gt, hyp, iou, CAMS, "hid", combined=True))


def test_perfect_hypothesis_scores_one():
    gt = _gt()
    hyp = _hyp({("c1", 1): 1, ("c1", 2): 2, ("c2", 1): 1, ("c2", 2): 2})
    m = _combined(gt, hyp)
    assert m["idf1"] == pytest.approx(1.0) and m["mota"] == pytest.approx(1.0)
    assert m["id_switches"] == 0 and m["precision"] == 1.0 and m["recall"] == 1.0
    per = mot.per_camera(gt, hyp, 0.5, CAMS, "hid")
    assert all(v["idf1"] == pytest.approx(1.0) for v in per.values())


def test_swapped_id_lowers_idf1_and_is_an_id_switch():
    gt = _gt()
    hyp = _hyp({("c1", 1): 1, ("c1", 2): 2, ("c2", 1): 1, ("c2", 2): 2})
    swapped = hyp.copy()  # from frame 20 on, the two people exchange hypothesis ids on c1
    late = (swapped["camera"] == "c1") & (swapped["frame"] >= 20)
    swapped.loc[late, "hid"] = swapped.loc[late, "hid"].map({1: 2, 2: 1})
    m = _combined(gt, swapped)
    assert 0.5 < m["idf1"] < 1.0 and m["id_switches"] >= 2
    assert m["mota"] < 1.0 and m["recall"] == 1.0  # boxes are still all matched


def test_cross_camera_consistency_is_penalized():
    gt = _gt()
    consistent = _hyp({("c1", 1): 1, ("c1", 2): 2, ("c2", 1): 1, ("c2", 2): 2})
    per_camera_ids = _hyp({("c1", 1): 1, ("c1", 2): 2, ("c2", 1): 3, ("c2", 2): 4})  # no linking
    merged = _hyp({("c1", 1): 1, ("c1", 2): 1, ("c2", 1): 1, ("c2", 2): 1})  # one gid for all
    good, split, same = (_combined(gt, h) for h in (consistent, per_camera_ids, merged))
    assert good["idf1"] == pytest.approx(1.0)
    # each camera is perfect on its own ...
    for cam, v in mot.per_camera(gt, per_camera_ids, 0.5, CAMS, "hid").items():
        assert v["idf1"] == pytest.approx(1.0), cam
    # ... but the global IDF1 drops to 0.5: a GT person can only match one of its two gids
    assert split["idf1"] == pytest.approx(0.5, abs=0.02)
    assert same["idf1"] < 0.6


def test_unidentified_tracklets_are_negative_and_never_match_each_other():
    gt = _gt()
    hyp = _hyp({("c1", 1): -5, ("c1", 2): -6, ("c2", 1): -7, ("c2", 2): -8})
    assert _combined(gt, hyp)["idf1"] == pytest.approx(0.5, abs=0.02)  # boxes found, ids useless


def test_fragmentation_and_purity():
    gt = _gt()
    hyp = _hyp({("c1", 1): 1, ("c1", 2): 2, ("c2", 1): 3, ("c2", 2): 2})  # person 1 split
    acc = mot.build_accumulator(gt, hyp, 0.5, CAMS, "hid", combined=True)
    pairs = mot.pair_counts(acc)
    fr = mot.fragmentation(pairs, n_gt_ids=2, n_gids=3, min_boxes=5)
    assert fr["gt_fragmented"] == 1 and fr["gt_one_gid"] == 1 and fr["gid_impure"] == 0
    assert fr["gids_per_gt_mean"] == pytest.approx(1.5) and fr["gid_purity"] == 1.0
    assert mot.majority_gt(pairs) == {1: 1, 2: 2, 3: 1}
    # two people sharing one gid: impure gid
    merged = _hyp({("c1", 1): 1, ("c1", 2): 1, ("c2", 1): 1, ("c2", 2): 1})
    pm = mot.pair_counts(mot.build_accumulator(gt, merged, 0.5, CAMS, "hid", combined=True))
    fm = mot.fragmentation(pm, 2, 1, min_boxes=5)
    assert fm["gid_impure"] == 1 and fm["gid_purity"] == pytest.approx(0.5)


def test_iou_threshold_and_empty_hypothesis():
    gt = _gt(people_by_cam={"c1": [1]})
    shifted = _hyp({("c1", 1): 1})
    shifted[["x1", "x2"]] += 30  # IoU 40*120 box shifted by 30 px -> 0.14
    assert mot.summarize(mot.build_accumulator(gt, shifted, 0.5, ["c1"], "hid"))["recall"] == 0.0
    empty = _hyp({("c1", 1): 1}).iloc[:0]
    m = mot.summarize(mot.build_accumulator(gt, empty, 0.5, ["c1"], "hid"))
    assert m["recall"] == 0.0 and m["idf1"] == 0.0


def test_processed_frame_mask_follows_runtime():
    cfg = Config.model_validate({"runtime": {"frame_stride": 2, "start_s": 1.0, "end_s": 2.0}})
    f = pd.Series(range(0, 100))
    kept = f[processed_frame_mask(f, cfg.runtime, fps=25.0)].tolist()
    assert kept[0] == 25 and kept[-1] == 49 and set(np.diff(kept)) == {2}


def test_ap_and_query_sampling():
    assert _ap(np.array([True, False, True])) == pytest.approx((1 + 2 / 3) / 2)
    assert _ap(np.array([False, False])) == 0.0
    gt = _gt(frames=range(0, 100), people_by_cam={"c1": [1], "c2": [1]})
    sampled = pd.DataFrame(  # a stored sample at c1 frame 50 covering person 1's box
        [("c1", 50, *_box(1, 50))], columns=["camera", "frame", "x1", "y1", "x2", "y2"]
    )
    q = sample_queries(gt, {"c1": (640, 360), "c2": (640, 360)}, sampled, per_id=6, min_height=80)
    assert len(q) == 6 and set(q["camera"]) == {"c1", "c2"} and q["held_out"].all()
    assert not ((q["camera"] == "c1") & ((q["frame"] - 50).abs() <= 10)).any()
    # people too small or cut by the frame edge are never queried
    assert sample_queries(gt, {"c1": (130, 360), "c2": (130, 360)}, sampled, per_id=3).empty


def _tracklet(cam, tid, gid, person, frames):
    pts = [TrackPoint(f, f / 25.0, *_box(person, f), 0.9, sampled=f % 10 == 0) for f in frames]
    return Tracklet(
        camera_id=cam, local_track_id=tid, start_frame=frames[0], end_frame=frames[-1],
        start_ts=frames[0] / 25.0, end_ts=frames[-1] / 25.0, gid=gid, points=pts,
    )  # fmt: skip


def test_evaluate_end_to_end_from_db(tmp_path):
    cfg = Config.model_validate({"paths": {"data_dir": str(tmp_path)}})
    conn = db.connect(tmp_path / "reid.db")
    for cid in CAMS:
        db.upsert_camera(conn, CameraInfo(cid, cid, "v.mp4", fps=25.0, width=640, height=360))
    rid = db.create_run(conn, cfg.to_json())
    frames = list(range(0, 40))
    db.insert_tracklets(
        conn,
        rid,
        [
            _tracklet("c1", 1, 1, 1, frames),
            _tracklet("c1", 2, 2, 2, frames),
            _tracklet("c2", 1, 1, 1, frames),
            _tracklet("c2", 2, None, 2, frames),  # unidentified
        ],
    )
    conn.close()
    gt = _gt()
    gt_path = tmp_path / "gt.csv"
    from reid.datasets import write_gt

    write_gt(gt, gt_path)
    res = evaluate(tmp_path, gt_path, retrieval=False)
    assert res["run_id"] == rid and res["counts"]["unidentified_tracklets"] == 1
    assert res["mtmc"]["recall"] == pytest.approx(1.0) and 0.5 < res["mtmc"]["idf1"] < 1.0
    assert res["per_camera"]["c1"]["idf1"] == pytest.approx(1.0)
    # per camera an unidentified tracklet is a consistent (negative) id, so it is not penalized
    assert res["per_camera"]["c2"]["idf1"] == pytest.approx(1.0)
    assert res["per_camera_tracklet"]["c2"]["idf1"] == pytest.approx(1.0)
    assert res["fragmentation"]["matched_boxes_unidentified"] > 0
    md = to_markdown(res)
    assert "CityFlow-style" in md and "Fragmentation" in md and "c2" in md


def test_cli_evaluate_saves_json_and_prints_markdown(tmp_path, capsys):
    import json

    from reid import cli
    from reid.datasets import write_gt

    cfg = Config.model_validate({"paths": {"data_dir": str(tmp_path)}})
    conn = db.connect(tmp_path / "reid.db")
    for cid in CAMS:
        db.upsert_camera(conn, CameraInfo(cid, cid, "v.mp4", fps=25.0, width=640, height=360))
    rid = db.create_run(conn, cfg.to_json())
    frames = list(range(40))
    db.insert_tracklets(
        conn, rid, [_tracklet("c1", 1, 1, 1, frames), _tracklet("c2", 1, 1, 1, frames)]
    )
    conn.close()
    write_gt(_gt(people_by_cam={"c1": [1], "c2": [1]}), tmp_path / "gt.csv")
    args = [
        "evaluate",
        "--data-dir",
        str(tmp_path),
        "--gt",
        str(tmp_path / "gt.csv"),
        "--no-retrieval",
    ]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "CityFlow-style" in out and "saved:" in out
    (saved,) = (tmp_path / "eval").glob("*.json")
    data = json.loads(saved.read_text())
    assert data["mtmc"]["idf1"] == pytest.approx(1.0) and "config" in data and "preset" in data
    assert cli.main(["evaluate", "--data-dir", str(tmp_path / "none"), "--gt", "x.csv"]) == 1
