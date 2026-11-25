"""AI City benchmark plumbing: scoring wrapper, fetcher extras, benchmark.json, notebook helper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reid.cli import build_parser
from reid.datasets import aicity24 as ac
from reid.datasets.gt import read_gt, write_gt
from reid.eval import aicity_official as off
from reid.eval.aicity_bench_io import read_benchmark, scale_gt_csv, update_benchmark
from reid.eval.aicity_pack import is_complete
from reid.eval.aicity_score import _concat, collect_scenes, results_markdown
from reid.notebook import CommandFailed, Job, run_parallel
from reid.storage import db

SAMPLE_STDOUT = """\
Total runtime: 187.0887589454651 seconds.
HOTA: 49.2825%
DetA: 49.1998%
AssA: 49.3655%
LocA: 77.0546%
"""


def test_parse_scores_and_missing():
    assert off.parse_scores(SAMPLE_STDOUT) == {
        "HOTA": 49.2825, "DetA": 49.1998, "AssA": 49.3655, "LocA": 77.0546,
    }  # fmt: skip
    noisy = "1 eval_sequence(scene_075)\nHOTA: tracker pedestrian  12 3\n" + SAMPLE_STDOUT
    assert off.parse_scores(noisy)["HOTA"] == 49.2825  # table lines without a % do not match
    with pytest.raises(ValueError, match="AssA"):
        off.parse_scores("HOTA: 1.0%\nDetA: 2.0%\nLocA: 3.0%\n")


def test_write_scene_map(tmp_path: Path):
    p = off.write_scene_map({"scene_075": [700, 701], "scene_078": [3]}, tmp_path / "m" / "s.json")
    assert json.loads(p.read_text()) == [
        {"scene_name": "scene_075", "camera_ids": [700, 701]},
        {"scene_name": "scene_078", "camera_ids": [3]},
    ]


def test_unsafe_paths_are_copied(tmp_path: Path):
    ok = tmp_path / "pred.txt"
    ok.write_text("x")
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    assert off._safe(ok, tmp) == ok.resolve()
    odd = tmp_path / "my dir" / "pred file.txt"  # the evaluator rejects spaces
    odd.parent.mkdir()
    odd.write_text("y")
    safe = off._safe(odd, tmp)
    assert off._SAFE_PATH.match(str(safe)) and safe.read_text() == "y"


def test_results_markdown_has_context_sentence():
    r = {
        "preset": "balanced", "date": "2026-11-25",
        "scenes": {"075": {"cameras": 16, "HOTA": 50.0, "DetA": 60.0, "AssA": 45.0, "LocA": 80.0}},
        "overall": {"HOTA": 50.0, "DetA": 60.0, "AssA": 45.0, "LocA": 80.0},
        "leaderboard_2024": off.LEADERBOARD_2024,
        "runs": {"075": {"frame_stride": 2, "gpu": "Tesla T4", "fps": 33.5,
                         "metrics_2d": {"idf1": 0.8, "mota": 0.7, "identities": 12, "gt_ids": 11,
                                        "retrieval": {"rank1": 0.9}}}},
    }  # fmt: skip
    md = results_markdown(r)
    assert "| 075 | 16 | 50.0 | 60.0 | 45.0 | 80.0 |" in md and "Tesla T4" in md
    assert "71.94" in md and "all 30 test scenes" in md and "not a like-for-like" in md
    assert "80.0%" in md and "90.0%" in md  # IDF1 and Rank-1


def _env(ws: Path, scene: str, preset: str, *, preds: bool = True) -> Path:
    env = ws / f"aicity24_{scene}_{preset}"
    (env / "aicity").mkdir(parents=True)
    meta = {"scene": scene, "camera_numbers": {"cam_0701": 701, "cam_0700": 700}}
    (env / "aicity" / "meta.json").write_text(json.dumps(meta))
    if preds:
        (env / "aicity" / "predictions.txt").write_text("700 1 0 1 2 3 4 0.5 0.5")  # no newline
        (env / "aicity" / "gt_official.txt").write_text("700 1 0 1 2 3 4 0.5 0.5\n")
    return env


def test_collect_and_concat(tmp_path: Path):
    _env(tmp_path, "078", "balanced")
    _env(tmp_path, "075", "balanced")
    _env(tmp_path, "075", "lite")  # other preset
    _env(tmp_path, "080", "balanced", preds=False)  # not exported yet
    scenes = __import__("reid.eval.aicity_score", fromlist=["x"]).collect_scenes(
        tmp_path, "balanced"
    )
    assert [s.scene for s in scenes] == ["075", "078"] and scenes[0].cameras == [700, 701]
    out = _concat([s.files_dir / "predictions.txt" for s in scenes], tmp_path / "all.txt")
    assert out.read_text().count("\n") == 2  # a missing final newline does not glue two rows
    assert collect_scenes(tmp_path, "max") == []


def test_is_complete_requires_all_artifacts(tmp_path: Path):
    name = "e"
    env = tmp_path / name
    assert not is_complete(tmp_path, name)
    conn = db.connect(env / "reid.db")
    rid = db.create_run(conn, "{}")
    db.set_run_status(conn, rid, "done")
    conn.close()
    assert not is_complete(tmp_path, name)
    (env / "aicity").mkdir()
    (env / "aicity" / "predictions.txt").write_text("")
    assert not is_complete(tmp_path, name)  # gt.csv is written last
    (env / "gt.csv").write_text("")
    assert is_complete(tmp_path, name)


def test_packed_complete_needs_zip_and_results(tmp_path: Path):
    from reid.eval.aicity_pack import packed_complete, results_path, zip_path

    assert not packed_complete(tmp_path, "e")
    zip_path(tmp_path, "e").parent.mkdir(parents=True)
    zip_path(tmp_path, "e").write_bytes(b"z")
    assert not packed_complete(tmp_path, "e")  # results folder missing
    results_path(tmp_path, "e").mkdir(parents=True)
    (results_path(tmp_path, "e") / "predictions.txt").write_text("")
    assert packed_complete(tmp_path, "e") and is_complete(tmp_path, "e")


def test_benchmark_json_merges(tmp_path: Path):
    update_benchmark(tmp_path, "e1", {"fps": 30.0, "timings_s": {"process": 1.0}})
    update_benchmark(tmp_path, "e2", {"fps": 20.0})
    update_benchmark(tmp_path, "e1", {"videos_720p_s": 5.0})
    runs = read_benchmark(tmp_path)["runs"]
    assert runs["e1"] == {"fps": 30.0, "timings_s": {"process": 1.0}, "videos_720p_s": 5.0}
    assert runs["e2"]["fps"] == 20.0
    assert read_benchmark(tmp_path / "missing") == {}


def test_scale_gt_csv(tmp_path: Path):
    gt = pd.DataFrame(
        {"camera": ["a", "a", "b"], "frame": [0, 1, 0], "gid": [1, 1, 2],
         "x1": [100.0, 200, 10], "y1": [60.0, 120, 10],
         "x2": [300.0, 400, 20], "y2": [180.0, 240, 30]}
    )  # fmt: skip
    write_gt(gt, tmp_path / "gt.csv")
    scale_gt_csv(
        tmp_path / "gt.csv", tmp_path / "out.csv",
        {"a": (1920, 1080), "b": (1000, 1000)}, {"a": (1280, 720), "b": (500, 500)},
    )  # fmt: skip
    out = read_gt(tmp_path / "out.csv").set_index(["camera", "frame"])
    assert np.allclose(
        out.loc[("a", 0)][["x1", "y1", "x2", "y2"]], [66.7, 40.0, 200.0, 120.0], atol=0.1
    )
    assert np.allclose(out.loc[("b", 0)][["x1", "y2"]], [5.0, 15.0])


FILES = [
    "MTMC_Tracking_2024/test/scene_075/ground_truth.txt",
    "MTMC_Tracking_2024/test/scene_075/calibration_2025_format.json",
    "MTMC_Tracking_2024/test/scene_075/camera_0700/video.mp4",
    "MTMC_Tracking_2024/test/scene_075/camera_0701/video.mp4",
]
H_W2I = [[100.0, 0.0, 500.0], [0.0, 100.0, 300.0], [0.0, 0.0, 1.0]]


def test_full_scene_fetch_keeps_originals_and_writes_calibration(tmp_path: Path, monkeypatch):
    def fake_download(repo_path, local_dir):
        p = Path(local_dir) / repo_path
        p.parent.mkdir(parents=True, exist_ok=True)
        if repo_path.endswith("ground_truth.txt"):
            rows = [f"{c} 1 {f} 100 200 50 150 1.0 2.0" for c in (700, 701, 702) for f in range(5)]
            p.write_text("\n".join(rows) + "\n")
        elif repo_path.endswith(".json"):
            sensors = [{"type": "camera", "id": f"Camera_{c:04d}", "homography": H_W2I}
                       for c in (700, 701)]  # fmt: skip
            p.write_text(json.dumps({"sensors": sensors}))
        else:
            p.write_bytes(b"original-" + repo_path.encode())
        return p

    monkeypatch.setattr(ac, "probe_video", lambda p: ac.VideoInfo(1920, 1080, 30.0, 5))

    def no_encode(*a, **k):
        raise AssertionError("must not re-encode")

    monkeypatch.setattr(ac, "trim_video", no_encode)
    out = tmp_path / "scene_075"
    res = ac.fetch_aicity24(
        75, cameras=0, minutes=None, height=None, out=out, yaml_path=out / "cameras.yaml",
        files=FILES, download=fake_download,
    )  # fmt: skip
    assert res.cameras == ["cam_0700", "cam_0701"]
    assert (out / "cam_0700.mp4").read_bytes().startswith(b"original-")  # moved, not re-encoded
    assert not (out / "raw").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["camera_numbers"] == {"cam_0700": 700, "cam_0701": 701}
    assert meta["src_sizes"]["cam_0700"] == [1920, 1080] and meta["gt_official_rows"] == 10
    assert (out / "calibration.json").exists()
    assert len((out / "gt_official.txt").read_text().splitlines()) == 10  # camera 702 dropped
    from reid.config import load_cameras

    cams = load_cameras(out / "cameras.yaml").cameras
    h = np.asarray(cams[0].homography)  # image -> floor: inverse of the calibration, [2][2] == 1
    assert np.allclose(h @ np.asarray(H_W2I), np.eye(3), atol=1e-9) and h[2, 2] == 1.0


def test_fetch_with_separate_raw_dir_cleans_only_the_scene(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ac, "probe_video", lambda p: ac.VideoInfo(1920, 1080, 30.0, 5))
    monkeypatch.setattr(ac, "trim_video", lambda *a, **k: ac.VideoInfo(1920, 1080, 30.0, 5))

    def fake_download(repo_path, local_dir):
        p = Path(local_dir) / repo_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            "700 1 0 1 2 3 4 0 0\n"
            if repo_path.endswith(".txt")
            else '{"sensors": []}'
            if repo_path.endswith(".json")
            else "x"
        )
        return p

    raw = tmp_path / "shared_raw"
    (raw / "keep.txt").parent.mkdir()
    (raw / "keep.txt").write_text("other scene's download")
    ac.fetch_aicity24(
        75, cameras=0, minutes=1, out=tmp_path / "o", raw_dir=raw, files=FILES,
        download=fake_download, yaml_path=tmp_path / "c.yaml",
    )  # fmt: skip
    assert (raw / "keep.txt").exists() and not (raw / ac.ROOT / "test" / "scene_075").exists()


def test_trim_video_command_flags(monkeypatch, tmp_path: Path):
    from reid.datasets import videos

    seen = {}
    monkeypatch.setattr(videos.shutil, "which", lambda n: f"/usr/bin/{n}")
    monkeypatch.setattr(videos.subprocess, "run", lambda cmd, **k: seen.setdefault("cmd", cmd))
    monkeypatch.setattr(videos, "probe_video", lambda p: videos.VideoInfo(1280, 720, 30.0, 1))
    videos.trim_video("in.mp4", tmp_path / "o.mp4", None, 720, crf=28, nice=10, threads=2)
    cmd = seen["cmd"]
    assert cmd[:3] == ["nice", "-n", "10"] and "-threads" in cmd and "scale=-2:720" in cmd
    assert cmd[cmd.index("-crf") + 1] == "28" and "-t" not in cmd


def test_run_parallel_waits_for_all_and_reports(tmp_path: Path, capsys):
    ok = [sys.executable, "-c", "print('alpha progress')"]
    codes = run_parallel(
        [Job("a", ok, tmp_path / "a.log"), Job("b", ok, tmp_path / "b.log", {"X": "1"})],
        poll_s=0.3,
    )
    out = capsys.readouterr().out
    assert codes == {"a": 0, "b": 0} and "alpha progress" in out and "failed: none" in out
    # a failing job does not stop the slow one: both are waited for
    bad = [sys.executable, "-c", "import sys; print('boom'); sys.exit(4)"]
    slow = [sys.executable, "-c", "import time; time.sleep(1.5); print('slow finished')"]
    jobs = [Job("slow", slow, tmp_path / "s.log"), Job("bad", bad, tmp_path / "x.log")]
    with pytest.raises(CommandFailed, match="bad exited with code 4"):
        run_parallel(jobs, poll_s=0.5)
    assert "slow finished" in (tmp_path / "s.log").read_text()  # it was not terminated
    assert "boom" in capsys.readouterr().out  # the failing job's log is shown once
    codes = run_parallel(jobs, poll_s=0.5, raise_on_failure=False)  # notebook mode: no raise
    assert codes == {"slow": 0, "bad": 4}


def test_run_parallel_prints_only_the_marker_line(tmp_path: Path, capsys):
    code = (
        "import time; print('noise 1'); print('ETA | scene 075 | old'); "
        "print('ETA | scene 075 | latest'); print('noise 2'); time.sleep(1.2)"
    )
    run_parallel(
        [Job("g", [sys.executable, "-c", code], tmp_path / "g.log")], poll_s=0.6, marker="ETA |"
    )
    out = capsys.readouterr().out
    assert "    ETA | scene 075 | latest" in out
    assert "    ETA | scene 075 | old" not in out and "    noise 1" not in out


def test_try_sh_returns_code():
    from reid.notebook import try_sh

    assert try_sh([sys.executable, "-c", "pass"]) == 0
    assert try_sh([sys.executable, "-c", "import sys; sys.exit(3)"]) == 3


def test_run_parallel_passes_environment(tmp_path: Path):
    cmd = [sys.executable, "-c", "import os; print('GPU=' + os.environ['CUDA_VISIBLE_DEVICES'])"]
    run_parallel([Job("g1", cmd, tmp_path / "g.log", {"CUDA_VISIBLE_DEVICES": "1"})], poll_s=0.3)
    assert "GPU=1" in (tmp_path / "g.log").read_text()


def test_cli_parses_benchmark_commands():
    p = build_parser()
    a = p.parse_args(["benchmark", "aicity24", "run", "--scenes", "75,078", "--workspace", "w",
                      "--preset", "lite", "--set", "a.b=1", "--frame-stride", "3"])  # fmt: skip
    assert a.scenes == ["075", "078"] and a.preset == "lite" and a.frame_stride == 3
    assert a.keep_videos == "720" and a.bench_cmd == "run" and a.set == ["a.b=1"]
    b = p.parse_args(["benchmark", "aicity24", "score", "--workspace", "w"])
    assert b.bench_cmd == "score" and b.preset == "balanced" and b.num_cores == 4
    d = p.parse_args(["dataset", "fetch", "aicity24", "--out", "o", "--camera-ids", "all"])
    assert d.camera_ids == "all"


def test_run_scene_flow_resume_and_outputs(tmp_path: Path, monkeypatch):
    """The per-scene orchestration with fakes for fetch/process/evaluate (no models, no network)."""
    from concurrent.futures import ThreadPoolExecutor

    import reid.datasets.aicity24 as dsmod
    from reid.config import load_config
    from reid.eval import aicity_bench as bench
    from reid.eval import aicity_steps as steps
    from reid.geometry import invert_homography, scale_homography
    from reid.types import CameraInfo, Tracklet, TrackPoint

    calls = {"fetch": 0, "process": 0}

    def fake_fetch(scene, *, out, **kw):
        calls["fetch"] += 1
        out.mkdir(parents=True)
        meta = {"scene": scene, "frames": 100, "camera_numbers": {"cam_0700": 700},
                "src_sizes": {"cam_0700": [1920, 1080]}}  # fmt: skip
        (out / "meta.json").write_text(json.dumps(meta))
        (out / "cameras.yaml").write_text("cameras: []\n")
        (out / "cam_0700.mp4").write_bytes(b"src")
        (out / "gt_official.txt").write_text("700 1 0 10 20 30 40 1.0 2.0\n")
        (out / "calibration.json").write_text("{}")
        gt = pd.DataFrame({"camera": ["cam_0700"], "frame": [0], "gid": [1], "x1": [300.0],
                           "y1": [150.0], "x2": [600.0], "y2": [450.0]})  # fmt: skip
        write_gt(gt, out / "gt.csv")

    def fake_process(opts, scene_dir, env):  # noqa: ARG001
        calls["process"] += 1
        cfg = load_config(None, {"paths.data_dir": str(env), "runtime.frame_stride": 2})
        env.mkdir(parents=True, exist_ok=True)
        conn = db.connect(env / "reid.db")
        hom = scale_homography(invert_homography(H_W2I), (1920, 1080), (1280, 720))
        db.upsert_camera(conn, CameraInfo("cam_0700", "c", str(scene_dir / "cam_0700.mp4"), 30.0,
                                          1280, 720, homography=hom))  # fmt: skip
        rid = db.create_run(conn, cfg.to_json())
        t = Tracklet("cam_0700", 1, 0, 2, 0.0, 0.1, gid=5)
        t.points = [TrackPoint(f, f / 30, 200.0, 100.0, 400.0, 300.0, 0.9) for f in (0, 2)]
        db.insert_tracklets(conn, rid, [t])
        db.set_run_status(conn, rid, "done")
        conn.close()
        return {"run_id": rid, "fps": 33.0, "frames_processed": 50}

    def fake_eval(env, gt, preset):
        (env / "eval").mkdir(exist_ok=True)
        (env / "eval" / "x.json").write_text("{}")
        return {"idf1": 0.9, "identities": 1, "gt_ids": 1}

    def fake_encode(sources, env, height=720, crf=28):
        (Path(env) / "videos").mkdir(parents=True, exist_ok=True)
        for cid in sources:
            (Path(env) / "videos" / f"{cid}.mp4").write_bytes(b"720p")
        return {}

    monkeypatch.setattr(dsmod, "fetch_aicity24", fake_fetch)
    monkeypatch.setattr(bench, "process_scene", fake_process)
    monkeypatch.setattr(bench, "evaluate_2d", fake_eval)
    monkeypatch.setattr(steps, "encode_videos", fake_encode)
    opts = bench.BenchOptions(
        workspace=tmp_path / "ws", raw_dir=tmp_path / "raw", preset="lite", pack=False
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        status, fut = bench.run_scene("075", opts, pool)
        assert status == "queued" and fut is not None
        assert fut.result() == "ok"
    env = tmp_path / "ws" / "aicity24_075_lite"
    pred = (env / "aicity" / "predictions.txt").read_text().split()
    assert pred[:3] == ["700", "5", "0"] and len(pred) == 27  # 3 rows of 9
    assert (env / "aicity" / "gt_official.txt").exists() and (env / "env.json").exists()
    assert (env / "videos" / "cam_0700.mp4").exists() and not (
        tmp_path / "raw" / "scene_075"
    ).exists()
    conn = db.connect(env / "reid.db")
    assert db.get_camera(conn, "cam_0700").video_path == "videos/cam_0700.mp4"
    conn.close()
    gt = read_gt(env / "gt.csv")  # source pixels -> the stored 1280x720 camera size
    assert np.allclose(gt.iloc[0][["x1", "y1", "x2", "y2"]].astype(float), [200, 100, 400, 300])
    run = read_benchmark(tmp_path / "ws")["runs"]["aicity24_075_lite"]
    assert run["fps"] == 33.0 and run["export"]["rows"] == 3 and "videos_720p_s" in run
    assert run["metrics_2d"]["idf1"] == 0.9 and run["timings_s"]["export"] >= 0
    assert is_complete(tmp_path / "ws", env.name) and run["status"] == "ok"
    assert bench.run_scene("075", opts) == ("skipped", None)  # nothing fetched or processed again
    assert calls == {"fetch": 1, "process": 1}


def test_balance_splits_by_weight():
    from reid.notebook import balance

    w = {"075": 16, "078": 16, "082": 16, "089": 15}
    assert balance(["075", "078", "082", "089"], w, 2) == [["075", "082"], ["078", "089"]]
    assert balance(["a", "b", "c", "d"], {}, 2) == [["a", "c"], ["b", "d"]]  # round-robin
    assert balance(["a", "b"], {}, 4) == [["a"], ["b"]]  # empty groups dropped
    assert balance(["a", "b", "c"], {"c": 10}, 2) == [["c"], ["a", "b"]]
    assert balance([], {}, 2) == []
