"""Crash safety, packing, ETA tracker and smoke projection of the AI City benchmark (CPU only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reid.config import load_config
from reid.datasets.gt import write_gt
from reid.eval import aicity_bench as bench
from reid.eval import aicity_steps as steps
from reid.eval.aicity_bench_io import read_benchmark, summary_text
from reid.eval.aicity_pack import packed_complete, results_path, zip_path
from reid.eval.aicity_progress import ProgressTracker, fmt_remaining
from reid.eval.aicity_projection import finish_smoke, project, projection_lines
from reid.eval.aicity_score import collect_scenes
from reid.geometry import invert_homography, scale_homography
from reid.storage import db
from reid.types import CameraInfo, Tracklet, TrackPoint

H_W2I = np.array([[100.0, 0.0, 500.0], [0.0, 100.0, 300.0], [0.0, 0.0, 1.0]])


class Flow:
    """Fakes for fetch / process / evaluate / encode so ``run_scene`` runs without models."""

    def __init__(self, monkeypatch, fail: dict[str, str] | None = None) -> None:
        import reid.datasets.aicity24 as dsmod

        self.fail = fail or {}  # scene -> step that raises
        self.fetch_kwargs: list[dict] = []
        monkeypatch.setattr(dsmod, "fetch_aicity24", self.fetch)
        monkeypatch.setattr(bench, "process_scene", self.process)
        monkeypatch.setattr(bench, "evaluate_2d", self.evaluate)
        monkeypatch.setattr(steps, "encode_videos", self.encode)

    def _maybe_fail(self, scene: str, step: str) -> None:
        if self.fail.get(scene) == step:
            raise RuntimeError(f"boom in {step} of {scene}")

    def fetch(self, scene, *, out, **kw):
        self.fetch_kwargs.append(kw)
        self._maybe_fail(scene, "fetch")
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

    def process(self, opts, scene_dir, env):
        scene = scene_dir.name.removeprefix("scene_")
        self._maybe_fail(scene, "process")
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
        (env / "cache").mkdir(exist_ok=True)
        (env / "cache" / f"run_{rid}.json").write_text("{}")
        if opts.tracker:
            opts.tracker.progress("cam_0700", 100, 100)
        return {"run_id": rid, "fps": 33.0, "frames_processed": 50, "wall_s": 3.0,
                "model_load_s": 1.0}  # fmt: skip

    def evaluate(self, env, gt, preset):
        scene = Path(gt).parent.name.removeprefix("scene_")
        self._maybe_fail(scene, "eval_2d")
        (env / "eval").mkdir(exist_ok=True)
        (env / "eval" / "x.json").write_text("{}")
        return {"idf1": 0.9, "identities": 1, "gt_ids": 1}

    def encode(self, sources, env, height=720, crf=28):
        for cid in sources:
            self._maybe_fail(Path(env).name.split("_")[1], "videos")
            (Path(env) / "videos").mkdir(parents=True, exist_ok=True)
            (Path(env) / "videos" / f"{cid}.mp4").write_bytes(b"720p")
        return {}


def _opts(tmp_path: Path, **kw) -> bench.BenchOptions:
    return bench.BenchOptions(
        workspace=tmp_path / "ws", raw_dir=tmp_path / "raw", preset="lite", **kw
    )


def test_a_failed_scene_is_recorded_and_the_next_one_runs(tmp_path: Path, monkeypatch):
    Flow(monkeypatch, fail={"075": "eval_2d", "078": "fetch", "082": "videos"})
    opts = _opts(tmp_path, pack=False)
    statuses = bench.run_scenes(["075", "078", "082", "086"], opts)
    assert statuses == {"075": "failed", "078": "failed", "082": "failed", "086": "ok"}
    runs = read_benchmark(opts.workspace)["runs"]
    assert runs["aicity24_075_lite"]["status"] == "failed"
    assert runs["aicity24_075_lite"]["step"] == "eval_2d"
    assert "boom in eval_2d" in runs["aicity24_075_lite"]["error"]
    assert runs["aicity24_078_lite"]["step"] == "fetch"
    assert runs["aicity24_082_lite"]["step"] == "videos"
    assert runs["aicity24_086_lite"]["status"] == "ok"
    for s in ("075", "078", "082", "086"):  # raw scene folders are cleaned up in every case
        assert not (tmp_path / "raw" / f"scene_{s}").exists()
    log = (opts.workspace / "aicity24_075_lite" / "logs" / "process.log").read_text()
    assert "Traceback" in log and "boom in eval_2d" in log  # traceback in the scene's log
    text = summary_text(opts.workspace, ["075", "086", "090"], "lite")
    assert "scene 075: FAILED in step eval_2d" in text and "scene 086: ok" in text
    assert "scene 090: not run" in text


def test_exit_code_is_nonzero_only_when_every_scene_failed(tmp_path: Path, monkeypatch):
    from reid.cli import main

    Flow(monkeypatch, fail={"075": "process", "078": "process"})
    ws = tmp_path / "ws"
    base = ["benchmark", "aicity24", "run", "--workspace", str(ws), "--preset", "lite",
            "--raw-dir", str(tmp_path / "raw"), "--no-pack"]  # fmt: skip
    assert main([*base, "--scenes", "075,078"]) == 1  # every scene failed
    assert main([*base, "--scenes", "075,086"]) == 0  # one succeeded
    assert main([*base, "--scenes", "086"]) == 0  # all already complete


def test_pack_roundtrip_and_resume(tmp_path: Path, monkeypatch, capsys):
    from reid.cli import main
    from reid.storage.portable import find_video

    Flow(monkeypatch)
    opts = _opts(tmp_path)  # pack on by default
    assert bench.run_scenes(["075"], opts) == {"075": "ok"}
    ws, name = opts.workspace, "aicity24_075_lite"
    assert packed_complete(ws, name) and not (ws / name).exists()  # unpacked folder is gone
    res = results_path(ws, name)
    assert {p.name for p in res.iterdir()} >= {
        "predictions.txt", "gt_official.txt", "meta.json", "eval_x.json", "process.log",
        "benchmark_entry.json",
    }  # fmt: skip
    assert json.loads((res / "benchmark_entry.json").read_text())["status"] == "ok"
    assert [s.scene for s in collect_scenes(ws, "lite")] == ["075"]  # score works without unzipping
    assert bench.run_scene("075", opts) == ("skipped", None)
    # `reid env import` restores a usable environment: db, 720p video, crops, eval, 720p gt.csv
    out = tmp_path / "pc"
    assert main(["env", "import", str(zip_path(ws, name)), "--workspace", str(out)]) == 0
    env = out / name
    assert (env / "reid.db").exists() and (env / "gt.csv").exists() and (env / "eval").is_dir()
    conn = db.connect(env / "reid.db")
    cam = db.get_camera(conn, "cam_0700")
    assert len(db.get_tracklets(conn)) == 1
    conn.close()
    assert cam.video_path == "videos/cam_0700.mp4"
    assert find_video(cam.video_path, env, "cam_0700").read_bytes() == b"720p"
    assert "imported" in capsys.readouterr().out


def test_resume_packs_a_finished_but_unpacked_scene(tmp_path: Path, monkeypatch):
    Flow(monkeypatch)
    assert bench.run_scenes(["075"], _opts(tmp_path, pack=False)) == {"075": "ok"}
    ws = tmp_path / "ws"
    assert (ws / "aicity24_075_lite").exists() and not zip_path(ws, "aicity24_075_lite").exists()
    assert bench.run_scenes(["075"], _opts(tmp_path)) == {"075": "ok"}  # pack only
    assert packed_complete(ws, "aicity24_075_lite") and not (ws / "aicity24_075_lite").exists()
    # a crash after the zip and results but before the folder was deleted: left over is removed
    (ws / "aicity24_075_lite").mkdir()
    assert bench.run_scenes(["075"], _opts(tmp_path)) == {"075": "skipped"}
    assert not (ws / "aicity24_075_lite").exists()


def test_smoke_fetches_two_cameras_two_minutes(tmp_path: Path, monkeypatch):
    flow = Flow(monkeypatch)
    assert bench.run_scenes(["075"], _opts(tmp_path, smoke=True, pack=False)) == {"075": "ok"}
    kw = flow.fetch_kwargs[0]
    assert kw["cameras"] == 2 and kw["minutes"] == 2 and kw["stream"] is True
    assert (
        flow.fetch_kwargs and read_benchmark(tmp_path / "ws")["runs"]["aicity24_075_lite"]["smoke"]
    )
    flow2 = Flow(monkeypatch)
    bench.run_scenes(["078"], _opts(tmp_path, pack=False))  # full mode: all cameras, whole video
    assert flow2.fetch_kwargs[0]["cameras"] == 0 and flow2.fetch_kwargs[0]["minutes"] is None


# ------------------------------------------------------------------------------------ ETA


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_fmt_remaining():
    assert fmt_remaining(40) == "40 s" and fmt_remaining(37 * 60) == "37 min"
    assert fmt_remaining(112 * 60) == "1 h 52 min" and fmt_remaining(3 * 3600) == "3 h 00 min"


def test_eta_line_speed_and_progress_file(tmp_path: Path):
    clock = FakeClock()
    tr = ProgressTracker("gpu0", tmp_path, ["075", "089"], 2, clock=clock, wall=lambda: 1e9)
    tr.set_scene("075", 2, 1000)  # 2000 source frames of work
    tr.set_scene("089", 2, 1000)  # 4000 in total
    tr.start_scene("075")
    snap = tr.snapshot()
    assert snap["remaining_s"] is None and "measuring speed" in tr.format_line(snap)
    tr.process_start()
    clock.t += 10.0
    tr.progress("cam_a", 500, 1000)  # 500 units in 10 s -> 50 units/s, 25 fps processed
    snap = tr.snapshot()
    assert snap["percent"] == 12.5 and snap["fps_processed"] == 25.0
    assert snap["remaining_s"] == 70 and snap["camera_index"] == 1  # 3500 left / 50 per s
    tr.progress("cam_a", 1000, 1000)
    tr.progress("cam_b", 100, 1000)
    clock.t += 10.0
    assert tr.snapshot()["camera_index"] == 2
    line = tr.format_line(tr.snapshot())
    assert line.startswith("ETA | scene 075 camera 2/2 | ") and "fps (processed)" in line
    assert "left | ends ~" in line and line.endswith("UTC")
    prog = json.loads((tmp_path / "progress_gpu0.json").read_text())
    assert prog["job"] == "gpu0" and prog["line"].startswith("ETA | scene 075")


def test_eta_includes_non_gpu_steps_and_skips(tmp_path: Path):
    clock = FakeClock()
    tr = ProgressTracker("j", tmp_path, ["075", "078"], 2, clock=clock, wall=lambda: 0.0)
    for s in ("075", "078"):
        tr.set_scene(s, 1, 1000)
    tr.start_scene("075")
    tr.process_start()
    clock.t += 10.0
    tr.progress("c", 1000, 1000)  # scene 075 fully processed: 100 units/s
    tr.scene_processed("075")
    tr.process_end()
    base = tr.snapshot()["remaining_s"]  # 1000 units at 100/s
    assert base == 10
    for step, secs in (("fetch", 30.0), ("eval_2d", 20.0), ("export", 10.0)):
        tr.step_done("075", step, secs)
    assert tr.snapshot()["remaining_s"] == 10 + 60  # scene 078 will need the same 60 s
    tr.skip_scene("078")  # failed or complete: no work left
    assert tr.snapshot()["remaining_s"] == 0 and tr.snapshot()["percent"] == 100.0


def test_tracker_emits_on_interval_and_camera_changes(tmp_path: Path, caplog):
    import logging

    clock = FakeClock()
    tr = ProgressTracker("j", tmp_path, ["075"], 2, interval_s=60, clock=clock, wall=lambda: 0.0)
    tr.set_scene("075", 2, 1000)
    tr.start_scene("075")
    tr.process_start()
    with caplog.at_level(logging.INFO, logger="reid"):
        tr.progress("a", 10, 1000)  # new camera: emits
        clock.t += 30
        tr.progress("a", 20, 1000)  # nothing
        clock.t += 31
        tr.progress("a", 30, 1000)  # interval passed: emits
        tr.progress("a", 998, 1000)  # camera end: emits
        tr.progress("b", 2, 1000)  # new camera: emits
    assert sum("ETA |" in r.message for r in caplog.records) == 4


# ----------------------------------------------------------------------------- projection


def _smoke_entry() -> dict:
    return {
        "cameras": 2, "source_frames": 3600, "fps": 30.0, "wall_s": 60.0, "model_load_s": 20.0,
        "timings_s": {"fetch": 12.0, "eval_2d": 24.0, "export": 1.2}, "videos_720p_s": 12.0,
    }  # fmt: skip


def test_projection_extrapolates_per_camera_minute():
    # sample = 2 cameras x 2 min = 4 camera-minutes; a full scene = 16 x 799.8 s / 60
    p = project(_smoke_entry(), ["075", "078", "082", "086"], 2)
    scene_minutes = 16 * 23994 / 30 / 60
    process = 20 + 60 / 4 * scene_minutes
    other = (12 + 24 + 1.2) / 4 * scene_minutes
    enc = 12 / 4 * scene_minutes
    per_gpu_gpu = 2 * (process + other)
    expected = per_gpu_gpu + enc + max(0.0, enc - (process + other))
    assert p["n_scenes"] == 4 and p["n_cameras"] == 64 and p["n_gpus"] == 2
    assert p["total_s"] == pytest.approx(expected, rel=1e-6)
    assert p["process_s"] == pytest.approx(2 * process, rel=1e-6)
    one = project(_smoke_entry(), ["075", "078", "082", "086"], 1)
    assert one["total_s"] > 1.8 * p["total_s"]  # one GPU takes about twice as long
    lines = projection_lines(p)
    assert lines[0].startswith("PROJECTION | 4 scenes, 64 cameras, 2 GPUs: ~")
    assert "process" in lines[0] and "eval 2d" in lines[0] and "export+encode" in lines[0]
    assert "extrapolation from a 2-minute sample" in lines[1] and "not a measurement" in lines[1]
    assert project(_smoke_entry(), ["089"], 1)["n_cameras"] == 15


def test_finish_smoke_writes_projection_even_if_scoring_fails(tmp_path: Path, monkeypatch, caplog):
    import logging

    import reid.eval.aicity_score as sc
    from reid.eval.aicity_bench_io import update_benchmark

    def boom(*a, **k):
        raise FileNotFoundError("nothing to score")

    monkeypatch.setattr(sc, "score_workspace", boom)
    ws = tmp_path / "smoke"
    assert finish_smoke(ws, "balanced", ["075"], 1) is None  # no entry yet: no projection
    update_benchmark(ws, "aicity24_075_balanced", {**_smoke_entry(), "status": "ok"})
    with caplog.at_level(logging.INFO, logger="reid"):
        p = finish_smoke(ws, "balanced", ["075", "078"], 2)
    assert p is not None and (ws / "projection.json").exists()
    assert any("PROJECTION | 2 scenes" in r.message for r in caplog.records)
