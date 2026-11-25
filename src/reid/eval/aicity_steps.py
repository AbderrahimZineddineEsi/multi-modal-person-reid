"""Steps of one benchmark scene: processing, 2D evaluation, export, 720p copies, log files.

Split from :mod:`reid.eval.aicity_bench` (which orchestrates them with failure handling).
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reid.config import load_cameras, load_config
from reid.eval.aicity_bench_io import camera_sizes, encode_videos, repoint_videos, scale_gt_csv
from reid.eval.aicity_export import export_predictions
from reid.eval.aicity_progress import ProgressTracker
from reid.storage import files
from reid.storage.maintenance import ENV_MARKER

log = logging.getLogger(__name__)

MAX_SIDE = 1280  # 720p processing (the speed measured on a T4)


def env_name(scene: str, preset: str) -> str:
    """Environment folder name of a scene and preset."""
    return f"aicity24_{scene}_{preset}"


@dataclass
class BenchOptions:
    """Settings shared by every scene of one ``reid benchmark aicity24 run`` call."""

    workspace: Path
    raw_dir: Path
    preset: str = "balanced"
    frame_stride: int = 2
    overrides: dict[str, Any] = field(default_factory=dict)
    config_path: str | None = None
    keep_videos: str = "720"  # "720" | "none"
    eval_2d: bool = True
    pack: bool = True  # zip each successful scene into envs/ and delete the folder
    smoke: bool = False  # 2 cameras, 2 minutes
    job: str = "run"  # names progress_<job>.json
    tracker: ProgressTracker | None = None


def progress_logger():
    last: dict[str, int] = {}

    def progress(cam: str, done: int, total: int) -> None:
        pct = int(100 * done / total) // 10 * 10 if total else 0
        if pct > last.get(cam, -1):
            last[cam] = pct
            log.info("%s: %d%% (%d / %d source frames)", cam, pct, done, total)

    return progress


def progress_callback(opts: BenchOptions):
    """Pipeline progress callback: 10% log lines plus the job's ETA tracker."""
    plain = progress_logger()
    tracker = opts.tracker

    def progress(cam: str, done: int, total: int) -> None:
        plain(cam, done, total)
        if tracker is not None:
            tracker.progress(cam, done, total)

    return progress


def process_scene(opts: BenchOptions, scene_dir: Path, env: Path) -> dict[str, Any]:
    """Run the pipeline on the scene's cameras into ``env``; returns ``RunResult.to_dict()``."""
    from reid.cli import _make_engine
    from reid.pipeline import run
    from reid.storage import db

    over = {
        "paths.data_dir": str(env),
        "runtime.max_side": MAX_SIDE,
        "runtime.frame_stride": opts.frame_stride,
        **opts.overrides,
    }
    cfg = load_config(opts.config_path, over, preset=opts.preset)
    cameras = load_cameras(scene_dir / "cameras.yaml")
    files.ensure_dirs(env)
    conn = db.connect(files.db_path(env))
    try:
        engine = _make_engine(cfg, cameras, conn)
        res = run(cfg, cameras, engine, progress_callback(opts), conn=conn)
    finally:
        conn.close()
    out = res.to_dict()
    stats = env / "cache" / f"run_{res.run_id}.json"  # the app's Metrics tab reads this
    stats.parent.mkdir(parents=True, exist_ok=True)
    stats.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def evaluate_2d(env: Path, gt: Path, preset: str) -> dict[str, Any]:
    from reid.eval import evaluate, save_result

    result = evaluate(env, gt, preset=preset)
    save_result(env, result)
    keep = ("idf1", "idp", "idr", "mota", "id_switches", "precision", "recall")
    out = {k: result["mtmc"].get(k) for k in keep}
    out["identities"] = result["counts"]["identities"]
    out["gt_ids"] = result["counts"]["gt_ids"]
    out["fragmentation"] = result["fragmentation"]
    out["retrieval"] = {k: v for k, v in result.get("retrieval", {}).items() if _scalar(v)}
    return out


def _scalar(v: Any) -> bool:
    return isinstance(v, (int, float, str)) and not isinstance(v, bool)


def export_scene(env: Path, scene_dir: Path, meta: dict[str, Any]) -> dict[str, int]:
    nums = {cid: int(n) for cid, n in meta["camera_numbers"].items()}
    sizes = {cid: (int(w), int(h)) for cid, (w, h) in meta["src_sizes"].items()}
    out = env / "aicity"
    stats = export_predictions(env, out / "predictions.txt", camera_numbers=nums, src_sizes=sizes)
    for name in ("gt_official.txt", "calibration.json", "meta.json"):
        if (scene_dir / name).exists():
            shutil.copyfile(scene_dir / name, out / name)
    return stats


def finish_scene(opts: BenchOptions, scene_dir: Path, env: Path, meta: dict[str, Any]) -> float:
    """720p copies (when asked), repoint the cameras, 720p-scaled ``gt.csv``, remove raw files."""
    t0 = time.perf_counter()
    ids = list(meta["camera_numbers"])
    if opts.keep_videos != "none":
        encode_videos(
            {cid: scene_dir / f"{cid}.mp4" for cid in ids}, env, int(opts.keep_videos), crf=28
        )
        repoint_videos(env, ids)
    sizes = {cid: tuple(wh) for cid, wh in meta["src_sizes"].items()}
    scale_gt_csv(scene_dir / "gt.csv", env / "gt.csv", sizes, camera_sizes(env))
    shutil.rmtree(scene_dir, ignore_errors=True)
    return time.perf_counter() - t0


def run_fields(run: dict[str, Any]) -> dict[str, Any]:
    keys = ("run_id", "frames_processed", "tracklets", "fps", "wall_s", "model_load_s",
            "ms_per_frame", "vram_peak_mb", "gpu_memory", "device_plan")  # fmt: skip
    return {k: run.get(k) for k in keys}


def write_marker(env: Path, scene: str, preset: str) -> None:
    marker = env / ENV_MARKER
    if not marker.exists():
        marker.write_text(
            json.dumps(
                {
                    "name": env.name,
                    "created_at": time.time(),
                    "description": f"AI City 2024 scene {scene}, preset {preset}",
                }
            ),
            encoding="utf-8",
        )


def attach_log(env: Path, scope: str = "all") -> logging.Handler:
    """Log ``reid`` records into ``env/logs/process.log`` until :func:`detach_log`.

    ``scope`` splits the two threads of a run: ``main`` skips the encoder thread's records,
    ``encode`` keeps only them, ``all`` keeps everything.
    """
    (env / "logs").mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(env / "logs" / "process.log", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    if scope != "all":
        want = scope == "encode"
        h.addFilter(lambda r: r.threadName.startswith("encode") == want)
    logging.getLogger("reid").addHandler(h)
    return h


def detach_log(h: logging.Handler) -> None:
    logging.getLogger("reid").removeHandler(h)
    h.close()
