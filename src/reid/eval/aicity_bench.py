"""One AI City 2024 scene end to end: fetch, process, evaluate, export, keep 720p videos.

``run_scene`` is idempotent and resumable: an environment that already has a finished run skips
processing, one with ``eval/*.json`` skips the 2D evaluation, one with ``aicity/predictions.txt``
skips the export. An environment holding only an interrupted run is deleted and redone. The
environment (``<workspace>/aicity24_<scene>_<preset>``) is self-contained afterwards: database,
vectors, crops, 720p videos, a 720p-scaled ``gt.csv``, official predictions and GT, logs, timings.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Sequence
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reid.config import load_cameras, load_config
from reid.eval.aicity_bench_io import (
    camera_sizes,
    commit_hash,
    encode_videos,
    gpu_name,
    repoint_videos,
    scale_gt_csv,
    update_benchmark,
)
from reid.eval.aicity_export import export_predictions
from reid.notebook import has_finished_run
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


def is_complete(env: Path) -> bool:
    """True when ``env`` has a finished run, exported predictions and the (last-written) gt.csv."""
    return (
        has_finished_run(env)
        and (env / "aicity" / "predictions.txt").exists()
        and (env / "gt.csv").exists()
    )


def _progress_logger():
    last: dict[str, int] = {}

    def progress(cam: str, done: int, total: int) -> None:
        pct = int(100 * done / total) // 10 * 10 if total else 0
        if pct > last.get(cam, -1):
            last[cam] = pct
            log.info("%s: %d%% (%d / %d source frames)", cam, pct, done, total)

    return progress


def _process(opts: BenchOptions, scene_dir: Path, env: Path) -> dict[str, Any]:
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
        res = run(cfg, cameras, engine, _progress_logger(), conn=conn)
    finally:
        conn.close()
    out = res.to_dict()
    stats = env / "cache" / f"run_{res.run_id}.json"  # the app's Metrics tab reads this
    stats.parent.mkdir(parents=True, exist_ok=True)
    stats.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def _evaluate_2d(env: Path, gt: Path, preset: str) -> dict[str, Any]:
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


def _export(env: Path, scene_dir: Path, meta: dict[str, Any]) -> dict[str, int]:
    nums = {cid: int(n) for cid, n in meta["camera_numbers"].items()}
    sizes = {cid: (int(w), int(h)) for cid, (w, h) in meta["src_sizes"].items()}
    out = env / "aicity"
    stats = export_predictions(env, out / "predictions.txt", camera_numbers=nums, src_sizes=sizes)
    for name in ("gt_official.txt", "calibration.json", "meta.json"):
        if (scene_dir / name).exists():
            shutil.copyfile(scene_dir / name, out / name)
    return stats


def _finish_scene(opts: BenchOptions, scene_dir: Path, env: Path, meta: dict[str, Any]) -> float:
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


def _finish_and_record(
    opts: BenchOptions, scene_dir: Path, env: Path, meta: dict[str, Any], name: str
) -> None:
    secs = _finish_scene(opts, scene_dir, env, meta)
    update_benchmark(opts.workspace, name, {"videos_720p_s": round(secs, 1)})
    log.info("%s: videos and raw cleanup done in %.0f s", name, secs)


def run_scene(scene: str, opts: BenchOptions, encoder: Executor | None = None) -> Future | None:
    """Run (or resume) one scene. Returns the encode future (``None`` when nothing was queued).

    With ``encoder`` the 720p encodes and the raw-file cleanup run there (so the next scene's GPU
    work overlaps with them); without it they run inline.
    """
    from reid.datasets.aicity24 import fetch_aicity24

    name = env_name(scene, opts.preset)
    env = opts.workspace / name
    if is_complete(env):
        log.info("%s: finished run and predictions exist, skipping", name)
        return None
    scene_dir = opts.raw_dir / f"scene_{scene}"
    t: dict[str, float] = {}
    if env.exists() and not has_finished_run(env):
        log.warning("%s: removing an interrupted environment", name)
        shutil.rmtree(env)
    handler = _attach_log(env)
    try:
        if not (scene_dir / "meta.json").exists():
            t0 = time.perf_counter()
            fetch_aicity24(
                scene, cameras=0, minutes=None, height=None, out=scene_dir,
                yaml_path=scene_dir / "cameras.yaml",
            )  # fmt: skip
            t["fetch"] = time.perf_counter() - t0
        meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
        entry: dict[str, Any] = {
            "scene": scene, "preset": opts.preset, "env": name, "gpu": gpu_name(),
            "commit": commit_hash(), "frame_stride": opts.frame_stride, "max_side": MAX_SIDE,
            "cameras": len(meta["camera_numbers"]), "source_frames": meta["frames"],
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }  # fmt: skip
        if not has_finished_run(env):
            t0 = time.perf_counter()
            run = _process(opts, scene_dir, env)
            t["process"] = time.perf_counter() - t0
            entry |= _run_fields(run)
            update_benchmark(opts.workspace, name, entry)  # keep it if a later step fails
        _write_marker(env, scene, opts.preset)
        if opts.eval_2d and not list((env / "eval").glob("*.json")):
            t0 = time.perf_counter()
            entry["metrics_2d"] = _evaluate_2d(env, scene_dir / "gt.csv", opts.preset)
            t["eval_2d"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        entry["export"] = _export(env, scene_dir, meta)
        t["export"] = time.perf_counter() - t0
        entry["timings_s"] = {k: round(v, 1) for k, v in t.items()}
        update_benchmark(opts.workspace, name, entry)
    finally:
        _detach_log(handler)
    if encoder is None:
        _finish_and_record(opts, scene_dir, env, meta, name)
        return None
    return encoder.submit(_finish_and_record, opts, scene_dir, env, meta, name)


def _run_fields(run: dict[str, Any]) -> dict[str, Any]:
    keys = ("run_id", "frames_processed", "tracklets", "fps", "wall_s", "model_load_s",
            "ms_per_frame", "vram_peak_mb", "gpu_memory", "device_plan")  # fmt: skip
    return {k: run.get(k) for k in keys}


def _write_marker(env: Path, scene: str, preset: str) -> None:
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


def _attach_log(env: Path) -> logging.Handler:
    """Log everything from ``reid`` into ``env/logs/process.log`` until :func:`_detach_log`."""
    (env / "logs").mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(env / "logs" / "process.log", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger("reid").addHandler(h)
    return h


def _detach_log(h: logging.Handler) -> None:
    logging.getLogger("reid").removeHandler(h)
    h.close()


def run_scenes(scenes: Sequence[str], opts: BenchOptions) -> list[str]:
    """Run scenes in order; the 720p encodes overlap with the next scene. Returns the scenes run."""
    from concurrent.futures import ThreadPoolExecutor

    futures: list[Future] = []
    done: list[str] = []
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="encode") as pool:
        for scene in scenes:
            fut = run_scene(scene, opts, pool)
            done.append(scene)
            if fut is not None:
                futures.append(fut)
        for fut in futures:  # wait for every encode; re-raise a failure
            fut.result()
    return done
