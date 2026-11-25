"""One AI City 2024 scene end to end: fetch, process, evaluate, export, 720p videos, pack.

``run_scene`` is idempotent and resumable: an environment that already has a finished run skips
processing, one with ``eval/*.json`` skips the 2D evaluation, one with ``aicity/predictions.txt``
skips the export. An environment holding only an interrupted run is deleted and redone. The
environment (``<workspace>/aicity24_<scene>_<preset>``) is self-contained afterwards: database,
vectors, crops, 720p videos, a 720p-scaled ``gt.csv``, official predictions and GT, logs, timings.
A failure in any step of one scene is logged with its traceback, recorded in ``benchmark.json``
(``status: failed``, ``step``, ``error``) and the next scene goes on, so a long run never loses
the scenes that finished. With packing (default) a successful scene is zipped into
``envs/`` (:mod:`reid.eval.aicity_pack`).
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Executor, Future
from pathlib import Path
from typing import Any

from reid.eval.aicity_bench_io import commit_hash, gpu_name, update_benchmark
from reid.eval.aicity_pack import pack_scene, packed_complete, unpacked_complete
from reid.eval.aicity_progress import ProgressTracker
from reid.eval.aicity_steps import (
    MAX_SIDE,
    BenchOptions,
    attach_log,
    detach_log,
    env_name,
    evaluate_2d,
    export_scene,
    finish_scene,
    process_scene,
    run_fields,
    write_marker,
)
from reid.notebook import has_finished_run

log = logging.getLogger(__name__)


def _record_failure(
    opts: BenchOptions, scene: str, name: str, step: str, exc: BaseException
) -> None:
    update_benchmark(
        opts.workspace, name,
        {"scene": scene, "preset": opts.preset, "env": name, "status": "failed", "step": step,
         "error": f"{type(exc).__name__}: {exc}"},
    )  # fmt: skip
    if opts.tracker is not None:
        opts.tracker.skip_scene(scene)


def _finalize(
    opts: BenchOptions, scene: str, scene_dir: Path, env: Path, meta: dict[str, Any] | None
) -> str:
    """720p copies and raw cleanup (``meta`` given), then pack; returns ``ok`` or ``failed``.

    Runs in the encoder thread (overlapping the next scene's GPU work) or inline.
    """
    name = env.name
    in_thread = threading.current_thread().name.startswith("encode")
    handler = attach_log(env, "encode" if in_thread else "all") if env.exists() else None
    step = "videos"
    try:
        if meta is not None:
            secs = finish_scene(opts, scene_dir, env, meta)
            update_benchmark(opts.workspace, name, {"videos_720p_s": round(secs, 1)})
            if opts.tracker is not None:
                opts.tracker.step_done(scene, "videos_720p", secs)
            log.info("%s: videos and raw cleanup done in %.0f s", name, secs)
        step = "pack"
        update_benchmark(opts.workspace, name, {"status": "ok", "error": None, "step": None})
        if opts.pack:
            zp = pack_scene(opts.workspace, name)
            log.info("%s: packed into %s (%.1f GB)", name, zp, zp.stat().st_size / 2**30)
        return "ok"
    except Exception as exc:  # noqa: BLE001 - one scene must not stop the others
        log.exception("%s failed in step %s", name, step)
        shutil.rmtree(scene_dir, ignore_errors=True)
        _record_failure(opts, scene, name, step, exc)
        return "failed"
    finally:
        if handler is not None:
            detach_log(handler)


def _dispatch(
    encoder: Executor | None, opts: BenchOptions, scene: str, scene_dir: Path, env: Path,
    meta: dict[str, Any] | None,
) -> tuple[str, Future | None]:  # fmt: skip
    if encoder is None:
        return _finalize(opts, scene, scene_dir, env, meta), None
    return "queued", encoder.submit(_finalize, opts, scene, scene_dir, env, meta)


def run_scene(
    scene: str, opts: BenchOptions, encoder: Executor | None = None
) -> tuple[str, Future | None]:
    """Run (or resume) one scene; returns ``(status, encode future)``.

    ``status`` is ``skipped`` (already complete), ``failed`` (recorded, nothing raised),
    ``queued`` (the encode/pack runs in ``encoder``; its future returns ``ok`` or ``failed``) or
    the final ``ok`` / ``failed`` when there is no encoder.
    """
    from reid.datasets.aicity24 import fetch_aicity24

    ws, name = opts.workspace, env_name(scene, opts.preset)
    env, tr = ws / name, opts.tracker
    scene_dir = opts.raw_dir / f"scene_{scene}"
    if packed_complete(ws, name) or (unpacked_complete(env) and not opts.pack):
        if packed_complete(ws, name):
            shutil.rmtree(env, ignore_errors=True)  # left over by a crash after packing
        log.info("%s: complete, skipping", name)
        tr and tr.skip_scene(scene)
        return "skipped", None
    if unpacked_complete(env):  # finished but not packed yet (crash or --no-pack before)
        tr and tr.skip_scene(scene)
        return _dispatch(encoder, opts, scene, scene_dir, env, None)
    if env.exists():  # only an interrupted run
        log.warning("%s: removing an interrupted environment", name)
        shutil.rmtree(env)
    tr and tr.start_scene(scene)
    t: dict[str, float] = {}
    step = "setup"
    handler = attach_log(env, "main")
    try:
        if not (scene_dir / "meta.json").exists():
            step = "fetch"
            tr and tr.note("fetch")
            t0 = time.perf_counter()
            extra = {"cameras": 2, "minutes": 2, "stream": True} if opts.smoke else {"cameras": 0}
            fetch_aicity24(
                scene, height=None, out=scene_dir, yaml_path=scene_dir / "cameras.yaml",
                **({"minutes": None} | extra),
            )  # fmt: skip
            t["fetch"] = time.perf_counter() - t0
            tr and tr.step_done(scene, "fetch", t["fetch"])
        meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
        tr and tr.set_scene(scene, len(meta["camera_numbers"]), int(meta["frames"]))
        entry: dict[str, Any] = {
            "scene": scene, "preset": opts.preset, "env": name, "gpu": gpu_name(),
            "commit": commit_hash(), "frame_stride": opts.frame_stride, "max_side": MAX_SIDE,
            "cameras": len(meta["camera_numbers"]), "source_frames": meta["frames"],
            "smoke": opts.smoke, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }  # fmt: skip
        if not has_finished_run(env):
            step = "process"
            tr and (tr.note("process"), tr.process_start())
            t0 = time.perf_counter()
            try:
                run = process_scene(opts, scene_dir, env)
            finally:
                tr and tr.process_end()
            t["process"] = time.perf_counter() - t0
            entry |= run_fields(run)
            update_benchmark(ws, name, entry)  # keep it if a later step fails
        tr and tr.scene_processed(scene)
        write_marker(env, scene, opts.preset)
        if opts.eval_2d and not list((env / "eval").glob("*.json")):
            step = "eval_2d"
            tr and tr.note("eval_2d")
            t0 = time.perf_counter()
            entry["metrics_2d"] = evaluate_2d(env, scene_dir / "gt.csv", opts.preset)
            t["eval_2d"] = time.perf_counter() - t0
            tr and tr.step_done(scene, "eval_2d", t["eval_2d"])
        step = "export"
        t0 = time.perf_counter()
        entry["export"] = export_scene(env, scene_dir, meta)
        t["export"] = time.perf_counter() - t0
        tr and tr.step_done(scene, "export", t["export"])
        entry["timings_s"] = {k: round(v, 1) for k, v in t.items()}
        update_benchmark(ws, name, entry)
    except Exception as exc:  # noqa: BLE001 - recorded; the other scenes must still run
        log.exception("%s failed in step %s", name, step)
        shutil.rmtree(scene_dir, ignore_errors=True)
        _record_failure(opts, scene, name, step, exc)
        return "failed", None
    finally:
        detach_log(handler)
    return _dispatch(encoder, opts, scene, scene_dir, env, meta)


def run_scenes(scenes: Sequence[str], opts: BenchOptions) -> dict[str, str]:
    """Run scenes in order and return ``{scene: status}``; no scene failure raises.

    The 720p encodes and packing overlap with the next scene. Statuses: ``ok``, ``skipped``
    (already complete) and ``failed`` (see ``benchmark.json`` for step and error).
    """
    from concurrent.futures import ThreadPoolExecutor

    if opts.tracker is None:
        opts.tracker = ProgressTracker(opts.job, opts.workspace, scenes, opts.frame_stride)
    statuses: dict[str, str] = {}
    futures: dict[str, Future] = {}
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="encode") as pool:
        for scene in scenes:
            try:
                statuses[scene], fut = run_scene(scene, opts, pool)
            except Exception as exc:  # noqa: BLE001 - setup problems of one scene
                log.exception("scene %s failed before it started", scene)
                _record_failure(opts, scene, env_name(scene, opts.preset), "setup", exc)
                statuses[scene], fut = "failed", None
            if fut is not None:
                futures[scene] = fut
        for scene, fut in futures.items():
            statuses[scene] = fut.result()
    for scene, st in statuses.items():
        log.info("scene %s: %s", scene, st)
    return statuses
