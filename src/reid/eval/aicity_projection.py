"""Smoke-run projection: extrapolate a 2-minute, 2-camera sample to the full benchmark.

The smoke run measures, for one scene, the seconds of every step per source camera-minute (one
camera, one minute of video). :func:`project` multiplies those rates by the size of the real run
(its scenes split over N GPUs with :func:`reid.notebook.balance`) and adds the fixed model-load
time per scene. It is an extrapolation from a small sample and says so: costs that grow faster
than linearly (identity consolidation, the 2D evaluation over 16 cameras) and the CPU contention
between the GPU jobs and the 720p encodes are not modelled.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reid.eval.aicity_progress import CAMERAS_BY_SCENE, DEFAULT_FRAMES, fmt_remaining
from reid.notebook import balance

log = logging.getLogger(__name__)

FULL_FPS = 30.0


def rates(entry: Mapping[str, Any], fps: float = FULL_FPS) -> dict[str, float]:
    """Seconds per source camera-minute of each step, from a smoke ``benchmark.json`` entry."""
    units = entry["cameras"] * entry["source_frames"] / fps / 60.0
    t = entry.get("timings_s", {})
    out = {
        "process": float(entry.get("wall_s") or t.get("process", 0.0)) / units,
        "fetch": float(t.get("fetch", 0.0)) / units,
        "eval_2d": float(t.get("eval_2d", 0.0)) / units,
        "export": float(t.get("export", 0.0)) / units,
        "encode": float(entry.get("videos_720p_s", 0.0)) / units,
    }
    out["model_load_s"] = float(entry.get("model_load_s") or 0.0)  # per scene, not per minute
    out["fps_processed"] = float(entry.get("fps") or 0.0)
    return out


def _scene_times(r: Mapping[str, float], scene: str, frames: int) -> dict[str, float]:
    cams = CAMERAS_BY_SCENE.get(scene, 16)
    units = cams * frames / FULL_FPS / 60.0
    return {
        "process": r["model_load_s"] + r["process"] * units,
        **{k: r[k] * units for k in ("fetch", "eval_2d", "export", "encode")},
    }


def project(
    entry: Mapping[str, Any],
    scenes: Sequence[str],
    n_gpus: int,
    *,
    frames: int = DEFAULT_FRAMES,
) -> dict[str, Any]:
    """Projected duration of the full run of ``scenes`` on ``n_gpus`` GPUs (see the docstring)."""
    r = rates(entry)
    weights = {s: float(CAMERAS_BY_SCENE.get(s, 16)) for s in scenes}
    worst: dict[str, Any] = {"total_s": 0.0}
    for group in balance(list(scenes), weights, max(1, n_gpus)):
        times = [_scene_times(r, s, frames) for s in group]
        gpu = [t["process"] + t["fetch"] + t["eval_2d"] + t["export"] for t in times]
        enc = [t["encode"] for t in times]
        # encodes of earlier scenes overlap with the GPU work that follows; the last one does not
        hidden_window = sum(gpu) - gpu[0]
        total = sum(gpu) + enc[-1] + max(0.0, sum(enc[:-1]) - hidden_window)
        if total > worst["total_s"]:
            worst = {
                "total_s": total,
                "scenes": list(group),
                "process_s": sum(t["process"] for t in times),
                "fetch_s": sum(t["fetch"] for t in times),
                "eval_2d_s": sum(t["eval_2d"] for t in times),
                "export_encode_s": sum(t["export"] + t["encode"] for t in times),
            }
    cameras = sum(CAMERAS_BY_SCENE.get(s, 16) for s in scenes)
    return {
        **worst, "n_scenes": len(scenes), "n_cameras": cameras, "n_gpus": max(1, n_gpus),
        "sample": {"cameras": entry["cameras"], "source_frames": entry["source_frames"]},
        "rates_s_per_camera_minute": r,
    }  # fmt: skip


def projection_lines(p: Mapping[str, Any]) -> list[str]:
    """The ``PROJECTION |`` log lines of a :func:`project` result."""
    f = fmt_remaining
    head = (
        f"PROJECTION | {p['n_scenes']} scenes, {p['n_cameras']} cameras, {p['n_gpus']} GPUs: "
        f"~{f(p['total_s'])} (process {f(p['process_s'])}, fetch {f(p['fetch_s'])}, "
        f"eval 2d {f(p['eval_2d_s'])}, export+encode {f(p['export_encode_s'])})"
    )
    s = p["sample"]
    note = (
        "PROJECTION | extrapolation from a "
        f"{s['source_frames'] / FULL_FPS / 60:.0f}-minute sample of {s['cameras']} cameras, not a "
        "measurement of the full run: costs that grow faster than linearly (consolidation, 2D "
        "evaluation) and CPU contention between jobs are not modelled."
    )
    return [head, note]


def write_projection(workspace: str | Path, p: Mapping[str, Any]) -> Path:
    path = Path(workspace) / "projection.json"
    path.write_text(json.dumps({**p, "lines": projection_lines(p)}, indent=2), encoding="utf-8")
    for line in projection_lines(p):
        log.info(line)
    return path


def finish_smoke(
    workspace: str | Path, preset: str, scenes: Sequence[str], n_gpus: int, num_cores: int = 4
) -> dict[str, Any] | None:
    """After a smoke run: score its scene, then log and write the projection of the full run.

    Nothing here raises: a scoring or projection problem is logged and the smoke still ends.
    Returns the projection, or None when the smoke scene did not finish.
    """
    from reid.eval.aicity_bench_io import read_benchmark
    from reid.eval.aicity_score import score_workspace
    from reid.eval.aicity_steps import env_name

    try:
        res = score_workspace(workspace, preset, num_cores)
        log.info("smoke score (2 cameras, 2 minutes): %s", res["overall"])
    except Exception:  # noqa: BLE001
        log.exception("smoke scoring failed (the projection is still computed)")
    entry = read_benchmark(workspace).get("runs", {}).get(env_name(scenes[0], preset), {})
    if entry.get("status") != "ok" or "cameras" not in entry or "wall_s" not in entry:
        log.error("PROJECTION | the smoke scene did not finish (see benchmark.json): no projection")
        return None
    try:
        p = project(entry, scenes, n_gpus)
        write_projection(workspace, p)
        return p
    except Exception:  # noqa: BLE001
        log.exception("could not compute the projection")
        return None
