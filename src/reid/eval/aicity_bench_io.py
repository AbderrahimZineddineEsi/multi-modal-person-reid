"""Small I/O helpers of the AI City benchmark: shared ``benchmark.json``, 720p copies, metadata."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reid.datasets.gt import read_gt, write_gt
from reid.datasets.videos import trim_video
from reid.storage import files

log = logging.getLogger(__name__)

BENCHMARK_FILE = "benchmark.json"


def update_benchmark(workspace: str | Path, env_name: str, fields: Mapping[str, Any]) -> None:
    """Merge ``fields`` into ``workspace/benchmark.json`` under ``runs[env_name]``.

    Two processes (one per GPU) and the encoder thread write this file, so the read-modify-write
    holds an exclusive ``flock`` on a lock file and replaces the file atomically.
    """
    import fcntl

    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    path = ws / BENCHMARK_FILE
    with open(ws / ".benchmark.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = read_benchmark(ws)
        data.setdefault("runs", {}).setdefault(env_name, {}).update(fields)
        fd, tmp = tempfile.mkstemp(dir=ws, prefix=".benchmark-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)


def read_benchmark(workspace: str | Path) -> dict[str, Any]:
    """Contents of ``benchmark.json`` (``{}`` if missing or unreadable)."""
    try:
        return json.loads((Path(workspace) / BENCHMARK_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def gpu_name() -> str | None:
    """Name of the first visible CUDA GPU (honours ``CUDA_VISIBLE_DEVICES``), else None."""
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - torch missing or CUDA broken: only a report field
        return None


def commit_hash() -> str | None:
    """Commit of the installed code (pip's ``direct_url.json`` for a git install, else git)."""
    try:
        from importlib.metadata import distribution

        raw = distribution("reid").read_text("direct_url.json")
        commit = (json.loads(raw).get("vcs_info") or {}).get("commit_id") if raw else None
        if commit:
            return str(commit)
    except Exception:  # noqa: BLE001
        pass
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
            capture_output=True, text=True, timeout=10,
        )  # fmt: skip
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def camera_sizes(env: str | Path) -> dict[str, tuple[int, int]]:
    """Stored (processed) ``(width, height)`` of every camera of an environment."""
    conn = sqlite3.connect(f"file:{files.db_path(env)}?mode=ro", uri=True)
    try:
        return {
            r[0]: (int(r[1]), int(r[2]))
            for r in conn.execute("SELECT id,width,height FROM cameras")
        }
    finally:
        conn.close()


def scale_gt_csv(
    gt_in: str | Path,
    gt_out: str | Path,
    src_sizes: Mapping[str, Sequence[int]],
    target_sizes: Mapping[str, Sequence[int]],
) -> Path:
    """Rewrite a uniform GT CSV from source pixels to each camera's ``target_sizes`` pixels."""
    gt = read_gt(gt_in)
    for cid, (sw, sh) in src_sizes.items():
        m = gt["camera"] == cid
        tw, th = target_sizes[cid]
        for cols, k in ((["x1", "x2"], tw / sw), (["y1", "y2"], th / sh)):
            gt.loc[m, cols] = gt.loc[m, cols] * k
    return write_gt(gt, gt_out)


def encode_videos(
    sources: Mapping[str, str | Path], env: str | Path, height: int = 720, crf: int = 28
) -> dict[str, float]:
    """Re-encode each camera video to ``env/videos/<camera>.mp4`` (low priority, 2 threads).

    Files are written as ``*.part.mp4`` and renamed when complete, so an interrupted encode is
    redone. Returns seconds per camera. Raises if ffmpeg fails.
    """
    out_dir = Path(env) / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    secs: dict[str, float] = {}
    for cid, src in sources.items():
        dst = out_dir / f"{cid}.mp4"
        if dst.exists():
            continue
        part = out_dir / f"{cid}.part.mp4"
        t0 = time.perf_counter()
        trim_video(src, part, None, height, crf=crf, nice=10, threads=2)
        part.replace(dst)
        secs[cid] = time.perf_counter() - t0
        log.info("%s: encoded %dp copy in %.0f s", cid, height, secs[cid])
    return secs


def repoint_videos(env: str | Path, camera_ids: Sequence[str]) -> None:
    """Point the cameras of ``env`` at ``videos/<camera>.mp4`` (relative, so the env can move)."""
    conn = sqlite3.connect(files.db_path(env))
    try:
        for cid in camera_ids:
            conn.execute("UPDATE cameras SET video_path=? WHERE id=?", (f"videos/{cid}.mp4", cid))
        conn.commit()
    finally:
        conn.close()
