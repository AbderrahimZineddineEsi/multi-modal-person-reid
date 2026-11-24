"""Annotated camera videos, GIF export and synchronized multi-camera mosaics (PLAN section 4)."""

from __future__ import annotations

import logging
import math
import sqlite3
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np

from reid.config import Config
from reid.storage import db, files
from reid.types import CameraInfo
from reid.viz.draw import (
    BG_BGR,
    CameraTracks,
    Style,
    draw_view,
    letterbox,
)
from reid.viz.video_io import FrameSource, VideoSink, ffmpeg_exe, resolve_video

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], None]


def default_out(cfg: Config, name: str) -> Path:
    """``data_dir/renders/<name>``."""
    return files.render_path(cfg.data_dir, name)


def _open_source(cfg: Config, cam) -> FrameSource:
    return FrameSource(resolve_video(cam.video_path, cfg.data_dir, cam.id), (cam.width, cam.height))


def render_camera(
    conn: sqlite3.Connection,
    cfg: Config,
    camera_id: str,
    out_path: str | Path,
    gid_filter: Iterable[int] | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    scale: float | None = None,
    highlight_gid: int | None = None,
    progress: ProgressFn | None = None,
) -> Path:
    """Re-read the camera's video and draw the stored track points on it.

    ``start_s`` / ``end_s`` are seconds from the start of the *video* (``frame / fps``). Boxes
    come from ``track_points`` (one query for the range, grouped by frame); no per-frame DB access.
    With ``highlight_gid`` that person is bold and spotlighted, everyone else dimmed; with
    ``gid_filter`` only those identities are drawn. Output: H.264 yuv420p mp4.
    """
    st = Style.from_cfg(cfg, highlight_gid, gid_filter)
    cam = db.get_camera(conn, camera_id)
    if cam is None:
        raise KeyError(f"unknown camera {camera_id!r}")
    sc = float(scale if scale is not None else cfg.render.scale)
    with _open_source(cfg, cam) as src:
        fps = src.fps or cam.fps
        n = src.n_frames if src.n_frames > 0 else 2**31 - 1
        f0 = max(0, int(round((start_s or 0.0) * fps)))
        f1 = n if end_s is None else min(n, int(round(end_s * fps)))
        if f1 <= f0:
            raise ValueError(f"empty frame range [{f0}, {f1}) for {camera_id}")
        tracks = CameraTracks(conn, camera_id, f0, f1 - 1, st.trail_len)
        size = (max(2, int(round(src.width * sc))), max(2, int(round(src.height * sc))))
        out_fps = cfg.render.out_fps or fps
        written = 0
        with VideoSink(out_path, size[0], size[1], out_fps, cfg.render.crf) as sink:
            for f in range(f0, f1):
                frame = src.get(f)
                if frame is None:
                    break
                ts = cam.start_ts + f / cam.fps
                sink.write(draw_view(frame, tracks, f, size, st, t=ts, name=cam.name))
                written += 1
                if progress and written % 25 == 0:
                    progress(written, f1 - f0)
    if written == 0:
        raise RuntimeError(f"no frames could be read from {cam.video_path}")
    return Path(out_path)


def export_gif(
    mp4: str | Path,
    gif: str | Path,
    fps: float = 12.0,
    width: int = 640,
) -> Path:
    """Convert a video to an optimized GIF (ffmpeg two-pass palette), e.g. for the README."""
    exe = ffmpeg_exe() or _any_ffmpeg()
    if exe is None:
        raise RuntimeError("ffmpeg is required for GIF export")
    Path(gif).parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"fps={fps:g},scale={int(width)}:-2:flags=lanczos,split[a][b];"
        "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    )
    subprocess.run(
        [exe, "-y", "-loglevel", "error", "-i", str(mp4), "-vf", vf, "-loop", "0", str(gif)],
        check=True,
    )
    return Path(gif)


def _any_ffmpeg() -> str | None:
    import shutil

    return shutil.which("ffmpeg")


# --------------------------------------------------------------------------------- mosaic


def grid_shape(n: int) -> tuple[int, int]:
    """``(cols, rows)`` of the mosaic for ``n`` tiles."""
    cols = 1 if n <= 1 else 2 if n <= 4 else math.ceil(math.sqrt(n))
    return cols, math.ceil(n / cols)


def frame_index_at(t: float, start_ts: float, fps: float, n_frames: int) -> int | None:
    """Frame of a camera shown at shared time ``t`` (None before it starts / after it ends)."""
    i = int(math.floor((t - start_ts) * fps + 1e-6))
    return i if 0 <= i < n_frames else None


def tile_at(
    cam: CameraInfo,
    src: FrameSource,
    trk: CameraTracks,
    t: float,
    size: tuple[int, int],
    st: Style,
    hud: bool | None = None,
) -> tuple[np.ndarray, bool]:
    """Annotated view of ``cam`` at shared time ``t`` and whether the highlighted gid is in it.

    Video frames are addressed on the video's fps; track points on the DB's (``cam.fps``).
    """
    idx = frame_index_at(t, cam.start_ts, src.fps, src.n_frames)
    frame = src.get(idx) if idx is not None else None
    didx = int(round((t - cam.start_ts) * cam.fps))
    tile = draw_view(frame, trk, didx, size, st, t=t, name=cam.name, hud=hud)
    seen = (
        frame is not None and st.highlight_gid is not None and st.highlight_gid in trk.gids_at(didx)
    )
    return tile, bool(seen)


def render_mosaic(
    conn: sqlite3.Connection,
    cfg: Config,
    camera_ids: Sequence[str] | None,
    out_path: str | Path,
    highlight_gid: int | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    tile_width: int | None = None,
    progress: ProgressFn | None = None,
) -> Path:
    """Synchronized grid of cameras on the shared timeline ``start_ts + frame / fps``.

    Cameras may differ in fps, length and ``start_ts``; a camera without a frame at time ``t``
    shows a dark "NO SIGNAL" tile. ``start_s`` / ``end_s`` are offsets from the earliest camera
    start. With ``highlight_gid`` the person is spotlighted in every tile; tiles where they are
    currently visible get an outline in their colour.
    """
    cams = [db.get_camera(conn, c) for c in (camera_ids or [c.id for c in db.list_cameras(conn)])]
    if not cams or any(c is None for c in cams):
        raise KeyError("unknown or no cameras for the mosaic")
    srcs = [_open_source(cfg, c) for c in cams if c is not None]
    try:
        st = Style.from_cfg(cfg, highlight_gid)
        tr = [CameraTracks(conn, c.id, 0, 2**31 - 1, st.trail_len) for c in cams if c]
        spans = [
            (c.start_ts, c.start_ts + s.n_frames / s.fps)
            for c, s in zip(cams, srcs, strict=True)
            if c
        ]
        t_min = min(a for a, _ in spans)
        t_max = max(b for _, b in spans)
        t0 = t_min + (start_s or 0.0)
        t1 = t_max if end_s is None else min(t_max, t_min + end_s)
        out_fps = cfg.render.out_fps or min(30.0, max(s.fps for s in srcs))
        tw = int(tile_width or cfg.render.mosaic_tile_width) // 2 * 2
        aspect = srcs[0].height / srcs[0].width
        th = int(round(tw * aspect)) // 2 * 2
        cols, rows = grid_shape(len(cams))
        gap = 4
        W, H = cols * tw + (cols + 1) * gap, rows * th + (rows + 1) * gap
        n_out = int(math.floor((t1 - t0) * out_fps))
        if n_out <= 0:
            raise ValueError("empty time range")
        with VideoSink(out_path, W, H, out_fps, cfg.render.crf) as sink:
            for k in range(n_out):
                t = t0 + k / out_fps
                canvas = np.empty((H, W, 3), np.uint8)
                canvas[:] = BG_BGR
                for i, (cam, src, trk) in enumerate(zip(cams, srcs, tr, strict=True)):
                    assert cam is not None
                    tile, seen = tile_at(cam, src, trk, t, (tw, th), st)
                    if seen and highlight_gid is not None:
                        _outline(tile, highlight_gid)
                    x = gap + (i % cols) * (tw + gap)
                    y = gap + (i // cols) * (th + gap)
                    canvas[y : y + th, x : x + tw] = tile
                sink.write(canvas)
                if progress and k % 25 == 0:
                    progress(k, n_out)
    finally:
        for s in srcs:
            s.close()
    return Path(out_path)


def _outline(tile: np.ndarray, gid: int) -> None:
    from reid.viz.colors import gid_color_bgr

    h, w = tile.shape[:2]
    cv2.rectangle(tile, (0, 0), (w - 1, h - 1), gid_color_bgr(gid), 3, cv2.LINE_AA)


def timed(fn: Callable[..., Path], *a, **kw) -> tuple[Path, float]:
    """Run a render function and return ``(path, seconds)`` (for benchmarks)."""
    t = time.perf_counter()
    p = fn(*a, **kw)
    return p, time.perf_counter() - t


__all__ = [
    "default_out",
    "export_gif",
    "frame_index_at",
    "grid_shape",
    "letterbox",
    "render_camera",
    "render_mosaic",
    "tile_at",
    "timed",
]
