"""Follow-person clips: cut between cameras along one identity's journey (PLAN section 4).

What the clip contains is decided by :func:`reid.viz.clip_plan.plan_clip`; this module renders
that plan. ``follow_person`` and ``plan_segments`` stay for the CLI and older callers.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from reid.config import Config
from reid.storage import db
from reid.types import CameraInfo
from reid.viz.clip_panels import person_card, ramp, tile_label, timeline_bg, union_len
from reid.viz.clip_plan import (
    ClipOptions,
    ClipPlan,
    PlanSegment,
    Segment,
    plan_clip,
    plan_segments,
    tile_every,
)
from reid.viz.colors import gid_color, gid_color_bgr
from reid.viz.draw import BG_BGR, CameraTracks, Style, chip_sprite, draw_view, paste
from reid.viz.render import ProgressFn, tile_at
from reid.viz.video_io import FrameSource, VideoSink, resolve_video

log = logging.getLogger(__name__)
_WHITE = (255, 255, 255)


@dataclass(slots=True)
class RenderResult:
    """Outcome of a render: file, wall time, frames written and seconds spent per stage."""

    path: Path
    seconds: float
    frames: int
    timings: dict[str, float] = field(default_factory=dict)


def frame_window(cam: CameraInfo, lo: float, hi: float) -> tuple[int, int]:
    """Frame range of ``cam`` covering the shared-time window ``[lo, hi]`` (open ends allowed)."""
    f0 = 0 if not np.isfinite(lo) else max(0, int((lo - cam.start_ts) * cam.fps) - 1)
    f1 = 2**31 - 1 if not np.isfinite(hi) else int(np.ceil((hi - cam.start_ts) * cam.fps)) + 1
    return f0, max(f1, f0)


class _Sources:
    """Lazily opened video readers; a missing thumbnail video is None (drawn as NO SIGNAL)."""

    def __init__(self, cfg: Config, cams: dict[str, CameraInfo]) -> None:
        self._cfg, self._cams = cfg, cams
        self._open: dict[str, FrameSource | None] = {}

    def need(self, cid: str) -> FrameSource:
        """Reader of a camera the main view needs (raises ``FileNotFoundError`` if missing)."""
        src = self._open.get(cid)
        if src is None:
            c = self._cams[cid]
            src = FrameSource(
                resolve_video(c.video_path, self._cfg.data_dir, c.id), (c.width, c.height)
            )
            self._open[cid] = src
        return src

    def optional(self, cid: str) -> FrameSource | None:
        try:
            return self.need(cid)
        except FileNotFoundError:
            self._open[cid] = None
            return None

    def close(self) -> None:
        for s in self._open.values():
            if s is not None:
                s.close()


def _tile(cam, src, trk, t, size, st, hud=None) -> tuple[np.ndarray, bool]:
    """Annotated view; a camera without video gives a NO SIGNAL tile."""
    if src is None:
        return draw_view(None, trk, 0, size, st, t=t, name=cam.name, hud=hud), False
    return tile_at(cam, src, trk, t, size, st, hud=hud)


def render_clip(
    conn: sqlite3.Connection,
    cfg: Config,
    gid: int,
    out_path: str | Path,
    opts: ClipOptions | None = None,
    progress: ProgressFn | None = None,
    plan: ClipPlan | None = None,
) -> RenderResult:
    """Render the follow clip of ``gid`` described by ``plan`` (default: ``plan_clip``).

    The active camera is shown large with the person spotlighted, other cameras sit in a strip on
    the right (same instant), and a timeline at the bottom shows the person's presence per camera
    with a moving playhead. A caption announces every cut ("-> cam_02, +4.2 s"). Raises
    ``ValueError`` for an empty plan and ``FileNotFoundError`` when a main-view video is missing.
    """
    opts = opts or ClipOptions.from_cfg(cfg)
    plan = plan or plan_clip(conn, gid, opts)
    if plan.empty or plan.layout is None:
        raise ValueError(f"identity {gid} has no tracklets in the requested window")
    cfg = opts.apply_cfg(cfg)
    lay, segs = plan.layout, plan.segments
    cams = {c.id: c for c in db.list_cameras(conn)}
    rgb, bgr = gid_color(gid), gid_color_bgr(gid)
    st = replace(Style.from_cfg(cfg, highlight_gid=gid), draw_others=opts.show_others)
    t_a, t_b = min(s.t0 for s in segs), max(s.t1 for s in segs)
    srcs = _Sources(cfg, cams)
    timings = {"main": 0.0, "thumbs": 0.0, "encode": 0.0}
    started = time.perf_counter()
    try:
        for cid in plan.camera_ids:  # fail early, before any frame is written
            srcs.need(cid)
        tracks = {
            cid: CameraTracks(conn, cid, *frame_window(cams[cid], t_a, t_b), st.trail_len)
            for cid in plan.all_camera_ids
        }
        tl_img, geo = timeline_bg(
            (lay.width - 2 * lay.margin, lay.tl_h), plan.cameras, plan.presence,
            plan.t_lo, plan.t_hi, rgb,
        )  # fmt: skip
        n_cams = len(plan.camera_ids)
        info = (
            f"{plan.n_sightings} sighting{'s' if plan.n_sightings != 1 else ''} in "
            f"{n_cams} camera{'s' if n_cams != 1 else ''}\n"
            f"visible {union_len(plan.presence):.1f} s over a {plan.t_hi - plan.t_lo:.1f} s window"
        )
        card = person_card((lay.strip_w, lay.card_h), conn, cfg, gid, info) if lay.strip_w else None  # noqa: E501
        W, H, m = lay.width, lay.height, lay.margin
        every, step, fps = tile_every(plan.fps), plan.frame_step(), plan.fps
        tiles: dict[str, np.ndarray] = {}
        done = 0
        with VideoSink(out_path, W, H, fps, cfg.render.crf) as sink:
            for si, seg in enumerate(segs):
                tiles.clear()  # a new segment changes which cameras are thumbnails
                cam, trk = cams[seg.camera_id], tracks[seg.camera_id]
                cap_sp = chip_sprite(_caption(si, gid, seg), rgb, 24, opacity=0.95)
                for k in range(seg.n_frames):
                    t0 = time.perf_counter()
                    t = seg.t0 + k * step
                    canvas = np.empty((H, W, 3), np.uint8)
                    canvas[:] = BG_BGR
                    view, _ = tile_at(cam, srcs.need(cam.id), trk, t, (lay.main_w, lay.main_h), st)
                    ramp(view, min(1.0, 0.35 + k / (0.25 * fps)) if si else 1.0)
                    age = k / fps
                    if age < 1.8:
                        op = 1.0 if age < 1.3 else (1.8 - age) / 0.5
                        paste(
                            view, cap_sp, (lay.main_w - cap_sp.w) // 2, int(lay.main_h * 0.08), op
                        )
                    canvas[m : m + lay.main_h, m : m + lay.main_w] = view
                    t1 = time.perf_counter()
                    timings["main"] += t1 - t0
                    if lay.strip_w:
                        ctx = (cams, srcs, tracks, tiles, st, bgr)
                        _paste_strip(canvas, lay, seg, ctx, t, k % every == 0)
                        sx, y0 = 2 * m + lay.main_w, lay.main_h + m - lay.card_h
                        canvas[y0 : y0 + lay.card_h, sx : sx + lay.strip_w] = card
                    t2 = time.perf_counter()
                    timings["thumbs"] += t2 - t1
                    _paste_timeline(canvas, lay, tl_img, geo, seg, t, plan)
                    sink.write(canvas)
                    timings["encode"] += time.perf_counter() - t2
                    done += 1
                    if progress and done % 25 == 0:
                        progress(done, plan.out_frames)
    finally:
        srcs.close()
    return RenderResult(Path(out_path), time.perf_counter() - started, done, timings)


def _caption(si: int, gid: int, seg: PlanSegment) -> str:
    if si == 0:
        return f"ID {gid}  ·  {seg.camera_name}"
    gap = f"   +{seg.gap_s:.1f} s" if seg.gap_s is not None else ""
    return f"→ {seg.camera_name}{gap}"


def _paste_strip(canvas, lay, seg, ctx, t: float, refresh: bool) -> None:
    """Side thumbnails of the segment's other cameras (refreshed at ~10 fps, reused between)."""
    cams, srcs, tracks, tiles, st, bgr = ctx
    sx, y = 2 * lay.margin + lay.main_w, lay.margin
    for cid in seg.thumbs[: lay.n_thumbs]:
        oc = cams[cid]
        tile = tiles.get(cid)
        if tile is None or refresh:
            src = srcs.optional(cid)
            size = (lay.tile_w, lay.tile_h)
            tile, seen = _tile(oc, src, tracks[cid], t, size, st, hud=False)
            tile_label(tile, oc.name)
            if seen:
                cv2.rectangle(tile, (0, 0), (lay.tile_w - 1, lay.tile_h - 1), bgr, 3, cv2.LINE_AA)
            tiles[cid] = tile
        canvas[y : y + lay.tile_h, sx : sx + lay.tile_w] = tile
        y += lay.tile_h + lay.margin


def _paste_timeline(canvas, lay, tl_img, geo, seg, t, plan) -> None:
    """Timeline panel with the current segment outlined and the playhead at ``t``."""
    tl = tl_img.copy()
    ya, yb = geo["rows"][seg.camera_id]
    xa = geo["X"](seg.t0)
    xb = max(geo["X"](seg.t1), xa + 3)
    cv2.rectangle(tl, (xa, ya), (xb, yb), _WHITE, 2, cv2.LINE_AA)
    xh = geo["X"](min(max(t, plan.t_lo), plan.t_hi))
    cv2.line(tl, (xh, geo["top"] - 4), (xh, geo["bot"] + 2), _WHITE, 2, cv2.LINE_AA)
    y0 = 2 * lay.margin + lay.main_h
    canvas[y0 : y0 + lay.tl_h, lay.margin : lay.width - lay.margin] = tl[: lay.height - y0]


def follow_person(
    conn: sqlite3.Connection,
    cfg: Config,
    gid: int,
    out_path: str | Path,
    progress: ProgressFn | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    options: ClipOptions | None = None,
) -> Path:
    """Render a clip that follows ``gid`` across cameras, in time order.

    ``start_s`` / ``end_s`` (shared-timeline seconds) keep only that window of the journey;
    ``options`` carries every other setting (cameras, speed, length cap, blur, thumbnails...).
    """
    opts = options or ClipOptions.from_cfg(cfg)
    if start_s is not None or end_s is not None:
        opts = replace(
            opts,
            start_s=start_s if start_s is not None else opts.start_s,
            end_s=end_s if end_s is not None else opts.end_s,
        )
    return render_clip(conn, cfg, gid, out_path, opts, progress).path


__all__ = [
    "ClipOptions",
    "RenderResult",
    "Segment",
    "draw_view",
    "follow_person",
    "plan_clip",
    "plan_segments",
    "render_clip",
]
