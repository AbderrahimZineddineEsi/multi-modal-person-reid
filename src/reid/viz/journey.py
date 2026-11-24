"""Follow-person clips: cut between cameras along one identity's journey (PLAN section 4)."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from reid import analytics
from reid.config import Config
from reid.storage import db
from reid.viz.colors import gid_color, gid_color_bgr
from reid.viz.draw import (
    BG_BGR,
    CameraTracks,
    Style,
    chip_sprite,
    draw_view,
    fmt_time,
    paste,
    text_sprite,
)
from reid.viz.render import ProgressFn, tile_at
from reid.viz.video_io import FrameSource, VideoSink, resolve_video

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Segment:
    """A continuous stretch of the clip shown from one camera (shared-timeline seconds)."""

    camera_id: str
    t0: float
    t1: float
    tracklet_id: int
    gap_s: float | None  # time skipped since the previous segment (None for the first)


def plan_segments(conn: sqlite3.Connection, gid: int, min_len_s: float = 1.0) -> list[Segment]:
    """Cut plan of the follow-person clip.

    Playback runs on the shared timeline. At every instant the active camera is one whose
    tracklet of ``gid`` covers it (the one lasting longest, so hand-offs between overlapping
    cameras happen as late as possible); when nobody sees the person the clip skips ahead to the
    next appearance. For non-overlapping tracklets the segments are exactly the journey's steps
    in time order. Tracklets shorter than ``min_len_s`` are shown for ``min_len_s``.
    """
    steps = analytics.journey(conn, gid)
    if not steps:
        return []

    def eff_end(s: analytics.JourneyStep) -> float:
        return max(s.end_ts, s.start_ts + min_len_s)

    segs: list[Segment] = []
    t = -np.inf
    while True:
        cands = [s for s in steps if eff_end(s) > t + 1e-9]
        if not cands:
            break
        cov = [s for s in cands if s.start_ts <= t + 1e-9]
        if cov:
            pick, start = max(cov, key=eff_end), t
        else:
            pick = min(cands, key=lambda s: (s.start_ts, s.tracklet_id))
            start = pick.start_ts
        end = eff_end(pick)
        if segs and cov and segs[-1].camera_id == pick.camera_id:
            segs[-1].t1 = end
        else:
            gap = None if not segs else max(0.0, start - segs[-1].t1)
            segs.append(Segment(pick.camera_id, start, end, pick.tracklet_id, gap))
        t = end
    return segs


# ------------------------------------------------------------------------------ layout pieces


def _timeline_bg(
    size: tuple[int, int],
    cams: list[tuple[str, str]],
    presence: dict[str, list[tuple[float, float]]],
    t_lo: float,
    t_hi: float,
    rgb: tuple[int, int, int],
) -> tuple[np.ndarray, dict]:
    """Static timeline panel; returns the image and the geometry for the playhead."""
    w, h = size
    img = np.empty((h, w, 3), np.uint8)
    img[:] = (30, 26, 22)
    fs = 12
    label_w = 110
    x0, x1 = label_w + 14, w - 20
    top, bot = 10, h - 22
    row_h = (bot - top) / max(len(cams), 1)
    span = max(t_hi - t_lo, 1e-6)

    def X(t: float) -> int:
        return int(round(x0 + (t - t_lo) / span * (x1 - x0)))

    bgr = (rgb[2], rgb[1], rgb[0])
    rows: dict[str, tuple[int, int]] = {}
    for i, (cid, name) in enumerate(cams):
        ya, yb = int(top + i * row_h + 3), int(top + (i + 1) * row_h - 3)
        rows[cid] = (ya, yb)
        cv2.rectangle(img, (x0, ya), (x1, yb), (44, 39, 34), -1, cv2.LINE_AA)
        sp = text_sprite(name, fs, (200, 200, 205), bold=False)
        paste(img, sp, 12, (ya + yb - sp.h) // 2)
        for a, b in presence.get(cid, []):
            xa, xb = X(a), max(X(b), X(a) + 3)
            cv2.rectangle(img, (xa, ya), (xb, yb), bgr, -1, cv2.LINE_AA)
    for k in range(5):
        tt = t_lo + span * k / 4
        sp = text_sprite(fmt_time(tt), 11, (150, 150, 158), bold=False)
        xx = X(tt)
        paste(img, sp, int(np.clip(xx - sp.w // 2, x0 - 10, w - sp.w - 4)), h - sp.h - 4)
    return img, {"X": X, "rows": rows, "top": top, "bot": bot}


def _person_card(
    size: tuple[int, int], conn: sqlite3.Connection, cfg: Config, gid: int, info: str
) -> np.ndarray:
    w, h = size
    card = np.empty((h, w, 3), np.uint8)
    card[:] = (30, 26, 22)
    rgb = gid_color(gid)
    cv2.rectangle(card, (0, 0), (5, h), (rgb[2], rgb[1], rgb[0]), -1)
    x = 18
    crop = None
    for t in db.get_tracklets(conn, gid=gid):
        if t.best_crop_path:
            p = Path(t.best_crop_path)
            p = p if p.is_absolute() else cfg.data_dir / p
            crop = cv2.imread(str(p)) if p.exists() else None
            if crop is not None:
                break
    if crop is not None:
        ch = h - 24
        cw = max(1, int(crop.shape[1] * ch / crop.shape[0]))
        cw = min(cw, w // 2)
        crop = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA)
        card[12 : 12 + ch, x : x + cw] = crop
        x += cw + 14
    paste(card, chip_sprite(f"ID {gid}", rgb, 18), x, 14)
    for i, line in enumerate(info.split("\n")):
        paste(card, text_sprite(line, 13, (205, 205, 210), bold=False), x, 52 + 22 * i)
    return card


def _union_len(presence: dict[str, list[tuple[float, float]]]) -> float:
    """Total time covered by at least one camera (overlaps counted once)."""
    ivs = sorted(iv for v in presence.values() for iv in v)
    tot, cur_a, cur_b = 0.0, None, None
    for a, b in ivs:
        if cur_b is None or a > cur_b:
            tot += 0.0 if cur_a is None else cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    return tot + (0.0 if cur_a is None else cur_b - cur_a)


def _ramp(frame: np.ndarray, a: float) -> None:
    if a < 1.0:
        frame[:] = (frame * a).astype(np.uint8)


# ------------------------------------------------------------------------------ main entry


def follow_person(
    conn: sqlite3.Connection,
    cfg: Config,
    gid: int,
    out_path: str | Path,
    progress: ProgressFn | None = None,
) -> Path:
    """Render a clip that follows ``gid`` across cameras, in time order.

    The active camera is shown large with the person spotlighted, the other cameras sit in a
    strip on the right (same instant), and a timeline at the bottom shows the person's presence
    per camera with a moving playhead. A caption announces every cut ("-> cam_02, +4.2 s").
    """
    segs = plan_segments(conn, gid)
    if not segs:
        raise ValueError(f"identity {gid} has no tracklets")
    steps = analytics.journey(conn, gid)
    cams = {c.id: c for c in db.list_cameras(conn)}
    rgb = gid_color(gid)
    st = Style.from_cfg(cfg, highlight_gid=gid)
    srcs = {
        cid: FrameSource(resolve_video(c.video_path, cfg.data_dir), (c.width, c.height))
        for cid, c in cams.items()
    }
    try:
        tracks = {cid: CameraTracks(conn, cid, 0, 2**31 - 1, st.trail_len) for cid in cams}
        aspect = srcs[segs[0].camera_id].height / srcs[segs[0].camera_id].width
        W = cfg.render.journey_width // 2 * 2
        m = 12
        strip_w = int(W * 0.26) // 2 * 2
        main_w = W - strip_w - 3 * m
        main_h = int(main_w * aspect) // 2 * 2
        tl_h = int(np.clip(34 + 26 * len(cams), 90, 190))
        H = (main_h + tl_h + 3 * m) // 2 * 2
        others_n = max(len(cams) - 1, 1)
        card_h = 96
        tile_h = int(min(strip_w * aspect, (main_h - card_h - m * others_n) / others_n))
        tile_w = int(tile_h / aspect) // 2 * 2
        tile_h = tile_h // 2 * 2

        t_lo = min(s.start_ts for s in steps)
        t_hi = max(s.end_ts for s in steps)
        presence: dict[str, list[tuple[float, float]]] = {}
        for s in steps:
            presence.setdefault(s.camera_id, []).append((s.start_ts, s.end_ts))
        tl_img, geo = _timeline_bg(
            (W - 2 * m, tl_h), [(c.id, c.name) for c in cams.values()], presence, t_lo, t_hi, rgb
        )
        total = t_hi - t_lo
        n_cams_seen = len({s.camera_id for s in steps})
        info = (
            f"{len(steps)} sighting{'s' if len(steps) != 1 else ''} in "
            f"{n_cams_seen} camera{'s' if n_cams_seen != 1 else ''}\n"
            f"visible {_union_len(presence):.1f} s over a {total:.1f} s window"
        )
        card = _person_card((strip_w, card_h), conn, cfg, gid, info)

        fps = cfg.render.out_fps or min(30.0, max(s.fps for s in srcs.values()))
        n_total = sum(max(1, int((s.t1 - s.t0) * fps)) for s in segs)
        done = 0
        bgr = gid_color_bgr(gid)
        with VideoSink(out_path, W, H, fps, cfg.render.crf) as sink:
            for si, seg in enumerate(segs):
                cam = cams[seg.camera_id]
                n = max(1, int((seg.t1 - seg.t0) * fps))
                if si == 0:
                    cap = f"ID {gid}  ·  {cam.name}"
                else:
                    gap = f"   +{seg.gap_s:.1f} s" if seg.gap_s is not None else ""
                    cap = f"→ {cam.name}{gap}"
                cap_sp = chip_sprite(cap, rgb, 24, opacity=0.95)
                for k in range(n):
                    t = seg.t0 + k / fps
                    canvas = np.empty((H, W, 3), np.uint8)
                    canvas[:] = BG_BGR
                    view, _ = tile_at(cam, srcs[cam.id], tracks[cam.id], t, (main_w, main_h), st)
                    _ramp(view, min(1.0, 0.35 + k / (0.25 * fps)) if si else 1.0)
                    age = k / fps
                    if age < 1.8:
                        op = 1.0 if age < 1.3 else (1.8 - age) / 0.5
                        paste(view, cap_sp, (main_w - cap_sp.w) // 2, int(main_h * 0.08), op)
                    canvas[m : m + main_h, m : m + main_w] = view
                    sx, y = 2 * m + main_w, m
                    for oc in cams.values():
                        if oc.id == cam.id:
                            continue
                        tile, seen = tile_at(
                            oc, srcs[oc.id], tracks[oc.id], t, (tile_w, tile_h), st, hud=False
                        )
                        _label(tile, oc.name)
                        if seen:
                            cv2.rectangle(
                                tile, (0, 0), (tile_w - 1, tile_h - 1), bgr, 3, cv2.LINE_AA
                            )
                        canvas[y : y + tile_h, sx : sx + tile_w] = tile
                        y += tile_h + m
                    canvas[main_h + m - card_h : main_h + m, sx : sx + strip_w] = card
                    tl = tl_img.copy()
                    ya, yb = geo["rows"][cam.id]
                    xa = geo["X"](seg.t0)
                    xb = max(geo["X"](seg.t1), xa + 3)
                    cv2.rectangle(tl, (xa, ya), (xb, yb), (255, 255, 255), 2, cv2.LINE_AA)
                    xh = geo["X"](min(max(t, t_lo), t_hi))
                    cv2.line(
                        tl,
                        (xh, geo["top"] - 4),
                        (xh, geo["bot"] + 2),
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    canvas[2 * m + main_h : 2 * m + main_h + tl_h, m : W - m] = tl[
                        : H - 2 * m - main_h
                    ]
                    sink.write(canvas)
                    done += 1
                    if progress and done % 25 == 0:
                        progress(done, n_total)
    finally:
        for s in srcs.values():
            s.close()
    return Path(out_path)


def _label(tile: np.ndarray, name: str) -> None:
    sp = text_sprite(name, 12, (240, 240, 240))
    from reid.viz.draw import blend_rect

    blend_rect(tile, 0, 0, sp.w + 14, sp.h + 10, (14, 12, 10), 0.65)
    paste(tile, sp, 7, 5)


__all__ = ["Segment", "draw_view", "follow_person", "plan_segments"]
