"""Static panels of the follow-person clip: timeline background, person card, thumbnail label."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import cv2
import numpy as np

from reid.config import Config
from reid.storage import db
from reid.viz.colors import gid_color
from reid.viz.draw import blend_rect, chip_sprite, fmt_time, paste, text_sprite
from reid.viz.privacy import strong_blur

PANEL_BGR = (30, 26, 22)


def timeline_bg(
    size: tuple[int, int],
    cams: list[tuple[str, str]],
    presence: dict[str, list[tuple[float, float]]],
    t_lo: float,
    t_hi: float,
    rgb: tuple[int, int, int],
) -> tuple[np.ndarray, dict]:
    """Static timeline panel (one row per camera); returns the image and the playhead geometry."""
    w, h = size
    img = np.empty((h, w, 3), np.uint8)
    img[:] = PANEL_BGR
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


def person_card(
    size: tuple[int, int], conn: sqlite3.Connection, cfg: Config, gid: int, info: str
) -> np.ndarray:
    """Card with the person's colour rail, best crop, id chip and ``info`` lines."""
    w, h = size
    card = np.empty((h, w, 3), np.uint8)
    card[:] = PANEL_BGR
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
        ch = max(h - 24, 8)
        cw = max(1, int(crop.shape[1] * ch / crop.shape[0]))
        cw = min(cw, w // 2)
        crop = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA)
        if cfg.render.blur_highlight:  # privacy: the followed person is not to be shown
            crop = strong_blur(crop)
        card[12 : 12 + ch, x : x + cw] = crop
        x += cw + 14
    paste(card, chip_sprite(f"ID {gid}", rgb, 18), x, 14)
    for i, line in enumerate(info.split("\n")):
        paste(card, text_sprite(line, 13, (205, 205, 210), bold=False), x, 52 + 22 * i)
    return card


def union_len(presence: dict[str, list[tuple[float, float]]]) -> float:
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


def ramp(frame: np.ndarray, a: float) -> None:
    """Fade ``frame`` in place to brightness ``a`` (1 = unchanged)."""
    if a < 1.0:
        frame[:] = (frame * a).astype(np.uint8)


def tile_label(tile: np.ndarray, name: str) -> None:
    """Camera name chip in the corner of a thumbnail."""
    sp = text_sprite(name, 12, (240, 240, 240))
    blend_rect(tile, 0, 0, sp.w + 14, sp.h + 10, (14, 12, 10), 0.65)
    paste(tile, sp, 7, 5)
