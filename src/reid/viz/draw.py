"""Drawing primitives: anti-aliased rounded boxes, PIL text chips, trails, HUD, track lookup.

Shapes use ``cv2`` with ``LINE_AA`` and 4-bit sub-pixel coordinates; text is rendered once with
PIL (DejaVu Sans from matplotlib's font directory) into cached sprites and alpha-composited,
so per-frame cost stays low.
"""

from __future__ import annotations

import functools
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from reid.config import Config
from reid.storage import db
from reid.viz.colors import RGB, gid_color, text_on
from reid.viz.video_io import display_resize

if TYPE_CHECKING:
    from reid.viz.privacy import PrivacyIndex

_SS = 3  # chip supersampling factor
BG_BGR = (22, 18, 15)  # dark canvas (RGB 15,18,22)


# ------------------------------------------------------------------------------ fonts / text


@functools.lru_cache(maxsize=16)
def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    """DejaVu Sans (bundled with matplotlib) at ``size`` px."""
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), size)


@dataclass(slots=True)
class Sprite:
    """BGR pixels with a float alpha channel, ready to composite."""

    bgr: np.ndarray  # (h, w, 3) uint8
    alpha: np.ndarray  # (h, w, 1) float32

    @property
    def h(self) -> int:
        return self.bgr.shape[0]

    @property
    def w(self) -> int:
        return self.bgr.shape[1]


def _to_sprite(img: Image.Image) -> Sprite:
    a = np.asarray(img, dtype=np.uint8)
    return Sprite(np.ascontiguousarray(a[..., 2::-1]), (a[..., 3:4] / 255.0).astype(np.float32))


@functools.lru_cache(maxsize=4096)
def text_sprite(text: str, size: int, rgb: RGB = (255, 255, 255), bold: bool = True) -> Sprite:
    """Anti-aliased text on a transparent background."""
    f = font(size, bold)
    x0, t, r, b = f.getbbox(text)
    img = Image.new("RGBA", (max(r - x0, 1) + 2, max(b - t, 1) + 2), (*rgb, 0))
    ImageDraw.Draw(img).text((1 - x0, 1 - t), text, font=f, fill=(*rgb, 255))
    return _to_sprite(img)


@functools.lru_cache(maxsize=1024)
def chip_sprite(
    text: str, fill: RGB, size: int, check: bool = False, opacity: float = 0.92
) -> Sprite:
    """Rounded label chip (``text`` on ``fill``) with an optional face check mark."""
    f = font(size * _SS, True)
    fg = text_on(fill)
    label = text + ("  ✓" if check else "")
    x0, t, r, b = f.getbbox(label)
    pad_x, pad_y = int(size * 0.55) * _SS, int(size * 0.28) * _SS
    w, h = r - x0 + 2 * pad_x, int(size * 1.0 * _SS) + 2 * pad_y
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2 - 1, fill=(*fill, int(255 * opacity)))
    ty = (h - (b - t)) // 2 - t
    d.text((pad_x - x0, ty), label, font=f, fill=(*fg, 255))
    return _to_sprite(img.resize((w // _SS, h // _SS), Image.LANCZOS))


def paste(frame: np.ndarray, sp: Sprite, x: int, y: int, opacity: float = 1.0) -> None:  # in place
    """Alpha-composite ``sp`` at top-left ``(x, y)`` (clipped to the frame)."""
    H, W = frame.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sp.w, W), min(y + sp.h, H)
    if x1 <= x0 or y1 <= y0:
        return
    s_bgr = sp.bgr[y0 - y : y1 - y, x0 - x : x1 - x]
    a = sp.alpha[y0 - y : y1 - y, x0 - x : x1 - x] * opacity
    roi = frame[y0:y1, x0:x1]
    roi[:] = (roi * (1.0 - a) + s_bgr * a + 0.5).astype(np.uint8)


def blend_rect(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, bgr, alpha: float) -> None:
    """Fill a rectangle with a translucent colour (in place)."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    roi[:] = (roi * (1.0 - alpha) + np.array(bgr, np.float32) * alpha + 0.5).astype(np.uint8)


# ------------------------------------------------------------------------------ boxes / trails


def _rrect(x1: float, y1: float, x2: float, y2: float, r: float, n: int = 6) -> np.ndarray:
    """Polygon of a rounded rectangle in 1/16-pixel fixed point (``cv2`` shift=4)."""
    r = max(0.0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    pts: list[tuple[float, float]] = []
    for cx, cy, a0 in (
        (x2 - r, y1 + r, -90),
        (x2 - r, y2 - r, 0),
        (x1 + r, y2 - r, 90),
        (x1 + r, y1 + r, 180),
    ):
        for k in range(n + 1):
            a = np.deg2rad(a0 + 90.0 * k / n)
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.round(np.array(pts) * 16).astype(np.int32)


def draw_box(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    bgr: tuple[int, int, int],
    thickness: float = 2.0,
    fill_alpha: float = 0.16,
    opacity: float = 1.0,
    radius: float | None = None,
) -> None:
    """Anti-aliased rounded box with a translucent fill, in place."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    m = int(np.ceil(thickness)) + 2
    rx0, ry0 = max(int(x1) - m, 0), max(int(y1) - m, 0)
    rx1, ry1 = min(int(x2) + m + 1, W), min(int(y2) + m + 1, H)
    if rx1 <= rx0 or ry1 <= ry0:
        return
    rad = radius if radius is not None else min(12.0, 0.18 * min(x2 - x1, y2 - y1))
    poly = _rrect(x1 - rx0, y1 - ry0, x2 - rx0, y2 - ry0, rad)
    shape = (ry1 - ry0, rx1 - rx0)
    fill = np.zeros(shape, np.uint8)
    cv2.fillPoly(fill, [poly], 255, cv2.LINE_AA, shift=4)
    line = np.zeros(shape, np.uint8)
    cv2.polylines(line, [poly], True, 255, max(1, int(round(thickness))), cv2.LINE_AA, shift=4)
    a = np.maximum(fill / 255.0 * fill_alpha, line / 255.0) * opacity
    roi = frame[ry0:ry1, rx0:rx1]
    roi[:] = (roi * (1 - a[..., None]) + np.array(bgr, np.float32) * a[..., None] + 0.5).astype(
        np.uint8
    )


def draw_trail(
    frame: np.ndarray,
    feet: np.ndarray,
    bgr: tuple[int, int, int],
    width: float = 3.0,
    opacity: float = 1.0,
) -> None:
    """Fading polyline through ``feet`` (oldest first) ending in a dot at the newest point."""
    n = len(feet)
    if n < 1:
        return
    H, W = frame.shape[:2]
    pad = int(width) + 6
    lo = np.floor(feet.min(0)).astype(int) - pad
    hi = np.ceil(feet.max(0)).astype(int) + pad
    x0, y0 = max(lo[0], 0), max(lo[1], 0)
    x1, y1 = min(hi[0], W), min(hi[1], H)
    if x1 <= x0 or y1 <= y0:
        return
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    pts = np.round((feet - (x0, y0)) * 16).astype(np.int32)
    for i in range(1, n):
        f = i / (n - 1) if n > 1 else 1.0
        cv2.line(
            mask, tuple(pts[i - 1]), tuple(pts[i]), int(255 * f**1.5),
            max(1, int(round(width * (0.35 + 0.65 * f)))), cv2.LINE_AA, shift=4,
        )  # fmt: skip
    cv2.circle(mask, tuple(pts[-1]), int(round(width * 1.3 * 16)) // 16 + 1, 255, -1, cv2.LINE_AA)
    a = (mask / 255.0 * opacity)[..., None]
    roi = frame[y0:y1, x0:x1]
    roi[:] = (roi * (1 - a) + np.array(bgr, np.float32) * a + 0.5).astype(np.uint8)


# ------------------------------------------------------------------------------ tracks


@dataclass(slots=True)
class Det:
    """One person to draw in one frame (pixel coordinates of the *source* frame)."""

    tracklet_id: int
    gid: int | None
    box: tuple[float, float, float, float]
    trail: np.ndarray  # (n, 2) foot points, oldest first, including the current one
    has_face: bool = False


@dataclass
class Style:
    """Rendering style, derived from :class:`reid.config.RenderCfg`."""

    thickness: float = 2.0
    fill_alpha: float = 0.16
    dim_alpha: float = 0.4
    spotlight_brightness: float = 0.62
    spotlight_margin: float = 1.0
    blur_others: bool = False
    blur_strength: float = 0.35
    blur_keep_gids: frozenset[int] = frozenset()  # never blurred (besides the highlighted gid)
    blur_highlight: bool = False  # True: the highlighted person is blurred like anybody else
    privacy: PrivacyIndex | None = None  # extra boxes to blur, from a ground-truth CSV
    draw_others: bool = True  # False: only the highlighted person gets a box (others stay blurred)
    trail_len: int = 30
    hud: bool = True
    unidentified: RGB = (128, 128, 128)
    highlight_gid: int | None = None
    gid_filter: frozenset[int] | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_cfg(cls, cfg: Config, highlight_gid: int | None = None, gid_filter=None) -> Style:
        r = cfg.render
        unk = tuple(int(c) for c in r.unidentified_color)  # config is BGR
        privacy = None
        if r.privacy_gt:
            from reid.viz.privacy import load_privacy_boxes

            privacy = load_privacy_boxes(r.privacy_gt, r.privacy_keep_gt)
        return cls(
            thickness=float(r.box_thickness),
            fill_alpha=r.fill_alpha,
            dim_alpha=r.dim_alpha,
            spotlight_brightness=r.spotlight_brightness,
            spotlight_margin=r.spotlight_margin,
            blur_others=r.blur_others,
            blur_strength=r.blur_strength,
            blur_keep_gids=frozenset(r.blur_keep_gids),
            blur_highlight=r.blur_highlight,
            privacy=privacy,
            trail_len=r.trail_len,
            hud=r.show_hud,
            unidentified=(unk[2], unk[1], unk[0]),
            highlight_gid=highlight_gid,
            gid_filter=frozenset(gid_filter) if gid_filter is not None else None,
        )


class CameraTracks:
    """All track points of one camera in a frame range, indexed for per-frame lookup.

    The points are loaded with a single query; ``dets_at`` then needs no database access.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        camera_id: str,
        start_frame: int = 0,
        end_frame: int = 2**31 - 1,
        trail_len: int = 30,
    ) -> None:
        self.camera_id = camera_id
        cam = db.get_camera(conn, camera_id)
        if cam is None:
            raise KeyError(f"unknown camera {camera_id!r}")
        self.camera = cam
        self.trail_len = trail_len
        self.hold = 3  # frames a stored box is kept on screen when the next frames have none
        tl = db.get_tracklets(conn, camera_id=camera_id)
        self.gid_of: dict[int, int | None] = {t.id or 0: t.gid for t in tl}
        face_gids = {i.gid for i in db.list_identities(conn) if i.has_face}
        face_gids |= {t.gid for t in db.get_tracklets(conn) if t.gid and t.face_quality is not None}
        self.face_gids = face_gids
        pts = db.get_points(conn, camera_id, max(0, start_frame - trail_len), end_frame)
        self.pts = pts
        self._frames = pts["frame"]
        self._tr: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for tid in np.unique(pts["tracklet_id"]):
            p = pts[pts["tracklet_id"] == tid]
            p = p[np.argsort(p["frame"], kind="stable")]
            feet = np.stack([(p["x1"] + p["x2"]) / 2.0, p["y2"]], axis=1).astype(np.float64)
            self._tr[int(tid)] = (p["frame"], feet)

    def dets_at(self, frame: int, gid_filter: frozenset[int] | None = None) -> list[Det]:
        """People present at ``frame`` (with their recent foot trail).

        Runs processed with ``frame_stride`` > 1 store points only every n-th frame; a frame with
        no points shows the closest earlier stored frame (up to ``self.hold`` frames back), so
        boxes do not blink on the frames in between.
        """
        for back in range(self.hold + 1):
            lo = int(np.searchsorted(self._frames, frame - back, "left"))
            hi = int(np.searchsorted(self._frames, frame - back, "right"))
            if hi > lo:
                break
        out: list[Det] = []
        for r in self.pts[lo:hi]:
            tid = int(r["tracklet_id"])
            gid = self.gid_of.get(tid)
            if gid_filter is not None and gid not in gid_filter:
                continue
            fr, feet = self._tr[tid]
            j = int(np.searchsorted(fr, frame, "right"))
            i = int(np.searchsorted(fr, frame - self.trail_len, "right"))
            out.append(
                Det(
                    tid,
                    gid,
                    (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])),
                    feet[i:j],
                    gid in self.face_gids if gid is not None else False,
                )
            )
        return out

    def gids_at(self, frame: int) -> set[int]:
        return {d.gid for d in self.dets_at(frame) if d.gid is not None}


# ------------------------------------------------------------------------------ frame drawing


def _spotlight(
    frame: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    brightness: float = 0.62,
    margin: float = 1.0,
) -> None:
    """Darken the frame to ``brightness`` (0 = black), then restore a feathered ellipse around
    each highlighted box, ``margin`` times the default halo (0 = exactly the box, no halo)."""
    orig = frame.copy() if boxes else None
    frame[:] = (frame * brightness).astype(np.uint8)
    if orig is None:
        return
    H, W = frame.shape[:2]
    if margin <= 0:
        for x1, y1, x2, y2 in boxes:
            a, b = max(int(x1), 0), max(int(y1), 0)
            c, d = min(int(np.ceil(x2)), W), min(int(np.ceil(y2)), H)
            if c > a and d > b:
                frame[b:d, a:c] = orig[b:d, a:c]
        return
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        mx, my = (bw * 0.45 + 8) * margin, (bh * 0.25 + 8) * margin
        rx0, ry0 = max(int(x1 - mx), 0), max(int(y1 - my), 0)
        rx1, ry1 = min(int(x2 + mx), W), min(int(y2 + my), H)
        if rx1 <= rx0 or ry1 <= ry0:
            continue
        m = np.zeros((ry1 - ry0, rx1 - rx0), np.float32)
        c = ((x1 + x2) / 2 - rx0, (y1 + y2) / 2 - ry0)
        cv2.ellipse(
            m, (int(c[0]), int(c[1])), (int(bw / 2 + mx * 0.5), int(bh / 2 + my * 0.5)),
            0, 0, 360, 1.0, -1, cv2.LINE_AA,
        )  # fmt: skip
        k = int(max(bw, bh) * 0.25) | 1
        m = cv2.GaussianBlur(m, (k, k), 0)[..., None]
        roi = frame[ry0:ry1, rx0:rx1]
        roi[:] = (roi * (1 - m) + orig[ry0:ry1, rx0:rx1] * m + 0.5).astype(np.uint8)


def label_font_px(H: int) -> int:
    """Label text size (px) for a frame of height ``H``: ~2.4 % of it, within 11..56."""
    return int(np.clip(round(H * 0.024), 11, 56))


def line_thickness(H: int, base: float = 2.0) -> float:
    """Box line width for height ``H``: ``base`` px at 720p and below, growing linearly above."""
    return float(base * np.clip(H / 720.0, 1.0, 4.0))


def box_radius(box: tuple[float, float, float, float], H: int) -> float:
    x1, y1, x2, y2 = box
    return min(12.0 * max(H / 720.0, 1.0), 0.18 * min(x2 - x1, y2 - y1))


def label_of(d: Det) -> str:
    return f"ID {d.gid}" if d.gid is not None else "?"


def draw_dets(
    frame: np.ndarray,
    dets: list[Det],
    st: Style,
    scale: float = 1.0,
    offset: tuple[float, float] = (0, 0),
) -> None:
    """Draw boxes, trails and label chips for ``dets`` (source coords mapped by scale/offset)."""
    hl = st.highlight_gid
    H = frame.shape[0]
    fs = label_font_px(H)
    ox, oy = offset

    def tbox(d: Det) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = d.box
        return (x1 * scale + ox, y1 * scale + oy, x2 * scale + ox, y2 * scale + oy)

    if hl is not None:
        boxes = [tbox(d) for d in dets if d.gid == hl]
        _spotlight(frame, boxes, st.spotlight_brightness, st.spotlight_margin)
    th = line_thickness(H, st.thickness)  # in output pixels
    for d in sorted(dets, key=lambda d: (d.gid == hl, d.tracklet_id)):
        rgb = gid_color(d.gid, st.unidentified)
        bgr = (rgb[2], rgb[1], rgb[0])
        box = tbox(d)
        dim = hl is not None and d.gid != hl
        strong = hl is not None and d.gid == hl
        if dim:
            draw_box(frame, box, bgr, max(1.0, th * 0.7), 0.0, st.dim_alpha)
            continue
        tw = th * (1.8 if strong else 1.0)
        if len(d.trail) > 1 and st.trail_len > 0:
            feet = d.trail * scale + np.array([ox, oy])
            draw_trail(frame, feet, bgr, width=tw * (1.5 if strong else 1.2))
        draw_box(
            frame, box, bgr, tw, st.fill_alpha * (1.7 if strong else 1.0), radius=box_radius(box, H)
        )
        chip = chip_sprite(label_of(d), rgb, int(fs * (1.15 if strong else 1.0)), d.has_face)
        gap = max(3, int(round(tw)))
        cx, cy = int(box[0]), int(box[1] - chip.h - gap)
        if cy < 0:
            cy = int(box[1] + tw + gap)
        # keep the label inside the frame
        cx = int(np.clip(cx, 0, max(frame.shape[1] - chip.w, 0)))
        cy = int(np.clip(cy, 0, max(H - chip.h, 0)))
        paste(frame, chip, cx, cy)


def fmt_time(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60.0)
    return f"{int(m):02d}:{s:04.1f}"


def draw_hud(frame: np.ndarray, name: str, t: float, n_people: int) -> None:
    """Translucent top bar: camera name (left), clock and people count (right)."""
    H, W = frame.shape[:2]
    fs = int(np.clip(round(H * 0.026), 11, 60))
    bh = int(fs * 2.2)
    blend_rect(frame, 0, 0, W, bh, (14, 12, 10), 0.62)
    cv2.circle(frame, (int(bh * 0.5), bh // 2), max(3, fs // 3), (60, 60, 235), -1, cv2.LINE_AA)
    nm = text_sprite(name, fs, (245, 245, 245))
    paste(frame, nm, int(bh * 0.5) + fs, (bh - nm.h) // 2)
    right = text_sprite(
        f"{fmt_time(t)}   |   {n_people} {'person' if n_people == 1 else 'people'}",
        fs,
        (225, 225, 225),
        bold=False,
    )
    paste(frame, right, W - right.w - int(fs * 0.9), (bh - right.h) // 2)


def letterbox(
    frame: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Fit ``frame`` in ``size`` (w, h); returns ``(image, scale, (x_off, y_off))``."""
    w, h = size
    fh, fw = frame.shape[:2]
    s = min(w / fw, h / fh)
    nw, nh = max(1, int(round(fw * s))), max(1, int(round(fh * s)))
    if (nw, nh) == (fw, fh):
        small = frame.copy()
    else:
        small = display_resize(frame, (nw, nh))
    if (nw, nh) == (w, h):
        return small, s, (0, 0)
    canvas = np.empty((h, w, 3), np.uint8)
    canvas[:] = BG_BGR
    ox, oy = (w - nw) // 2, (h - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = small
    return canvas, s, (ox, oy)


def draw_view(
    frame: np.ndarray | None,
    tracks: CameraTracks,
    frame_idx: int,
    size: tuple[int, int],
    st: Style,
    t: float | None = None,
    name: str | None = None,
    hud: bool | None = None,
) -> np.ndarray:
    """Annotated camera view of exactly ``size`` (letterboxed); "no signal" tile if no frame."""
    w, h = size
    name = name or tracks.camera.name
    if frame is None:
        img = np.empty((h, w, 3), np.uint8)
        img[:] = BG_BGR
        sp = text_sprite("NO SIGNAL", int(np.clip(h * 0.06, 12, 40)), (150, 150, 155))
        paste(img, sp, (w - sp.w) // 2, (h - sp.h) // 2)
        if st.hud if hud is None else hud:
            draw_hud_no_signal(img, name, t or 0.0)
        return img
    img, s, off = letterbox(frame, size)
    dets = tracks.dets_at(frame_idx, st.gid_filter)
    if st.blur_others or st.privacy:
        _privacy_blur(img, tracks.camera_id, frame_idx, dets, st, s, off)
    n_people = len(dets)
    if not st.draw_others and st.highlight_gid is not None:
        dets = [d for d in dets if d.gid == st.highlight_gid]
    draw_dets(img, dets, st, s, off)
    if st.hud if hud is None else hud:
        draw_hud(img, name, t if t is not None else frame_idx / tracks.camera.fps, n_people)
    return img


def _privacy_blur(
    img: np.ndarray,
    camera_id: str,
    frame_idx: int,
    dets: list[Det],
    st: Style,
    scale: float,
    off: tuple[float, float],
) -> None:
    """Blur the other people (stored boxes and the optional ground-truth boxes) in place.

    The highlighted person's own box is restored afterwards, except where another person's box
    overlaps it (someone walking right behind or in front of them stays blurred).
    """
    from reid.viz.privacy import blur_boxes

    hl = None if st.blur_highlight else st.highlight_gid
    ox, oy = off

    def tb(b) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = b
        return (x1 * scale + ox, y1 * scale + oy, x2 * scale + ox, y2 * scale + oy)

    keep_gids = st.blur_keep_gids | ({hl} if hl is not None else set())
    others = [tb(d.box) for d in dets if d.gid not in keep_gids] if st.blur_others else []
    if st.privacy:
        extra = st.privacy.get((camera_id, frame_idx))
        if extra is not None:
            others += [tb(b) for b in extra]
    mine = [tb(d.box) for d in dets if d.gid in keep_gids]
    orig = img.copy() if mine else None
    blur_boxes(img, others, st.blur_strength)
    if orig is None:
        return
    H, W = img.shape[:2]
    keep = np.zeros((H, W), bool)
    for x1, y1, x2, y2 in mine:
        keep[
            max(int(y1), 0) : max(int(np.ceil(y2)), 0), max(int(x1), 0) : max(int(np.ceil(x2)), 0)
        ] = True
    for x1, y1, x2, y2 in others:
        keep[
            max(int(y1), 0) : max(int(np.ceil(y2)), 0), max(int(x1), 0) : max(int(np.ceil(x2)), 0)
        ] = False
    img[keep] = orig[keep]


def draw_hud_no_signal(frame: np.ndarray, name: str, t: float) -> None:
    H, W = frame.shape[:2]
    fs = int(np.clip(round(H * 0.026), 11, 60))
    bh = int(fs * 2.2)
    blend_rect(frame, 0, 0, W, bh, (40, 36, 32), 0.7)
    nm = text_sprite(name, fs, (170, 170, 175))
    paste(frame, nm, int(bh * 0.5), (bh - nm.h) // 2)
    right = text_sprite(fmt_time(t), fs, (130, 130, 135), bold=False)
    paste(frame, right, W - right.w - int(fs * 0.9), (bh - right.h) // 2)
