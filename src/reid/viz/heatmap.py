"""Foot-point heatmaps, transitions matrix and floor-plan plots (fixes v1 B17).

v1 accumulated one blob per sighting and read a ``.png`` background that was saved as ``.jpg``.
Here every stored track point contributes (so dwell time shows), and the background is looked
up by the path that is actually stored.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

from reid import analytics  # noqa: E402
from reid.config import Config  # noqa: E402
from reid.storage import db  # noqa: E402
from reid.viz.colors import gid_color, hex_color  # noqa: E402
from reid.viz.video_io import FrameSource, resolve_video  # noqa: E402

_INK, _MUTED = "#1f2430", "#6b7280"


# ------------------------------------------------------------------------------ background


def _read_image(path: Path) -> np.ndarray | None:
    for p in (path, path.with_suffix(".png"), path.with_suffix(".jpg"), path.with_suffix(".jpeg")):
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img
    return None


def camera_background(
    conn: sqlite3.Connection, cfg: Config, camera_id: str, n_frames: int = 15
) -> np.ndarray:
    """Background image of a camera (BGR).

    Order: the camera's ``background_path`` (absolute or relative to ``data_dir``; the other
    image extension is tried too), a cached background, else the temporal median of
    ``n_frames`` frames spread over the video (which removes moving people), cached as
    ``data_dir/cache/background_<camera>.jpg``.
    """
    cam = db.get_camera(conn, camera_id)
    if cam is None:
        raise KeyError(f"unknown camera {camera_id!r}")
    if cam.background_path:
        bp = Path(cam.background_path).expanduser()
        for cand in (bp, cfg.data_dir / bp):
            img = _read_image(cand)
            if img is not None:
                return img
    cache = cfg.data_dir / "cache" / f"background_{camera_id}.jpg"
    img = _read_image(cache)
    if img is not None:
        return img
    with FrameSource(
        resolve_video(cam.video_path, cfg.data_dir, cam.id), (cam.width, cam.height)
    ) as src:
        n = max(src.n_frames, 1)
        frames = []
        for i in np.linspace(0, n - 1, num=min(n_frames, n)).astype(int):
            f = src.get(int(i))
            if f is not None:
                frames.append(f)
    if not frames:
        raise RuntimeError(f"cannot read frames of {cam.video_path}")
    img = np.median(np.stack(frames), axis=0).astype(np.uint8) if len(frames) > 2 else frames[0]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cache), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return img


# ------------------------------------------------------------------------------ foot points


def foot_points(conn: sqlite3.Connection, camera_id: str, gid: int | None = None) -> np.ndarray:
    """All foot points ``(n, 2)`` (bottom-centre of every stored box) of a camera / identity."""
    tl = db.get_tracklets(conn, camera_id=camera_id, gid=gid)
    if not tl:
        return np.zeros((0, 2))
    ids = {t.id for t in tl}
    pts = db.get_points(conn, camera_id, 0, 2**31 - 1)
    pts = pts[np.isin(pts["tracklet_id"], list(ids))]
    return np.stack([(pts["x1"] + pts["x2"]) / 2.0, pts["y2"]], axis=1).astype(np.float64)


def foot_heatmap(
    conn: sqlite3.Connection,
    cfg: Config,
    camera_id: str,
    gid: int | None = None,
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Density of foot points as a float ``(h, w)`` array normalized to max 1.

    Accumulated at ``heat_scale`` x the frame size (``size`` = ``(width, height)`` overrides the
    camera's), blurred with ``heat_sigma`` (in accumulation pixels), one count per track point.
    """
    cam = db.get_camera(conn, camera_id)
    if cam is None:
        raise KeyError(f"unknown camera {camera_id!r}")
    w, h = size or (cam.width, cam.height)
    s = cfg.render.heat_scale
    hw, hh = max(int(round(w * s)), 2), max(int(round(h * s)), 2)
    acc = np.zeros((hh, hw), np.float32)
    pts = foot_points(conn, camera_id, gid)
    if len(pts):
        ix = np.clip((pts[:, 0] * s).astype(int), 0, hw - 1)
        iy = np.clip((pts[:, 1] * s).astype(int), 0, hh - 1)
        np.add.at(acc, (iy, ix), 1.0)
        acc = cv2.GaussianBlur(acc, (0, 0), cfg.render.heat_sigma)
    m = acc.max()
    return acc / m if m > 0 else acc


def overlay_heat(
    bg: np.ndarray, heat: np.ndarray, alpha: float, cmap: str = "inferno"
) -> np.ndarray:
    """Blend a normalized heat map on ``bg`` (opacity grows with heat; empty areas stay clear)."""
    h, w = bg.shape[:2]
    hm = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    rgb = (plt.get_cmap(cmap)(hm)[..., :3] * 255).astype(np.float32)[..., ::-1]
    a = (np.clip(hm * 1.6, 0, 1) ** 0.8 * alpha)[..., None]
    base = bg.astype(np.float32) * 0.8
    return (base * (1 - a) + rgb * a).clip(0, 255).astype(np.uint8)


def render_heatmap(
    conn: sqlite3.Connection,
    cfg: Config,
    camera_id: str,
    out_path: str | Path,
    gid: int | None = None,
) -> Path:
    """Foot-point heatmap over the camera background, saved as an image (PNG/JPG by extension)."""
    from reid.viz.draw import blend_rect, paste, text_sprite

    bg = camera_background(conn, cfg, camera_id)
    heat = foot_heatmap(conn, cfg, camera_id, gid, size=(bg.shape[1], bg.shape[0]))
    img = overlay_heat(bg, heat, cfg.render.heat_alpha)
    n = len(foot_points(conn, camera_id, gid))
    cam = db.get_camera(conn, camera_id)
    title = f"{cam.name if cam else camera_id}" + (f"  ·  ID {gid}" if gid is not None else "")
    fs = int(np.clip(img.shape[0] * 0.03, 12, 28))
    blend_rect(img, 0, 0, img.shape[1], int(fs * 2.3), (14, 12, 10), 0.65)
    t = text_sprite(title, fs, (245, 245, 245))
    paste(img, t, int(fs * 0.8), int(fs * 0.55))
    r = text_sprite(f"{n} foot points", fs, (210, 210, 215), bold=False)
    paste(img, r, img.shape[1] - r.w - int(fs * 0.8), int(fs * 0.6))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return Path(out_path)


# ------------------------------------------------------------------------------ transitions


def plot_transitions(
    conn: sqlite3.Connection, out_path: str | Path | None = None, run_id: int | None = None
):
    """Camera-to-camera transition counts as an annotated matrix (matplotlib figure)."""
    cams, mat = analytics.transitions(conn, run_id)
    names = {c.id: c.name for c in db.list_cameras(conn)}
    labels = [names.get(c, c) for c in cams]
    n = max(len(cams), 1)
    fig, ax = plt.subplots(figsize=(1.6 + 0.9 * n, 1.4 + 0.8 * n), dpi=160)
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(int(mat.max()), 1))
    ax.set_xticks(range(len(cams)), labels, rotation=30, ha="right", color=_INK, fontsize=9)
    ax.set_yticks(range(len(cams)), labels, color=_INK, fontsize=9)
    ax.set_xlabel("to", color=_MUTED, fontsize=9)
    ax.set_ylabel("from", color=_MUTED, fontsize=9)
    vmax = max(int(mat.max()), 1)
    for i in range(len(cams)):
        for j in range(len(cams)):
            ax.text(
                j, i, str(int(mat[i, j])), ha="center", va="center", fontsize=10,
                color="white" if mat[i, j] > 0.55 * vmax else _INK,
            )  # fmt: skip
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Camera transitions", loc="left", color=_INK, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).outline.set_visible(False)
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, facecolor="white")
    return fig


# ------------------------------------------------------------------------------ floor plan


def floor_points(
    conn: sqlite3.Connection,
    camera_ids: list[str] | None = None,
    gid: int | None = None,
) -> dict[str, np.ndarray]:
    """Foot points mapped to floor coordinates by each camera's homography (cameras without
    a homography are skipped)."""
    out: dict[str, np.ndarray] = {}
    for cam in db.list_cameras(conn):
        if camera_ids is not None and cam.id not in camera_ids:
            continue
        if not cam.homography:
            continue
        p = foot_points(conn, cam.id, gid)
        if len(p):
            H = np.asarray(cam.homography, np.float64).reshape(3, 3)
            out[cam.id] = cv2.perspectiveTransform(p.reshape(-1, 1, 2), H).reshape(-1, 2)
    return out


def _need_floor(conn: sqlite3.Connection, camera_ids, gid) -> dict[str, np.ndarray]:
    fp = floor_points(conn, camera_ids, gid)
    if not fp:
        raise ValueError("no camera with a homography (and track points) for a floor plan")
    return fp


def plot_floor_heatmap(
    conn: sqlite3.Connection,
    cfg: Config,
    out_path: str | Path | None = None,
    camera_ids: list[str] | None = None,
    gid: int | None = None,
    bins: int = 80,
):
    """Floor-plan density of all cameras' foot points (needs camera homographies)."""
    allp = np.concatenate(list(_need_floor(conn, camera_ids, gid).values()))
    lo, hi = allp.min(0), allp.max(0)
    pad = 0.05 * np.maximum(hi - lo, 1e-6)
    rng = [[lo[0] - pad[0], hi[0] + pad[0]], [lo[1] - pad[1], hi[1] + pad[1]]]
    hist, xe, ye = np.histogram2d(allp[:, 0], allp[:, 1], bins=bins, range=rng)
    hist = cv2.GaussianBlur(hist.T.astype(np.float32), (0, 0), max(cfg.render.heat_sigma / 4, 0.8))
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    masked = np.ma.masked_less_equal(hist, hist.max() * 0.02)
    ax.set_facecolor("#f3f4f6")
    im = ax.imshow(masked, origin="lower", extent=(xe[0], xe[-1], ye[0], ye[-1]), cmap="inferno")
    ax.set_aspect("equal")
    ax.set_title("Floor-plan heatmap", loc="left", color=_INK, fontsize=12, fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=_MUTED, labelsize=8, length=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).outline.set_visible(False)
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, facecolor="white")
    return fig


def plot_floor_trajectories(
    conn: sqlite3.Connection,
    out_path: str | Path | None = None,
    camera_ids: list[str] | None = None,
    gid: int | None = None,
):
    """Trajectories on the floor plan, one colour per identity (grey if unassigned)."""
    cams = [c for c in db.list_cameras(conn) if c.homography]
    if camera_ids is not None:
        cams = [c for c in cams if c.id in camera_ids]
    if not cams:
        raise ValueError("no camera with a homography for a floor plan")
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    drawn = False
    for cam in cams:
        H = np.asarray(cam.homography, np.float64).reshape(3, 3)
        for t in db.get_tracklets(conn, camera_id=cam.id, gid=gid):
            p = db.get_tracklet_points(conn, t.id or 0)
            if len(p) < 2:
                continue
            feet = np.stack([(p["x1"] + p["x2"]) / 2.0, p["y2"]], axis=1)
            xy = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), H).reshape(-1, 2)
            col = hex_color(gid_color(t.gid))
            ax.plot(xy[:, 0], xy[:, 1], color=col, lw=1.8, alpha=0.9, solid_capstyle="round")
            ax.scatter(xy[-1, 0], xy[-1, 1], color=col, s=14, zorder=3)
            drawn = True
    if not drawn:
        raise ValueError("no trajectories to draw")
    ax.set_aspect("equal")
    ax.grid(color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Floor-plan trajectories", loc="left", color=_INK, fontsize=12, fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=_MUTED, labelsize=8, length=0)
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, facecolor="white")
    return fig
