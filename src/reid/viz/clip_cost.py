"""Render time estimate for a follow clip, calibrated to the machine it runs on.

The model is per output frame::

    main   = decode * (source frames advanced) + fixed + draw * (main view pixels)
    thumbs = live * (decode * (source frames advanced) + (fixed + draw * tile pixels) / every)
    encode = encode * (frame pixels)

A thumbnail whose camera has no footage at that instant (cameras that do not overlap in time) is
drawn as NO SIGNAL and costs almost nothing, so only the thumbnails with footage (``live``) count.

Source frames advanced per output frame is ``speed * source_fps / fps``: playing at 4x decodes
four times as many source frames per output frame, so the speed-up of a fast clip is smaller
than the frame count alone suggests. The default constants were measured on the GTX 1050 test
machine (CPU 4 cores, 8 threads; see ``docs/dev/changes_log.md``). Every real render records
``actual / modelled`` in ``<data_dir>/cache/clip_speed.json`` (:func:`record`), and
:func:`estimate` multiplies by that factor, so a faster or slower machine converges after a
render or two.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from reid.viz.clip_plan import ClipPlan, tile_every

log = logging.getLogger(__name__)

SEEK_FRAMES = 120  # FrameSource seeks instead of decoding through a longer jump


@dataclass(frozen=True)
class CostModel:
    """Seconds per unit; the defaults are measured on the reference machine."""

    decode_s: float = 0.0031  # per decoded 1080p source frame (grab)
    view_fixed_s: float = 0.0080  # per drawn view: retrieve the frame and downscale it
    draw_s_per_kpx: float = 0.000105  # resize + spotlight + boxes, per 1000 pixels of the view
    blur_mult: float = 1.8  # privacy blur multiplies the per-pixel draw cost
    encode_s_per_kpx: float = 0.0000025  # compose + pipe to ffmpeg, per 1000 frame pixels
    seek_s: float = 0.22  # one long or backwards seek in a video
    setup_s_per_camera: float = 0.5  # open the video, load the track points
    factor: float = 1.0  # machine calibration (actual / modelled)


def modelled_seconds(plan: ClipPlan, model: CostModel | None = None) -> float:
    """Model time for ``plan`` without the calibration factor."""
    m = model or CostModel()
    lay = plan.layout
    if plan.empty or lay is None:
        return 0.0
    adv = plan.speed * plan.src_fps / plan.fps  # source frames advanced per output frame
    every = tile_every(plan.fps)
    draw = m.draw_s_per_kpx * (m.blur_mult if plan.options.blur_others else 1.0)
    main = m.decode_s * adv + m.view_fixed_s + draw * lay.main_w * lay.main_h / 1000
    refresh = m.view_fixed_s + draw * lay.tile_w * lay.tile_h / 1000  # one thumbnail redraw
    encode = m.encode_s_per_kpx * lay.width * lay.height / 1000
    skip = every * adv  # source frames a thumbnail camera moves between two refreshes
    thumb_dec = m.decode_s * adv if skip <= SEEK_FRAMES else m.seek_s / every
    total = m.setup_s_per_camera * len(plan.all_camera_ids)
    for seg in plan.segments:
        total += seg.n_frames * (main + encode + seg.live_thumbs * (thumb_dec + refresh / every))
    return total


def estimate(plan: ClipPlan, model: CostModel | None = None) -> float:
    """Expected render time in seconds, scaled by the calibration of this machine."""
    m = model or CostModel()
    return modelled_seconds(plan, m) * m.factor


def _path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "cache" / "clip_speed.json"


def load_model(data_dir: str | Path) -> CostModel:
    """Default constants with the calibration factor stored for ``data_dir`` (1.0 if none)."""
    try:
        factor = float(json.loads(_path(data_dir).read_text())["factor"])
    except (OSError, ValueError, KeyError, TypeError):
        return CostModel()
    return CostModel(factor=min(max(factor, 0.1), 10.0))


def record(data_dir: str | Path, plan: ClipPlan, seconds: float, frames: int) -> dict:
    """Store the measured speed of a finished render and update the calibration factor.

    ``seconds_per_frame`` is the raw measurement; ``factor`` is the running ratio of measured to
    modelled time (each render counts half), which :func:`load_model` applies to later estimates.
    """
    modelled = modelled_seconds(plan)
    if frames <= 0 or seconds <= 0 or modelled <= 0:
        return {}
    ratio = min(max(seconds / modelled, 0.1), 10.0)
    old = load_model(data_dir)
    seen = _path(data_dir).exists()
    factor = 0.5 * old.factor + 0.5 * ratio if seen else ratio
    entry = {
        "factor": round(factor, 4),
        "seconds_per_frame": round(seconds / frames, 5),
        "last_frames": frames,
        "last_seconds": round(seconds, 2),
        "last_modelled_seconds": round(modelled, 2),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    p = _path(data_dir)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(entry, indent=2))
    except OSError as exc:  # a read-only environment still renders
        log.warning("could not store the clip speed calibration: %s", exc)
    return entry


def fmt_seconds(seconds: float) -> str:
    """``42 s`` / ``3m 05s`` / ``1h 12m`` for the preview."""
    s = int(round(seconds))
    if s < 90:
        return f"{max(s, 1)} s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {s % 3600 // 60:02d}m"
