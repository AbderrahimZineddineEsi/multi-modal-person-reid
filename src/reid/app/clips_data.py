"""Logic behind the Clips tab: what a person's clip can cover, option parsing, cached render.

No Gradio here. The tab (:mod:`reid.app.tabs.clips`) passes control values in and shows what
comes back; the plan, the estimate and the renderer live in :mod:`reid.viz`.
"""

from __future__ import annotations

import math
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from reid import analytics
from reid.app.state import AppContext
from reid.storage import db
from reid.viz.clip_plan import ClipOptions, ClipPlan, plan_clip
from reid.viz.journey import render_clip

SPEEDS: list[tuple[str, float]] = [(f"{s:g}x", s) for s in (0.5, 1, 2, 4, 8, 16)]
WIDTHS: list[int] = [640, 960, 1280, 1600, 1920]
FPS_CHOICES: list[tuple[str, int]] = [("Auto", 0), ("12", 12), ("15", 15), ("24", 24), ("30", 30)]
DEFAULT_MAX_S = 120  # app default cap on the clip length (0 = no limit)


@dataclass(frozen=True)
class Scope:
    """What the controls offer for one person."""

    cameras: list[tuple[str, str]]  # (name, id) of every camera of the environment
    default: list[str]  # cameras that see the person
    lo: float  # first sighting, floored (shared-timeline seconds)
    hi: float  # last sighting, ceiled


def person_scope(conn: sqlite3.Connection, gid: int | None) -> Scope:
    """Camera choices and time bounds of ``gid`` (the footage range when nobody is picked)."""
    cams = db.list_cameras(conn)
    steps = analytics.journey(conn, gid) if gid is not None else []
    if steps:
        lo, hi = min(s.start_ts for s in steps), max(s.end_ts for s in steps)
    else:
        lo = min((c.start_ts for c in cams), default=0.0)
        hi = max((c.start_ts for c in cams), default=0.0) + 1
    seen = {s.camera_id for s in steps}
    lo, hi = float(math.floor(lo)), float(math.ceil(hi))
    return Scope(
        [(c.name, c.id) for c in cams], [c.id for c in cams if c.id in seen], lo, max(hi, lo + 1)
    )


def people_choices(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    """Dropdown entries ``(label, gid)`` of the identities that have sightings."""
    with_tracklets = {t.gid for t in db.get_tracklets(conn) if t.gid is not None}
    return [
        (f"{i.label} (#{i.gid})" if i.label else f"Person {i.gid}", i.gid)
        for i in db.list_identities(conn)
        if i.gid in with_tracklets
    ]


def make_prefill(
    gid: int | None,
    cameras: Sequence[str] | None = None,
    start: float | None = None,
    end: float | None = None,
) -> dict[str, Any] | None:
    """Value for ``ctx.ui["clips_prefill"]``: open the Clips tab on this person and scope.

    ``cameras`` empty or None means the cameras that see the person; ``start`` / ``end`` are
    clamped to the person's presence by the tab. The timestamp makes every click a new value.
    """
    if gid is None:
        return None
    return {
        "gid": int(gid),
        "cameras": list(cameras) if cameras else None,
        "start": start,
        "end": end,
        "nonce": time.time(),
    }


def to_options(
    cameras: Sequence[str] | None,
    t_from: float | None,
    t_to: float | None,
    speed: float,
    max_s: float,
    blur: bool,
    others: bool,
    thumbs: bool,
    n_thumbs: int,
    width: int,
    fps: float,
) -> ClipOptions:
    """Control values -> :class:`ClipOptions` (0 means "no limit" / "auto")."""
    return ClipOptions(
        cameras=tuple(cameras or ()),
        start_s=t_from,
        end_s=t_to,
        speed=float(speed or 1.0),
        max_seconds=float(max_s) if max_s else None,
        blur_others=bool(blur),
        show_others=bool(others),
        thumbs=bool(thumbs),
        max_thumbs=int(n_thumbs or 0),
        width=int(width),
        fps=float(fps) if fps else None,
    )


def effective_options(ctx: AppContext, opts: ClipOptions) -> ClipOptions:
    """``opts``, with the privacy blur forced on in showcase mode (the plan and estimate know)."""
    return replace(opts, blur_others=True) if ctx.showcase is not None else opts


def preview_plan(ctx: AppContext, gid: int, opts: ClipOptions) -> tuple[ClipPlan, float]:
    """The plan and its estimated render time (seconds), the same plan the renderer uses."""
    from reid.viz.clip_cost import estimate

    opts = effective_options(ctx, opts)
    with ctx.db() as conn:
        plan = plan_clip(conn, gid, opts)
    return plan, estimate(plan, ctx.clip_model())


def render_cached(
    ctx: AppContext, gid: int, opts: ClipOptions, progress: Callable[[int, int], None]
) -> tuple[Path, ClipPlan, float | None]:
    """Render the clip (or reuse the cached file). Returns ``(path, plan, seconds or None)``.

    Every option is in the cache key, so different settings never share a file. A fresh render
    records its measured speed for later estimates; ``seconds`` is None on a cache hit.
    """
    measured: list[float] = []
    opts = effective_options(ctx, opts)
    with ctx.db() as conn:
        cfg = ctx.render_cfg(conn, gid)  # showcase: a blurring copy, never the shared config
        plan = plan_clip(conn, gid, opts)
        if plan.empty:
            raise ValueError("The person has no sighting in the selected cameras and window.")

        def make(path: Path) -> None:
            res = render_clip(conn, cfg, gid, path, opts, progress, plan)
            measured.append(res.seconds)
            ctx.record_clip(plan, res.seconds, res.frames)

        params = {"opts": opts.key(), "render": cfg.render.model_dump()}
        path = ctx.cached(conn, "clip", params, ".mp4", make, tag=f"gid{int(gid)}")
    return path, plan, (measured[0] if measured else None)
