"""Follow-person clip planning: options, cut plan, frame counts and layout (no video I/O).

``plan_clip`` is the single source of truth for a clip. The Clips tab preview, the render time
estimate and the renderer (:mod:`reid.viz.journey`) all read the same :class:`ClipPlan`, so what
the preview announces is what gets rendered.

Playback runs on the shared timeline. Output frame ``k`` of a segment shows shared time
``t0 + k * speed / fps``: a higher ``speed`` advances faster per frame, so it renders fewer
frames and finishes sooner. Gaps where nobody sees the person are skipped (the caption of the cut
and the plan table say how long was skipped).
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from reid import analytics
from reid.config import Config
from reid.storage import db
from reid.viz.clip_layout import MAX_WIDTH, MIN_WIDTH, Layout, make_layout

MIN_SEG_S = 1.0  # a tracklet shorter than this is shown for this long
TILE_FPS = 10.0  # thumbnails refresh at about this rate (see ``tile_every``)


@dataclass(slots=True)
class Segment:
    """A continuous stretch of the clip shown from one camera (shared-timeline seconds)."""

    camera_id: str
    t0: float
    t1: float
    tracklet_id: int
    gap_s: float | None  # time skipped since the previous segment (None for the first)


@dataclass(frozen=True)
class ClipOptions:
    """Everything the user can change about a follow clip (all optional).

    ``cameras``: restrict the clip (cuts and thumbnails) to these cameras; None means every
    camera that sees the person. ``start_s`` / ``end_s``: window on the shared timeline.
    ``speed``: playback speed multiplier. ``max_seconds``: cap on the *output* duration.
    ``max_thumbs``: side thumbnails of other cameras (0 or ``thumbs=False`` removes the strip).
    ``width`` / ``fps``: output size and frame rate (None = automatic).
    """

    cameras: tuple[str, ...] | None = None
    start_s: float | None = None
    end_s: float | None = None
    speed: float = 1.0
    max_seconds: float | None = None
    blur_others: bool = False
    show_others: bool = True
    thumbs: bool = True
    max_thumbs: int = 4
    width: int = 1280
    fps: float | None = None

    @classmethod
    def from_cfg(cls, cfg: Config, **kw: Any) -> ClipOptions:
        """Defaults taken from ``cfg.render`` (width, fps, blur), overridden by ``kw``."""
        base = {
            "width": cfg.render.journey_width,
            "fps": cfg.render.out_fps,
            "blur_others": cfg.render.blur_others,
        }
        return cls(**{**base, **kw})

    def apply_cfg(self, cfg: Config) -> Config:
        """Copy of ``cfg`` with the options that live in the render section (never mutates)."""
        out = cfg.model_copy(deep=True)
        out.render.blur_others = bool(self.blur_others or cfg.render.blur_others)
        out.render.journey_width = int(self.width)
        out.render.out_fps = self.fps
        return out

    def key(self) -> dict[str, Any]:
        """JSON-able form with every option, for render cache keys."""
        d = asdict(self)
        d["cameras"] = sorted(self.cameras) if self.cameras is not None else None
        return d

    def clean(self) -> ClipOptions:
        """Options with out-of-range values clamped (speed, width, thumbnails)."""
        return replace(
            self,
            speed=float(np.clip(self.speed, 0.1, 64.0)),
            width=int(np.clip(int(self.width), MIN_WIDTH, MAX_WIDTH)) // 2 * 2,
            max_thumbs=max(0, int(self.max_thumbs)),
            fps=None if not self.fps else float(np.clip(self.fps, 1.0, 60.0)),
            max_seconds=None if not self.max_seconds else max(float(self.max_seconds), 0.5),
        )


@dataclass(slots=True)
class PlanSegment:
    """One planned stretch: which camera, which shared-time span, how many output frames."""

    camera_id: str
    camera_name: str
    t0: float
    t1: float
    tracklet_id: int
    gap_s: float | None
    n_frames: int
    thumbs: tuple[str, ...] = ()  # cameras shown in the side strip during this segment
    live_thumbs: float = 0.0  # expected thumbnails that have footage (the rest read NO SIGNAL)

    @property
    def source_s(self) -> float:
        return self.t1 - self.t0


@dataclass(slots=True)
class ClipPlan:
    """What a clip will contain; see :func:`plan_clip`."""

    gid: int
    options: ClipOptions
    segments: list[PlanSegment] = field(default_factory=list)
    fps: float = 25.0
    src_fps: float = 25.0
    speed: float = 1.0
    layout: Layout | None = None
    out_frames: int = 0
    out_seconds: float = 0.0
    full_out_seconds: float = 0.0  # before the max-length cap
    source_seconds: float = 0.0  # shared-timeline time covered by the segments
    truncated: bool = False
    cameras: list[tuple[str, str]] = field(default_factory=list)  # (id, name) rows of the timeline
    presence: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    t_lo: float = 0.0
    t_hi: float = 0.0
    n_sightings: int = 0

    @property
    def empty(self) -> bool:
        return not self.segments

    @property
    def camera_ids(self) -> list[str]:
        """Cameras whose video is needed for the main view, in order of first use."""
        return list(dict.fromkeys(s.camera_id for s in self.segments))

    @property
    def all_camera_ids(self) -> list[str]:
        """Main and thumbnail cameras."""
        return list(dict.fromkeys(self.camera_ids + [c for s in self.segments for c in s.thumbs]))

    def frame_step(self) -> float:
        """Shared-timeline seconds between two output frames."""
        return self.speed / self.fps


def segments_from_steps(
    steps: list[analytics.JourneyStep], min_len_s: float = MIN_SEG_S
) -> list[Segment]:
    """Cut plan from a person's sightings (the logic of :func:`plan_segments`).

    At every instant the active camera is one whose tracklet covers it (the one lasting longest,
    so hand-offs between overlapping cameras happen as late as possible); when nobody sees the
    person the plan skips ahead to the next appearance. Tracklets shorter than ``min_len_s`` are
    shown for ``min_len_s``.
    """
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


def plan_segments(
    conn: sqlite3.Connection, gid: int, min_len_s: float = MIN_SEG_S
) -> list[Segment]:
    """Cut plan of the whole journey of ``gid`` (no camera filter, no window)."""
    return segments_from_steps(analytics.journey(conn, gid), min_len_s)


def _camera_spans(conn: sqlite3.Connection, cams: dict) -> dict[str, tuple[float, float]]:
    """Shared-timeline span each camera has footage for: from its start to its last detection
    (a lower bound of the video's end; the plan has no other source of the duration)."""
    last = dict(conn.execute("SELECT camera_id, MAX(end_ts) FROM tracklets GROUP BY camera_id"))
    return {
        cid: (c.start_ts, max(last.get(cid) or c.start_ts, c.start_ts)) for cid, c in cams.items()
    }


def _live_fraction(span: tuple[float, float], seg: PlanSegment) -> float:
    """Share of ``seg`` during which a camera with footage ``span`` has a frame."""
    length = max(seg.t1 - seg.t0, 1e-9)
    return max(0.0, min(span[1], seg.t1) - max(span[0], seg.t0)) / length


def _rank_thumbs(
    seg: PlanSegment, presence: dict[str, list[tuple[float, float]]], pool: list[str], n: int
) -> tuple[str, ...]:
    """Other cameras most worth showing next to ``seg``: overlapping sightings first, then the
    nearest in time."""
    mid = (seg.t0 + seg.t1) / 2

    def score(cid: str) -> tuple[float, float, int]:
        ivs = presence.get(cid, [])
        ov = sum(max(0.0, min(b, seg.t1) - max(a, seg.t0)) for a, b in ivs)
        near = min((0.0 if a <= mid <= b else min(abs(a - mid), abs(b - mid)) for a, b in ivs),
                   default=1e12)  # fmt: skip
        return (-ov, near, pool.index(cid))

    return tuple(sorted((c for c in pool if c != seg.camera_id), key=score)[:n])


def plan_clip(conn: sqlite3.Connection, gid: int, opts: ClipOptions | None = None) -> ClipPlan:
    """Plan the follow clip of ``gid``: segments, frame counts, thumbnails and layout.

    Tracklets without an identity are never part of a plan. An empty plan (``plan.empty``) means
    the person has no sighting in the selected cameras and window.
    """
    opts = (opts or ClipOptions()).clean()
    lo = -np.inf if opts.start_s is None else float(opts.start_s)
    hi = np.inf if opts.end_s is None else float(opts.end_s)
    cams = {c.id: c for c in db.list_cameras(conn)}
    allowed = None if opts.cameras is None else set(opts.cameras)
    steps = [
        s for s in analytics.journey(conn, gid) if s.camera_id in cams and (
            allowed is None or s.camera_id in allowed)
    ]  # fmt: skip
    segs = [s for s in segments_from_steps(steps) if s.t1 > lo and s.t0 < hi]
    for i, s in enumerate(segs):
        s.t0, s.t1 = max(s.t0, lo), min(s.t1, hi)
        if i == 0:
            s.gap_s = None
    in_win = [s for s in steps if s.end_ts > lo and s.start_ts < hi]
    plan = ClipPlan(gid, opts, speed=opts.speed)
    if not segs:
        return plan
    presence: dict[str, list[tuple[float, float]]] = {}
    for s in in_win:
        presence.setdefault(s.camera_id, []).append((max(s.start_ts, lo), min(s.end_ts, hi)))
    rows = [cid for cid in cams if cid in presence or (allowed and cid in allowed)]
    plan.cameras = [(cid, cams[cid].name) for cid in rows]
    plan.presence = presence
    plan.t_lo = max(min(s.start_ts for s in in_win), lo)
    plan.t_hi = min(max(s.end_ts for s in in_win), hi)
    plan.n_sightings = len(in_win)
    seg_cams = [cams[s.camera_id] for s in segs]
    plan.src_fps = max(c.fps for c in seg_cams)
    plan.fps = float(opts.fps or min(30.0, plan.src_fps))
    step = plan.frame_step()
    frames = [max(1, int((s.t1 - s.t0) * plan.fps / plan.speed)) for s in segs]
    plan.full_out_seconds = sum(frames) / plan.fps
    if opts.max_seconds is not None:  # cap the output duration
        budget = max(1, int(opts.max_seconds * plan.fps))
        kept: list[Segment] = []
        kept_frames: list[int] = []
        for s, n in zip(segs, frames, strict=True):
            if budget <= 0:
                break
            if n > budget:
                n, s.t1 = budget, s.t0 + budget * step
            budget -= n
            kept.append(s)
            kept_frames.append(n)
        plan.truncated = sum(kept_frames) < sum(frames)
        segs, frames = kept, kept_frames
    first = cams[segs[0].camera_id]
    pool = list(opts.cameras) if opts.cameras is not None else list(presence)
    pool = [c for c in pool if c in cams]
    want = min(opts.max_thumbs, max(len(pool) - 1, 0)) if opts.thumbs else 0
    plan.layout = make_layout(
        opts.width, first.height / first.width, len(plan.cameras), want, strip=opts.thumbs
    )
    n_thumbs = plan.layout.n_thumbs
    plan.segments = [
        PlanSegment(s.camera_id, cams[s.camera_id].name, s.t0, s.t1, s.tracklet_id, s.gap_s, n, ())
        for s, n in zip(segs, frames, strict=True)
    ]
    spans = _camera_spans(conn, cams)
    for ps in plan.segments:
        ps.thumbs = _rank_thumbs(ps, presence, pool, n_thumbs)
        ps.live_thumbs = sum(_live_fraction(spans[c], ps) for c in ps.thumbs)
    plan.out_frames = sum(frames)
    plan.out_seconds = plan.out_frames / plan.fps
    plan.source_seconds = sum(s.t1 - s.t0 for s in segs)
    return plan


def tile_every(fps: float) -> int:
    """Thumbnails are refreshed every this many output frames (about ``TILE_FPS`` per second)."""
    return max(1, round(fps / TILE_FPS))
