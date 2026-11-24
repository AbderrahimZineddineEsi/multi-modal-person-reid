"""HTML builders for cards, tiles, badges and journeys (pure functions, no Gradio, no I/O)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from html import escape

import numpy as np

from reid.viz.colors import gid_color, hex_color

STATUS_TONE = {"MATCH": "ok", "AMBIGUOUS": "warn", "UNKNOWN": "dim", "NO_FACE": "dim"}
STATUS_TEXT = {
    "MATCH": "Match",
    "AMBIGUOUS": "Ambiguous",
    "UNKNOWN": "Unknown person",
    "NO_FACE": "No face found",
}


@dataclass(slots=True)
class PersonCard:
    """What an identity card shows (built from the database by ``app.views``)."""

    gid: int
    label: str | None = None
    n_tracklets: int = 0
    n_cameras: int = 0
    duration_s: float = 0.0
    has_face: bool = False
    thumb: str | None = None  # data URI

    @property
    def name(self) -> str:
        return self.label or f"Person {self.gid}"


def fmt_dur(seconds: float) -> str:
    """``4.2 s`` below a minute, ``2m 05s`` above."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m {s:02d}s"


def fmt_clock(ts: float) -> str:
    """Timeline position as ``m:ss.s``."""
    m, s = divmod(ts, 60)
    return f"{int(m)}:{s:04.1f}"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def badge(text: str, tone: str = "dim") -> str:
    """Small status pill; ``tone`` is ok | warn | bad | dim | accent."""
    return f'<span class="rd-badge rd-{tone}">{escape(text)}</span>'


def status_badge(status: str) -> str:
    """Badge for a ``SearchStatus`` value (MATCH / AMBIGUOUS / UNKNOWN / NO_FACE)."""
    return badge(STATUS_TEXT.get(status, status), STATUS_TONE.get(status, "dim"))


def metric_tile(
    label: str, value: str, hint: str = "", pct: float | None = None, key: bool = False
) -> str:
    """One KPI tile; ``pct`` (0-100) adds a thin meter under the value, ``key`` marks the
    numbers the page is about (GPU memory) with a stronger frame."""
    off = " rd-off" if pct is None else ""  # tiles without a meter keep its height
    width = max(0.0, min(pct or 0.0, 100.0))
    meter = f'<div class="rd-meter{off}"><i style="width:{width:.0f}%"></i></div>'
    hint_html = f'<div class="rd-tile-hint">{escape(hint)}</div>' if hint else ""
    cls = "rd-tile rd-key" if key else "rd-tile"
    return (
        f'<div class="{cls}"><div class="rd-tile-value">{escape(value)}</div>{meter}'
        f'<div class="rd-tile-label">{escape(label)}</div>{hint_html}</div>'
    )


def tiles(items: Iterable[str]) -> str:
    return f'<div class="rd-tiles">{"".join(items)}</div>'


# viewfinder: four corner brackets around a dot, the app's motif for "nothing in frame yet"
_FRAME_ICON = (
    '<svg class="rd-empty-icon" viewBox="0 0 32 32" width="28" height="28" fill="none" '
    'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true">'
    '<path d="M4 11V6a2 2 0 0 1 2-2h5M21 4h5a2 2 0 0 1 2 2v5M28 21v5a2 2 0 0 1-2 2h-5'
    'M11 28H6a2 2 0 0 1-2-2v-5"/><circle cx="16" cy="16" r="2.2" fill="currentColor" '
    'stroke="none"/></svg>'
)


def empty_state(title: str, hint: str = "") -> str:
    sub = f"<p>{escape(hint)}</p>" if hint else ""
    return f'<div class="rd-empty">{_FRAME_ICON}<strong>{escape(title)}</strong>{sub}</div>'


MISSING_VIDEO_TITLE = "Source video not available on this machine"
MISSING_VIDEO_HINT = "Clips need the videos; search and browsing still work."


def notice(title: str, hint: str = "", tone: str = "warn") -> str:
    """Inline message bar for something that cannot be done right now (not an error)."""
    sub = f"<p>{escape(hint)}</p>" if hint else ""
    return f'<div class="rd-notice rd-{tone}"><strong>{escape(title)}</strong>{sub}</div>'


def missing_video_notice() -> str:
    """Shown instead of a clip when the camera videos are not on this machine."""
    return notice(MISSING_VIDEO_TITLE, MISSING_VIDEO_HINT)


def _avatar(p: PersonCard) -> str:
    if p.thumb:
        return f'<img src="{p.thumb}" alt="{escape(p.name)}" loading="lazy">'
    return f'<span class="rd-initial">{p.gid}</span>'


def identity_card(p: PersonCard, big: bool = False) -> str:
    """Card with the person's colour rail, best crop, sightings and badges."""
    color = hex_color(gid_color(p.gid))
    face = badge("face confirmed", "ok") if p.has_face else badge("body only", "dim")
    meta = (
        f"{plural(p.n_tracklets, 'sighting')} in {plural(p.n_cameras, 'camera')}"
        f" &middot; on screen {fmt_dur(p.duration_s)}"
    )
    size = " rd-big" if big else ""
    return (
        f'<div class="rd-card rd-person{size}" style="--pc:{color}">'
        f'<div class="rd-avatar">{_avatar(p)}</div>'
        f'<div class="rd-person-body"><div class="rd-person-name">{escape(p.name)}'
        f'<span class="rd-gid">#{p.gid}</span></div>'
        f'<div class="rd-person-meta">{meta}</div><div class="rd-row">{face}</div></div></div>'
    )


def person_grid(people: Sequence[PersonCard]) -> str:
    if not people:
        return empty_state("No people yet", "Process some footage in the Process tab.")
    cells = "".join(identity_card(p) for p in people)
    return f'<div class="rd-grid">{cells}</div>'


@dataclass(slots=True)
class StepView:
    """One journey row: a tracklet seen by one camera."""

    camera: str
    start_ts: float
    end_ts: float
    gap_s: float | None
    source: str
    thumb: str | None = None


def journey_html(gid: int, steps: Sequence[StepView]) -> str:
    """Vertical journey rail in the person's colour: camera, time span, gap to the last sighting."""
    if not steps:
        return empty_state("No sightings", "This identity has no tracklets.")
    color = hex_color(gid_color(gid))
    rows = []
    for s in steps:
        if s.gap_s is None:
            gap = badge("first seen", "accent")
        elif s.gap_s < 0:
            gap = badge(f"overlaps by {fmt_dur(-s.gap_s)}", "warn")
        else:
            gap = badge(f"after {fmt_dur(s.gap_s)}", "dim")
        img = f'<img src="{s.thumb}" alt="">' if s.thumb else ""
        rows.append(
            f'<li><span class="rd-dot"></span><div class="rd-step-thumb">{img}</div>'
            f'<div class="rd-step-body"><div class="rd-step-cam">{escape(s.camera)}</div>'
            f'<div class="rd-step-time">{fmt_clock(s.start_ts)} to {fmt_clock(s.end_ts)}'
            f" &middot; {fmt_dur(s.end_ts - s.start_ts)}</div></div>"
            f'<div class="rd-row">{gap}{badge(s.source or "unassigned", "dim")}</div></li>'
        )
    return f'<ol class="rd-journey" style="--pc:{color}">{"".join(rows)}</ol>'


def candidates_html(
    cands: Sequence[tuple[int, float]], names: dict[int, str], best: int | None
) -> str:
    """Ranked identities with a similarity bar (cosine, 0 to 1)."""
    rows = []
    for gid, score in cands:
        pct = max(0.0, min(score, 1.0)) * 100
        color = hex_color(gid_color(gid))
        cls = " rd-best" if gid == best else ""
        name = escape(names.get(gid, str(gid)))
        rows.append(
            f'<div class="rd-cand{cls}" style="--pc:{color}"><span>{name}</span>'
            f'<div class="rd-meter"><i style="width:{pct:.0f}%"></i></div><b>{score:.2f}</b></div>'
        )
    return f'<div class="rd-cands">{"".join(rows)}</div>' if rows else ""


def transitions_html(cams: Sequence[str], mat: np.ndarray) -> str:
    """Camera-to-camera counts as a heat table (rows = from, columns = to)."""
    if len(cams) == 0 or not mat.size or mat.max() == 0:
        return empty_state("No transitions yet", "People need to be seen by two cameras in turn.")
    top = float(mat.max())
    head = "".join(f"<th>{escape(c)}</th>" for c in cams)
    body = []
    for i, c in enumerate(cams):
        cells = "".join(
            f'<td style="--a:{mat[i, j] / top:.2f}">{int(mat[i, j]) or ""}</td>'
            for j in range(len(cams))
        )
        body.append(f"<tr><th>{escape(c)}</th>{cells}</tr>")
    return (
        '<table class="rd-matrix"><thead><tr><th>from / to</th>'
        f"{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


@dataclass(slots=True)
class CameraRow:
    name: str
    size: str  # "960x756 @ 24 fps"
    n_people: int
    is_face_cam: bool = False


def cameras_html(rows: Sequence[CameraRow]) -> str:
    """Compact camera list with the number of people each camera saw."""
    if not rows:
        return empty_state("No cameras")
    items = "".join(
        f"<li><div><b>{escape(r.name)}</b><span>{escape(r.size)}</span></div>"
        f'<div class="rd-row">{badge("face cam", "accent") if r.is_face_cam else ""}'
        f"{badge(f'{r.n_people} ' + ('person' if r.n_people == 1 else 'people'))}</div></li>"
        for r in rows
    )
    return f'<ul class="rd-cams">{items}</ul>'


def page_header(title: str, subtitle: str, pills: Sequence[str] = ()) -> str:
    """Top bar: product name, one-line description and status pills."""
    return (
        f'<header class="rd-header">{_logo()}<div class="rd-brand"><h1>{escape(title)}</h1>'
        f'<p>{escape(subtitle)}</p></div><div class="rd-row">{"".join(pills)}</div></header>'
    )


def _logo() -> str:
    """One person's path across three cameras: three dots in identity colours on one line."""
    dots = "".join(
        f'<circle cx="{x}" cy="{y}" r="4" fill="{hex_color(gid_color(g))}"/>'
        for g, x, y in ((1, 6, 22), (2, 20, 8), (3, 34, 18))
    )
    return (
        '<svg class="rd-logo" viewBox="0 0 40 30" width="40" height="30" aria-hidden="true">'
        '<path d="M6 22 20 8 34 18" fill="none" stroke="#929a9e" stroke-width="1.4" '
        f'stroke-dasharray="2 3"/>{dots}</svg>'
    )
