"""Gantt-style presence charts (matplotlib PNG, optional plotly figure for the UI)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

from reid.storage import db  # noqa: E402
from reid.viz.colors import PALETTE, gid_color, hex_color  # noqa: E402

try:  # optional: interactive figure for the Gradio app
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:  # pragma: no cover - depends on the environment
    go = None  # type: ignore[assignment]
    HAS_PLOTLY = False

_INK, _MUTED, _GRID = "#1f2430", "#6b7280", "#e5e7eb"


def _rows(conn: sqlite3.Connection, gid: int | None):
    """``(row_labels, bars, camera_names)``; bars are ``(row, t0, t1, gid, camera_id, tid)``."""
    cams = {c.id: c.name for c in db.list_cameras(conn)}
    tl = db.get_tracklets(conn, gid=gid)
    if gid is not None:
        labels = list(cams.values())
        idx = {cid: i for i, cid in enumerate(cams)}
        bars = [(idx[t.camera_id], t.start_ts, t.end_ts, t.gid, t.camera_id, t.id) for t in tl]
    else:
        gids = sorted({t.gid for t in tl if t.gid is not None})
        labels = [f"ID {g}" for g in gids]
        idx = {g: i for i, g in enumerate(gids)}
        if any(t.gid is None for t in tl):
            labels.append("unassigned")
            idx[None] = len(gids)  # type: ignore[index]
        bars = [(idx[t.gid], t.start_ts, t.end_ts, t.gid, t.camera_id, t.id) for t in tl]
    return labels, bars, cams


def _cam_colors(cams: dict[str, str]) -> dict[str, tuple[int, int, int]]:
    return {cid: PALETTE[(5 + 2 * i) % len(PALETTE)] for i, cid in enumerate(cams)}


def plot_timeline(
    conn: sqlite3.Connection, gid: int | None = None, out_path: str | Path | None = None
):
    """Presence chart: rows are cameras for one ``gid``, identities (coloured by camera) for all.

    Returns the matplotlib figure (saved to ``out_path`` as PNG when given).
    """
    labels, bars, cams = _rows(conn, gid)
    n = max(len(labels), 1)
    fig, ax = plt.subplots(figsize=(10, min(1.0 + 0.42 * n, 14)), dpi=160)
    ccol = _cam_colors(cams)
    t_all = [b[1] for b in bars] + [b[2] for b in bars]
    t_lo = min(t_all, default=0.0)
    for row, a, b, g, cid, _tid in bars:
        rgb = gid_color(g) if gid is not None else ccol[cid]
        ax.barh(
            row, max(b - a, 0.15), left=a - t_lo, height=0.62,
            color=hex_color(rgb), edgecolor="white", linewidth=0.6,
        )  # fmt: skip
    ax.set_yticks(range(len(labels)), labels, color=_INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("time (s)", color=_MUTED, fontsize=9)
    ax.grid(axis="x", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=8, length=0)
    ax.set_title(
        f"Presence of ID {gid} per camera" if gid is not None else "Presence per identity",
        loc="left", color=_INK, fontsize=12, fontweight="bold",
    )  # fmt: skip
    if gid is None and cams:
        from matplotlib.patches import Patch

        ax.legend(
            handles=[Patch(color=hex_color(ccol[c]), label=nm) for c, nm in cams.items()],
            loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(cams), frameon=False,
            fontsize=8, labelcolor=_INK,
        )  # fmt: skip
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, facecolor="white")
    return fig


def timeline_figure(conn: sqlite3.Connection, gid: int | None = None) -> Any:
    """Interactive plotly version of :func:`plot_timeline` (``ImportError`` without plotly)."""
    if not HAS_PLOTLY:
        raise ImportError("plotly is not installed: use plot_timeline (matplotlib) instead")
    labels, bars, cams = _rows(conn, gid)
    ccol = _cam_colors(cams)
    fig = go.Figure()
    for row, a, b, g, cid, tid in bars:
        rgb = gid_color(g) if gid is not None else ccol[cid]
        fig.add_trace(
            go.Bar(
                y=[labels[row]], x=[max(b - a, 0.15)], base=[a], orientation="h",
                marker={"color": hex_color(rgb), "line": {"color": "white", "width": 1}},
                name=cams[cid], showlegend=False,
                hovertemplate=(
                    f"{cams[cid]} (tracklet {tid})<br>{a:.1f}s to {b:.1f}s<extra></extra>"
                ),
            )
        )  # fmt: skip
    fig.update_layout(
        barmode="overlay", template="plotly_white", margin={"l": 70, "r": 20, "t": 40, "b": 40},
        title="Presence" + (f" of ID {gid}" if gid is not None else " per identity"),
        xaxis_title="time (s)", yaxis={"autorange": "reversed"},
    )  # fmt: skip
    return fig
