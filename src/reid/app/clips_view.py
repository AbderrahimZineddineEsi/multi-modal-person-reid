"""Preview of a planned clip: summary tiles, notices, plan table and timetable chart."""

from __future__ import annotations

from html import escape
from typing import Any

from reid.app import components as C
from reid.app import videos
from reid.app.state import AppContext
from reid.app.theme import BG, MUTED, PANEL, TEXT
from reid.viz.clip_cost import fmt_seconds
from reid.viz.clip_plan import ClipPlan
from reid.viz.colors import gid_color, hex_color
from reid.viz.timeline import HAS_PLOTLY, go

LONG_RENDER_S = 600  # warn above this estimated render time


def tiles_html(plan: ClipPlan, est_s: float, calibrated: bool) -> str:
    """KPI tiles: clip length, source time covered, cameras, output size, estimated render time."""
    if plan.empty or plan.layout is None:
        return ""
    lay = plan.layout
    n_cams = len(plan.camera_ids)
    n_thumbs = max((len(s.thumbs) for s in plan.segments), default=0)
    return C.tiles(
        [
            C.metric_tile("Clip length", C.fmt_dur(plan.out_seconds),
                          f"{plan.out_frames:,} frames at {plan.fps:g} fps"),
            C.metric_tile("Footage covered", C.fmt_dur(plan.source_seconds),
                          f"at {plan.speed:g}x, {C.plural(len(plan.segments), 'cut')}"),
            C.metric_tile("Cameras", str(n_cams),
                          f"{C.plural(n_thumbs, 'thumbnail')} on the side" if n_thumbs
                          else "no side thumbnails"),
            C.metric_tile("Output", f"{lay.width}x{lay.height}", "pixels"),
            C.metric_tile("Render time", "~" + fmt_seconds(est_s),
                          "calibrated to this machine" if calibrated else "typical PC estimate",
                          key=True),
        ]
    )  # fmt: skip


def notices_html(ctx: AppContext, plan: ClipPlan, est_s: float) -> str:
    """Warnings: nothing to render, clip capped, render would take long, videos missing."""
    if plan.empty:
        return C.notice(
            "Nothing to render",
            "The person has no sighting in the selected cameras and time window.",
        )
    out = []
    if plan.truncated:
        out.append(
            C.notice(
                f"Clip capped at {C.fmt_dur(plan.out_seconds)}",
                f"The whole journey would play for {C.fmt_dur(plan.full_out_seconds)}. Raise Max "
                "length or the speed to cover it.",
            )
        )
    if est_s > LONG_RENDER_S:
        out.append(
            C.notice(
                f"Rendering will take about {fmt_seconds(est_s)}",
                "Lower Max length, pick fewer cameras, raise the speed or use a smaller width.",
            )
        )
    out.append(videos.availability_note(ctx, plan.camera_ids))
    return "".join(out)


def plan_table_html(plan: ClipPlan) -> str:
    """Segments as a table: camera, shared-timeline span, source and output length, gap."""
    if plan.empty:
        return ""
    rows = []
    for i, s in enumerate(plan.segments, 1):
        gap = "first" if s.gap_s is None else (f"+{C.fmt_dur(s.gap_s)} skipped" if s.gap_s else "-")
        rows.append(
            f"<tr><td>{i}</td><td>{escape(s.camera_name)}</td><td>{C.fmt_clock(s.t0)}</td>"
            f"<td>{C.fmt_clock(s.t1)}</td><td>{C.fmt_dur(s.source_s)}</td>"
            f"<td>{C.fmt_dur(s.n_frames / plan.fps)}</td><td>{gap}</td></tr>"
        )
    head = "".join(
        f"<th>{h}</th>" for h in ("#", "Camera", "From", "To", "Footage", "In clip", "Gap")
    )
    return (
        '<div class="rd-scroll"><table class="rd-plan"><thead><tr>'
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def timetable_figure(plan: ClipPlan) -> Any:
    """Presence (faint) and planned segments (solid) per camera on the shared timeline."""
    if not HAS_PLOTLY or plan.empty:
        return None
    color = hex_color(gid_color(plan.gid))
    names = [n for _, n in plan.cameras]
    name_of = dict(plan.cameras)
    fig = go.Figure()
    for cid, ivs in plan.presence.items():
        for a, b in ivs:
            fig.add_trace(go.Bar(
                y=[name_of[cid]], x=[max(b - a, 0.15)], base=[a], orientation="h", opacity=0.3,
                marker={"color": color}, showlegend=False,
                hovertemplate=f"seen {a:.1f}s to {b:.1f}s<extra>{escape(name_of[cid])}</extra>",
            ))  # fmt: skip
    for i, s in enumerate(plan.segments, 1):
        gap = "" if s.gap_s is None else f"<br>after {C.fmt_dur(s.gap_s)} gap"
        fig.add_trace(go.Bar(
            y=[s.camera_name], x=[max(s.source_s, 0.15)], base=[s.t0], orientation="h",
            marker={"color": color, "line": {"color": PANEL, "width": 1}}, showlegend=False,
            text=[str(i)] if s.source_s > 0 else None, textposition="inside",
            insidetextanchor="middle", textfont={"color": "#101314"},
            hovertemplate=(f"cut {i}: {s.t0:.1f}s to {s.t1:.1f}s "
                           f"({C.fmt_dur(s.n_frames / plan.fps)} in the clip)"
                           f"{gap}<extra>{escape(s.camera_name)}</extra>"),
        ))  # fmt: skip
    fig.update_layout(
        barmode="overlay", template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=BG,
        font={"color": TEXT}, margin={"l": 10, "r": 10, "t": 10, "b": 40},
        height=int(min(90 + 28 * len(names), 520)), xaxis_title="time (s)",
        yaxis={"autorange": "reversed", "categoryorder": "array", "categoryarray": names},
    )  # fmt: skip
    fig.update_xaxes(gridcolor="#252a3a", zerolinecolor="#252a3a", color=MUTED)
    return fig


def result_note(plan: ClipPlan, est_s: float, seconds: float | None) -> str:
    """Line under the video: how long the render took against the estimate."""
    if seconds is None:
        return C.notice("Reused a clip rendered earlier with these settings")
    return C.notice(
        f"Rendered in {fmt_seconds(seconds)}",
        f"{plan.out_frames:,} frames ({seconds / max(plan.out_frames, 1) * 1000:.0f} ms each); "
        f"the estimate was {fmt_seconds(est_s)}.",
    )
