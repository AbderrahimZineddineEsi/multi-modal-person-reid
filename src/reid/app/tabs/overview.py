"""Landing dashboard: headline numbers, everyone seen so far, cameras and the latest run."""

from __future__ import annotations

import gradio as gr

from reid import analytics
from reid.app import components as C
from reid.app import envbar, views
from reid.app.state import AppContext


def render(ctx: AppContext) -> tuple[str, str, str]:
    """HTML for (tiles, people grid, cameras and run info)."""
    with ctx.db() as conn:
        run_id = ctx.latest_run_id(conn)
        if run_id is None:
            empty = C.empty_state("Nothing processed yet", "Open the Process tab to start a run.")
            return empty, "", ""
        summary = analytics.run_summary(conn, run_id)
        stats = ctx.load_run_stats(run_id)
        cards = views.people_cards(ctx, conn)
        cams = views.camera_rows(conn)
    side = C.cameras_html(cams) + _run_note(run_id, stats)
    return views.summary_tiles(summary, stats), C.person_grid(cards), side


def _run_note(run_id: int, stats: dict | None) -> str:
    if not stats:
        return f'<p class="rd-hint">Latest run: #{run_id}. Timings exist for runs started here.</p>'
    return (
        f'<p class="rd-hint">Latest run: #{run_id}, {stats["frames_processed"]} frames in '
        f"{C.fmt_dur(stats['wall_s'])}.</p>"
    )


def build(ctx: AppContext) -> gr.Tab:
    with gr.Tab("Overview", id="overview") as tab:
        kpis = gr.HTML()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                gr.Markdown("### People")
                people = gr.HTML()
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Cameras")
                side = gr.HTML()
    gr.on(
        envbar.refresh_on(ctx, tab.select, ctx.ui["demo"].load),
        lambda: render(ctx),
        None,
        [kpis, people, side],
    )
    return tab
