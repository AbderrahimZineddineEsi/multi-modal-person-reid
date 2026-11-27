"""Explore tab: pick a person, see their profile, journey and timeline (clips: Clips tab)."""

from __future__ import annotations

import gradio as gr

from reid import analytics
from reid.app import clips_data, views
from reid.app import components as C
from reid.app.state import AppContext
from reid.app.theme import BG, MUTED, PANEL, TEXT
from reid.viz.timeline import HAS_PLOTLY, plot_timeline, timeline_figure


def refresh_people(ctx: AppContext, current: int | None = None):
    """Picker contents: (gallery items, gids, dropdown update); keeps ``current`` if it exists."""
    with ctx.db() as conn:
        items, gids = views.gallery_items(ctx, conn)
        choices = views.person_choices(views.people_cards(ctx, conn))
    known = [v for _, v in choices]
    value = current if current in known else (known[0] if known else None)
    return items, gids, gr.update(choices=choices, value=value)


def timeline_plot(conn, gid: int):
    """Presence chart in the app's colours (plotly when installed, else matplotlib)."""
    if not HAS_PLOTLY:
        return plot_timeline(conn, gid)
    fig = timeline_figure(conn, gid)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=BG, font={"color": TEXT},
        title={"text": ""}, margin={"l": 10, "r": 10, "t": 10, "b": 40}, height=280,
    )  # fmt: skip
    fig.update_traces(marker_line_color=PANEL)
    fig.update_xaxes(gridcolor="#252a3a", zerolinecolor="#252a3a", color=MUTED)
    return fig


def show(ctx: AppContext, gid: int | None):
    """Everything about one person: (profile, journey, timeline)."""
    if gid is None:
        hint = C.empty_state("Pick a person", "Choose someone from the list on the left.")
        return hint, "", None
    gid = int(gid)
    with ctx.db() as conn:
        card = next((c for c in views.people_cards(ctx, conn) if c.gid == gid), None)
        if card is None or not analytics.journey(conn, gid):
            raise gr.Error(f"Person {gid} no longer exists. Refresh the list.")
        steps = views.journey_steps(ctx, conn, gid)
        return C.identity_card(card, big=True), C.journey_html(gid, steps), timeline_plot(conn, gid)


def build(ctx: AppContext) -> gr.Tab:
    with gr.Tab("Explore", id="explore") as tab:
        gids = gr.State([])
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=280):
                person = gr.Dropdown([], label="Person", info="Or click a face below.")
                picker = gr.Gallery(
                    label=None, columns=2, height=520, object_fit="cover", allow_preview=False,
                    elem_classes=["rd-picker"],
                    show_label=False, buttons=[],
                )  # fmt: skip
            with gr.Column(scale=3):
                profile = gr.HTML()
                journey = gr.HTML()
                gr.Markdown("### Presence over time")
                timeline = gr.Plot(show_label=False)
                gr.Markdown("### Follow clip")
                to_clips = gr.Button("Open in Clips", variant="primary")
    ctx.ui["explore_person"] = person

    def pick(ids: list[int], evt: gr.SelectData) -> int:
        return ids[evt.index]

    def reload():
        """Picker and the first person's view (ids repeat across environments, so show it)."""
        items, ids, update = refresh_people(ctx)
        return items, ids, update, *show(ctx, update["value"])

    shown = [profile, journey, timeline]
    gr.on(
        [tab.select, ctx.ui["demo"].load], lambda: refresh_people(ctx), None, [picker, gids, person]
    )
    ctx.ui["env_version"].change(reload, None, [picker, gids, person, *shown])
    picker.select(pick, gids, person)
    person.change(lambda g: show(ctx, g), person, shown)
    to_clips.click(
        lambda g: (clips_data.make_prefill(g), gr.Tabs(selected="clips")),
        person,
        [ctx.ui["clips_prefill"], ctx.ui["tabs"]],
    )
    return tab
