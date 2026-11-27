"""Find tab: filter the annotated appearances by what people look like (layout and wiring only).

Results on the left, the filter panel on the right, a detail drawer that slides in over the panel.
Logic lives in :mod:`reid.app.find_data` and :mod:`reid.find`, HTML in :mod:`reid.app.find_views`.
Every filter change re-runs the query on the cached attribute table (a few milliseconds).
"""

from __future__ import annotations

import math
from typing import Any

import gradio as gr

from reid.app import clips_data, envbar, find_data, find_states, find_views, views
from reid.app.find_data import PAGE, VIEWS, Selection
from reid.app.state import AppContext
from reid.attributes import ui_labels
from reid.attributes.annotate import EnvStatus
from reid.find import PersonHit, person_attributes
from reid.storage import db

# accordion title -> filter sections inside it (``ui_labels.SECTIONS`` ids)
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Person", ("person",)),
    ("Upper body", ("upper", "upper_colour")),
    ("Lower body", ("lower", "lower_colour")),
    ("Head", ("head", "hair_colour")),
    ("Carrying", ("carrying",)),
)
SUBTITLE = {"upper_colour": "Colour", "lower_colour": "Colour", "hair_colour": "Hair colour"}
STRICT_INFO = (
    "On: only people whose predicted attributes pass every filter. "
    "Off: everyone is ranked by how close they are, so a rare mix still shows candidates."
)
SECTIONS = {s.id: s for s in ui_labels.SECTIONS}


def choices(section: ui_labels.UiSection, experimental: bool) -> list[tuple[str, str]]:
    """Chips of a section as ``(display, id)``; the "(apparent)" suffix is in the section note."""
    return [(i.name.removesuffix(" (apparent)"), i.id) for i in
            ui_labels.visible_items(section, experimental)]  # fmt: skip


def to_selection(cams, t_from, t_to, strict, view, *groups) -> Selection:
    """Control values -> :class:`Selection` (one list of chip ids per filter section)."""
    chips = tuple(i for g in groups for i in (g or []))
    return Selection(
        tuple(cams or ()), t_from, t_to, chips, bool(strict), view if view in VIEWS else VIEWS[0]
    )


def scope_names(ctx: AppContext) -> tuple[dict[int, str], dict[str, str]]:
    """(person names, camera names) of the environment."""
    with ctx.db() as conn:
        return views.person_names(conn), {c.id: c.name for c in db.list_cameras(conn)}


def render(ctx: AppContext, sel: Selection, shown: int = PAGE) -> tuple[Any, ...]:
    """Values for ``[state, annotate_btn, main, stats, grid, more_btn, *accordion labels]``."""
    out = find_data.run_find(ctx, sel)
    labels = tuple(
        gr.update(label=_acc_label(title, sel, ids)) for title, ids in GROUPS
    )  # fmt: skip
    if not out.ready:
        status: EnvStatus = out.status
        state = find_states.state_panel(status)
        annotate = gr.update(visible=status.state == "not_annotated")
        return state, annotate, gr.update(visible=False), "", "", gr.update(visible=False), *labels
    names, cams = scope_names(ctx)
    with ctx.db() as conn:
        view = ctx.show_view(conn)
    html, more = find_views.results_html(ctx.data_dir, out, sel, names, cams, shown, view)
    btn = gr.update(visible=more > 0, value=f"Show more ({more:,} left)")
    return (
        "", gr.update(visible=False), gr.update(visible=True), find_views.stats_html(out, sel),
        html, btn, *labels,
    )  # fmt: skip


def _acc_label(title: str, sel: Selection, section_ids: tuple[str, ...]) -> str:
    ids = [i.id for s in section_ids for i in SECTIONS[s].items]
    n = find_data.section_state(sel, ids)
    return f"{title} ({n})" if n else title


def scope_updates(
    ctx: AppContext, cams=None, t_from=None, t_to=None, bounds=None, reset: bool = True
) -> tuple[Any, Any, Any, tuple[float, float]]:
    """Camera dropdown and time sliders for the selected environment.

    ``reset`` clears the choices (a new environment). Otherwise the user's camera choice and any
    time bound they moved are kept; a slider still at the old end of the footage follows the
    new end.
    """
    with ctx.db() as conn:
        options = [(c.name, c.id) for c in db.list_cameras(conn)]
        table = find_data.get_table(ctx, conn)
    lo, hi = find_data.time_bounds(table)
    lo, hi = float(math.floor(lo)), float(math.ceil(hi))
    hi = max(hi, lo + 1)
    keep = {i for _, i in options}
    cams = [] if reset else [c for c in (cams or []) if c in keep]
    old_lo, old_hi = bounds or (lo, hi)
    start = lo if reset or t_from is None or t_from <= old_lo else min(max(t_from, lo), hi)
    end = hi if reset or t_to is None or t_to >= old_hi else min(max(t_to, lo), hi)
    return (
        gr.update(choices=options, value=cams),
        gr.update(minimum=lo, maximum=hi, value=start),
        gr.update(minimum=lo, maximum=hi, value=end),
        (lo, hi),
    )


def open_detail(ctx: AppContext, key: str, sel: Selection):
    """Drawer content for the card ``key`` of the current results."""
    out = find_data.run_find(ctx, sel)
    person = find_data.find_person(out, key) if out.ready else None
    if person is None:
        gr.Info("That result changed. Click a card again.")
        return (gr.update(),) * 6
    named = person.gid is not None
    head, body = detail_for(ctx, out, person, sel)
    return (
        gr.update(visible=True), head, body, person.gid,
        gr.update(visible=named), gr.update(visible=named),
    )  # fmt: skip


def detail_for(
    ctx: AppContext, out: find_data.Outcome, person: PersonHit, sel: Selection
) -> tuple[str, str]:
    names, cams = scope_names(ctx)
    assert out.table is not None
    if person.gid is not None:
        with ctx.db() as conn:
            probs = person_attributes(conn, person.gid, out.table)
            face = any(i.has_face for i in db.list_identities(conn) if i.gid == person.gid)
    else:
        probs = dict(zip(out.table.head.labels, person.best.probs.tolist(), strict=True))
        face = False
    with ctx.db() as conn:
        view = ctx.show_view(conn)
    return find_views.detail_html(
        ctx.data_dir, person, ui_labels.describe(probs), sel, names, cams, face, sel.has_terms,
        view,
    )  # fmt: skip


def build(ctx: AppContext) -> gr.Tab:
    with gr.Tab("Find", id="find") as tab:
        shown = gr.State(PAGE)
        detail_gid = gr.State(None)
        bounds = gr.State((0.0, 1.0))
        state = gr.HTML(elem_classes="rd-fstate")
        annotate = gr.Button("Annotate now", variant="primary", visible=False, size="lg")
        with gr.Row(equal_height=False, visible=False) as main:
            with gr.Column(scale=3, min_width=460):
                with gr.Row(equal_height=True, elem_classes="rd-fhead"):
                    stats = gr.HTML(elem_classes="rd-fstats-slot")
                    view = gr.Radio(
                        list(VIEWS), value=VIEWS[0], show_label=False, container=False,
                        elem_classes="rd-seg", scale=0, min_width=240,
                    )  # fmt: skip
                grid = gr.HTML()
                more = gr.Button("Show more", visible=False, elem_classes="rd-more")
            with gr.Column(scale=1, min_width=320, elem_classes="rd-filter"):
                with gr.Row(elem_classes="rd-filter-top"):
                    gr.HTML('<div class="rd-filter-title">Filters</div>')
                    clear = gr.Button("Clear all", size="sm", scale=0, min_width=90)
                strict = gr.Checkbox(
                    label="Strict matching", info=STRICT_INFO, value=False, elem_classes="rd-switch"
                )
                with gr.Accordion("Where and when", open=False):
                    cameras = gr.Dropdown(
                        [],
                        multiselect=True,
                        label="Cameras",
                        info="Empty means all cameras. Times are seconds into the footage.",
                    )
                    t_from = gr.Slider(0, 1, value=0, step=1, label="From (s)")
                    t_to = gr.Slider(0, 1, value=1, step=1, label="To (s)")
                groups: list[gr.CheckboxGroup] = []
                accordions: list[gr.Accordion] = []
                for title, ids in GROUPS:
                    with gr.Accordion(title, open=ids[0] in ("person", "upper")) as acc:
                        accordions.append(acc)
                        for sid in ids:
                            groups.append(_chip_group(SECTIONS[sid]))
                experimental = gr.Checkbox(
                    label="Show experimental filters",
                    info="Adds labels the model gets wrong more often, such as jacket or hat.",
                    elem_classes="rd-switch",
                )  # fmt: skip
        drawer, close, head, detail, to_explore, to_clips = _drawer()
        pick = gr.Textbox(elem_id="find-pick", elem_classes="rd-hidden", show_label=False)

    controls = [cameras, t_from, t_to, strict, view, *groups]
    outs = [state, annotate, main, stats, grid, more, *accordions]

    def run(*vals):
        return (*render(ctx, to_selection(*vals)), PAGE)

    def run_more(n, *vals):
        return (*render(ctx, to_selection(*vals), n + PAGE), n + PAGE)

    def reset_groups(experimental_on: bool, *current):
        updates = []
        for sid, now in zip(section_ids(), current, strict=True):
            options = choices(SECTIONS[sid], experimental_on)
            keep = [v for v in (now or []) if v in {i for _, i in options}]
            updates.append(gr.update(choices=options, value=keep))
        return updates

    def clear_all():
        cams, lo, hi, limits = scope_updates(ctx)
        return cams, lo, hi, limits, False, *[[] for _ in groups]

    def keep_scope(cams, t_from, t_to, limits):
        return scope_updates(ctx, cams, t_from, t_to, limits, reset=False)

    def on_pick(raw: str, *vals):
        return open_detail(ctx, (raw or "").split("|")[0], to_selection(*vals))

    def open_clip(gid, cams, start, end, limits):
        """Open Clips on this person with Find's cameras and time window (unchanged = open)."""
        lo, hi = limits
        a = None if start is None or start <= lo else start
        b = None if end is None or end >= hi else end
        return clips_data.make_prefill(gid, cams, a, b), gr.Tabs(selected="clips")

    def annotate_click():
        gr.Info(find_data.annotate_now(ctx))

    run_io = dict(fn=run, inputs=controls, outputs=[*outs, shown], show_progress="hidden")
    changed = [c.input for c in (cameras, strict, view, *groups)] + [t_from.release, t_to.release]
    gr.on(changed, **run_io)
    experimental.input(reset_groups, [experimental, *groups], groups).then(**run_io)
    clear.click(clear_all, None, [cameras, t_from, t_to, bounds, strict, *groups]).then(**run_io)
    more.click(run_more, [shown, *controls], [*outs, shown], show_progress="hidden")
    scope_io = dict(fn=keep_scope, inputs=[cameras, t_from, t_to, bounds],
                    outputs=[cameras, t_from, t_to, bounds])  # fmt: skip
    annotate.click(annotate_click, None, None, concurrency_limit=1, concurrency_id="annotate").then(
        **scope_io
    ).then(**run_io)
    gr.on(
        envbar.refresh_on(ctx, ctx.ui["demo"].load),
        lambda: scope_updates(ctx),
        None,
        [cameras, t_from, t_to, bounds],
    ).then(**run_io)
    tab.select(**scope_io).then(**run_io)
    pick.change(on_pick, [pick, *controls],
               [drawer, head, detail, detail_gid, to_explore, to_clips],
               show_progress="hidden")  # fmt: skip
    close.click(lambda: gr.update(visible=False), None, drawer)
    to_explore.click(
        lambda g: (g, gr.Tabs(selected="explore")), detail_gid,
        [ctx.ui["explore_person"], ctx.ui["tabs"]],
    )  # fmt: skip
    to_clips.click(
        open_clip, [detail_gid, cameras, t_from, t_to, bounds],
        [ctx.ui["clips_prefill"], ctx.ui["tabs"]],
    )  # fmt: skip
    return tab


def section_ids() -> list[str]:
    """Section ids in the order the chip groups are created."""
    return [sid for _, ids in GROUPS for sid in ids]


def _chip_group(section: ui_labels.UiSection) -> gr.CheckboxGroup:
    if section.id in SUBTITLE:
        gr.HTML(f'<div class="rd-fsub-title">{SUBTITLE[section.id]}</div>')
    kind = "rd-sw" if section.kind == "swatches" else "rd-chips"
    group = gr.CheckboxGroup(
        choices(section, False), show_label=False, container=False,
        elem_classes=["rd-filter-group", kind],
    )  # fmt: skip
    if section.note:
        gr.HTML(f'<p class="rd-hint">{section.note}</p>')
    return group


def _drawer():
    """The detail panel: fixed to the right edge, hidden until a card is clicked."""
    with gr.Column(visible=False, elem_classes="rd-drawer") as drawer:
        close = gr.Button("Close", size="sm", elem_classes="rd-drawer-close", scale=0)
        head = gr.HTML()
        with gr.Row(elem_classes="rd-drawer-actions"):
            to_explore = gr.Button("Open in Explore", variant="primary", visible=False)
            to_clips = gr.Button("Follow clip", visible=False)
        detail = gr.HTML()
    return drawer, close, head, detail, to_explore, to_clips
