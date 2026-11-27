"""Clips tab: plan a follow-person clip, preview it with a render time estimate, then render it.

Layout and wiring only. Planning, the estimate and rendering live in :mod:`reid.viz.clip_plan`,
:mod:`reid.viz.clip_cost` and :mod:`reid.viz.journey`; the tab's logic is in
:mod:`reid.app.clips_data` and its HTML in :mod:`reid.app.clips_view`. Other tabs open it through
``ctx.ui["clips_prefill"]`` (see :func:`reid.app.clips_data.make_prefill`).
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from reid.app import clips_data as D
from reid.app import clips_view as V
from reid.app import components as C
from reid.app import envbar
from reid.app.state import AppContext
from reid.config import Config


def _width_default(cfg: Config) -> int:
    return min(D.WIDTHS, key=lambda w: abs(w - cfg.render.journey_width))


def scope_updates(
    ctx: AppContext,
    gid: int | None,
    cams: list[str] | None = None,
    start: float | None = None,
    end: float | None = None,
) -> tuple[Any, Any, Any, tuple[float, float]]:
    """Camera checkboxes and time sliders for ``gid``; ``cams`` / ``start`` / ``end`` are kept
    (clamped to what exists) when given, else the person's own cameras and presence."""
    with ctx.db() as conn:
        sc = D.person_scope(conn, gid)
    known = {i for _, i in sc.cameras}
    keep = [c for c in (cams or []) if c in known] or sc.default

    def clamp(v: float | None, fallback: float) -> float:
        return fallback if v is None else min(max(float(v), sc.lo), sc.hi)

    t0, t1 = clamp(start, sc.lo), clamp(end, sc.hi)
    if t1 <= t0:
        t0, t1 = sc.lo, sc.hi
    return (
        gr.update(choices=sc.cameras, value=keep),
        gr.update(minimum=sc.lo, maximum=sc.hi, value=t0),
        gr.update(minimum=sc.lo, maximum=sc.hi, value=t1),
        (sc.lo, sc.hi),
    )


def people_update(
    ctx: AppContext, current: int | None, reset: bool = False, keep_value: bool = False
):
    """Person dropdown with fresh choices; ``current`` stays selected if it still exists.

    ``keep_value`` leaves the selection to the browser (used when a tab opens, where another
    handler may be setting the person at the same time).
    """
    with ctx.db() as conn:
        choices = D.people_choices(conn)
    if keep_value:
        return gr.update(choices=choices)
    known = [g for _, g in choices]
    if reset or current not in known:
        current = known[0] if known else None
    return gr.update(choices=choices, value=current)


def build(ctx: AppContext) -> gr.Tab:
    prefill = ctx.ui["clips_prefill"]
    with gr.Tab("Clips", id="clips") as tab:
        bounds = gr.State((0.0, 1.0))
        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=300):
                person = gr.Dropdown([], label="Person", info="Whose journey to follow.")
                cameras = gr.CheckboxGroup(
                    [], label="Cameras", info="The clip cuts only between the ticked cameras."
                )
                with gr.Accordion("Time and speed", open=True):
                    t_from = gr.Slider(0, 1, value=0, step=1, label="From (s)")
                    t_to = gr.Slider(0, 1, value=1, step=1, label="To (s)")
                    speed = gr.Radio(
                        D.SPEEDS, value=1.0, label="Playback speed",
                        info="Faster playback also renders faster.",
                    )  # fmt: skip
                    max_s = gr.Slider(
                        0, 600, value=D.DEFAULT_MAX_S, step=10, label="Max clip length (s)",
                        info="0 = no limit. A longer journey is cut off at this length.",
                    )  # fmt: skip
                with gr.Accordion("Look", open=True):
                    blur = gr.Checkbox(
                        label="Blur other people",
                        info="Privacy: only the followed person stays clear.",
                    )
                    others = gr.Checkbox(label="Show other people's boxes", value=True)
                    thumbs = gr.Checkbox(label="Side thumbnails of other cameras", value=True)
                    n_thumbs = gr.Slider(1, 8, value=4, step=1, label="Thumbnails shown")
                    width = gr.Dropdown(
                        D.WIDTHS, value=_width_default(ctx.cfg), label="Width (px)",
                        filterable=False,
                    )  # fmt: skip
                    fps = gr.Dropdown(D.FPS_CHOICES, value=0, label="Frame rate", filterable=False)
            with gr.Column(scale=3):
                tiles = gr.HTML()
                notices = gr.HTML(elem_classes="rd-note")
                gr.Markdown("### Plan")
                plot = gr.Plot(show_label=False)
                table = gr.HTML()
                render = gr.Button("Render clip", variant="primary", size="lg")
                result = gr.HTML(elem_classes="rd-note")
                video = gr.Video(show_label=False, autoplay=False)
    ctx.ui["clips_person"] = person

    settings = [cameras, t_from, t_to, speed, max_s, blur, others, thumbs, n_thumbs, width, fps]
    outs = [tiles, notices, plot, table, render]

    def preview(gid, *vals):
        if gid is None:
            hint = C.notice("Pick a person", "Choose whose journey to turn into a clip.")
            return "", hint, None, "", gr.update(interactive=False)
        plan, est = D.preview_plan(ctx, int(gid), D.to_options(*vals))
        return (
            V.tiles_html(plan, est, ctx.clip_model().factor != 1.0),
            V.notices_html(ctx, plan, est),
            V.timetable_figure(plan),
            V.plan_table_html(plan),
            gr.update(interactive=not plan.empty),
        )

    def on_person(gid):
        return (*scope_updates(ctx, gid), None, "")

    def reload():
        """After an environment change (ids repeat across environments): start over."""
        upd = people_update(ctx, None, reset=True)
        return (upd, *scope_updates(ctx, upd["value"]), None, "")

    def apply_prefill(p):
        if not p:
            return (gr.update(),) * 4 + (gr.update(), None, "")
        upd = people_update(ctx, p["gid"])
        gid = upd["value"]
        keep = p["cameras"] if gid == p["gid"] else None
        return (upd, *scope_updates(ctx, gid, keep, p["start"], p["end"]), None, "")

    def render_click(gid, *vals, progress=gr.Progress()):  # noqa: B008 (Gradio's progress idiom)
        if gid is None:
            raise gr.Error("Pick a person first.")
        opts = D.to_options(*vals)
        _, est = D.preview_plan(ctx, int(gid), opts)
        progress(0.0, desc="Rendering the clip")
        try:
            path, plan, seconds = D.render_cached(
                ctx, int(gid), opts, lambda done, total: progress(done / max(total, 1))
            )
        except FileNotFoundError:
            return None, C.missing_video_notice()
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        return str(path), V.result_note(plan, est, seconds)

    scope = [cameras, t_from, t_to, bounds]
    cleared = [video, result]
    preview_io = dict(fn=preview, inputs=[person, *settings], outputs=outs, show_progress="hidden")
    changed = [c.input for c in (cameras, speed, blur, others, thumbs, n_thumbs, width, fps)]
    changed += [t_from.release, t_to.release, max_s.release]
    gr.on(changed, **preview_io)
    gr.on(changed, lambda: "", None, result, queue=False)  # the note belongs to the last render
    person.input(on_person, person, [*scope, *cleared]).then(**preview_io)
    gr.on(
        envbar.refresh_on(ctx, ctx.ui["demo"].load), reload, None, [person, *scope, *cleared]
    ).then(**preview_io)
    prefill.change(apply_prefill, prefill, [person, *scope, *cleared]).then(**preview_io)
    # only the choices: a link from another tab sets the person at the same moment
    tab.select(lambda: people_update(ctx, None, keep_value=True), None, person).then(**preview_io)
    render.click(lambda: "", None, result, queue=False).then(
        render_click, [person, *settings], cleared, concurrency_limit=1, concurrency_id="render"
    ).then(**preview_io)
    return tab
