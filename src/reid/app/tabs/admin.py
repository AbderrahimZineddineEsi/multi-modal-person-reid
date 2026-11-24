"""Admin tab (only with ``reid app --admin``): re-cluster, merge, split, rename and reset."""

from __future__ import annotations

import gradio as gr

from reid.app import components as C
from reid.app import views
from reid.app.state import AppContext
from reid.identity.consolidate import (
    merge_identities,
    recluster,
    rename_identity,
    split_tracklet,
)
from reid.storage import db, files

CONFIRM_WORD = "RESET"


def people_update(ctx: AppContext):
    """Refresh the person dropdowns after a change."""
    with ctx.db() as conn:
        ch = views.person_choices(views.people_cards(ctx, conn))
    return [gr.update(choices=ch, value=None)] * 4


def tracklet_choices(ctx: AppContext, gid: int | None):
    if gid is None:
        return gr.update(choices=[], value=None)
    with ctx.db() as conn:
        rows = [
            (f"Sighting {t.id} on {t.camera_id}, {t.start_ts:.1f} s", t.id)
            for t in db.get_tracklets(conn, gid=int(gid))
        ]
    return gr.update(choices=rows, value=None)


def _done(text: str) -> str:
    return C.badge(text, "ok")


def do_recluster(ctx: AppContext):
    with ctx.db() as conn:
        res = recluster(conn, ctx.vectors(), ctx.cfg)
    return _done(
        f"{len(res.merges)} merged, {res.n_before} to {res.n_after} people"
    ), *people_update(ctx)


def do_merge(ctx: AppContext, a, b):
    if a is None or b is None:
        raise gr.Error("Choose two people.")
    try:
        with ctx.db() as conn:
            kept = merge_identities(conn, ctx.vectors(), ctx.cfg, int(a), int(b))
    except (ValueError, KeyError) as exc:
        raise gr.Error(str(exc)) from exc
    return _done(f"Merged into person {kept}"), *people_update(ctx)


def do_split(ctx: AppContext, tracklet_id):
    if tracklet_id is None:
        raise gr.Error("Choose a sighting to split off.")
    try:
        with ctx.db() as conn:
            new = split_tracklet(conn, ctx.vectors(), ctx.cfg, int(tracklet_id))
    except (ValueError, KeyError) as exc:
        raise gr.Error(str(exc)) from exc
    return _done(f"Created person {new}"), *people_update(ctx)


def do_rename(ctx: AppContext, gid, label: str):
    if gid is None:
        raise gr.Error("Choose a person.")
    with ctx.db() as conn:
        rename_identity(conn, int(gid), label.strip() or None)
    return _done("Renamed"), *people_update(ctx)


def do_reset(ctx: AppContext, typed: str):
    if typed.strip() != CONFIRM_WORD:
        raise gr.Error(f"Type {CONFIRM_WORD} to confirm.")
    files.reset_data_dir(ctx.data_dir, confirm=True)
    return C.badge("All data deleted", "bad"), *people_update(ctx)


def build(ctx: AppContext) -> gr.Tab:
    with gr.Tab("Admin", id="admin") as tab:
        msg = gr.HTML()
        gr.Markdown(
            "Changes apply right away. A split can be undone by a merge; a reset cannot.",
            elem_classes="rd-hint",
        )
        with gr.Row(equal_height=False):
            with gr.Column():
                gr.Markdown("### Merge two people")
                a = gr.Dropdown([], label="First person")
                b = gr.Dropdown([], label="Second person")
                merge_btn = gr.Button("Merge")
                gr.Markdown("### Re-cluster everyone")
                recluster_btn = gr.Button("Re-cluster all people")
            with gr.Column():
                gr.Markdown("### Rename or split")
                who = gr.Dropdown([], label="Person")
                label = gr.Textbox(label="New name", placeholder="Leave empty to clear")
                rename_btn = gr.Button("Rename")
                sighting = gr.Dropdown([], label="Sighting to split off into a new person")
                split_btn = gr.Button("Split off sighting")
        with gr.Accordion("Delete all data", open=False):
            gr.Markdown(f"Removes the database, vectors, crops and renders in `{ctx.data_dir}`.")
            typed = gr.Textbox(label=f"Type {CONFIRM_WORD} to confirm")
            reset_btn = gr.Button("Delete everything", variant="stop")

    dropdowns = [a, b, who, sighting]
    gr.on([tab.select, ctx.ui["demo"].load], lambda: people_update(ctx), None, dropdowns)
    who.change(lambda g: tracklet_choices(ctx, g), who, sighting)
    outs = [msg, *dropdowns]
    recluster_btn.click(lambda: do_recluster(ctx), None, outs)
    merge_btn.click(lambda x, y: do_merge(ctx, x, y), [a, b], outs)
    split_btn.click(lambda t: do_split(ctx, t), sighting, outs)
    rename_btn.click(lambda g, t: do_rename(ctx, g, t), [who, label], outs)
    reset_btn.click(lambda t: do_reset(ctx, t), typed, outs)
    return tab
