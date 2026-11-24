"""Admin tab: re-cluster, merge, split, rename, and delete or wipe data (not with --read-only)."""

from __future__ import annotations

import shutil

import gradio as gr

from reid.app import components as C
from reid.app import envbar, views
from reid.app import environments as envs
from reid.app.state import AppContext
from reid.identity.consolidate import (
    merge_identities,
    recluster,
    rename_identity,
    split_tracklet,
)
from reid.storage import db, maintenance


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


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise gr.Error(what)


def _run_choices(ctx: AppContext):
    with ctx.db() as conn:
        rows = [(f"Run {r['id']} ({r['status']})", int(r["id"])) for r in db.list_runs(conn)]
    return gr.update(choices=rows, value=None)


def _maint(ctx: AppContext, op, *args, **kw) -> dict:
    """Run ``op(conn, vectors, cfg, *args)`` from the maintenance module and drop cached renders."""
    with ctx.db() as conn:
        out = op(conn, ctx.vectors(), ctx.cfg, *args, **kw)
    shutil.rmtree(ctx.data_dir / "renders" / "app", ignore_errors=True)  # stale after deletions
    return out


def do_delete_person(ctx: AppContext, gid, keep: bool, ok: bool):
    _need(gid is not None, "Choose a person.")
    _need(ok, "Tick the box to confirm.")
    try:
        out = _maint(ctx, maintenance.delete_identity, int(gid), keep_tracklets=bool(keep))
    except KeyError as exc:
        raise gr.Error(str(exc)) from exc
    what = (
        f"{out['tracklets_unassigned']} sightings kept as unidentified"
        if keep
        else f"{out['tracklets_deleted']} sightings deleted"
    )
    return _done(f"Deleted person {int(gid)}, {what}"), *people_update(ctx), _run_choices(ctx)


def do_delete_run(ctx: AppContext, run_id, ok: bool):
    _need(run_id is not None, "Choose a run.")
    _need(ok, "Tick the box to confirm.")
    try:
        out = _maint(ctx, maintenance.delete_run, int(run_id))
    except KeyError as exc:
        raise gr.Error(str(exc)) from exc
    text = (
        f"Deleted run {int(run_id)}: {out['tracklets_deleted']} sightings, "
        f"{out['identities_dropped']} people left without sightings removed"
    )
    return _done(text), *people_update(ctx), _run_choices(ctx)


def do_wipe(ctx: AppContext, typed: str):
    _need(typed.strip() == ctx.env, f"Type the environment name ({ctx.env}) to confirm.")
    maintenance.wipe_data_dir(ctx.data_dir)
    return C.badge(f"Environment {ctx.env} emptied", "bad"), *people_update(ctx), _run_choices(ctx)


def do_delete_env(ctx: AppContext, typed: str):
    """Delete the selected environment and switch to another (``default`` is created if none)."""
    _need(typed.strip() == ctx.env, f"Type the environment name ({ctx.env}) to confirm.")
    gone = ctx.env
    try:
        envs.delete_environment(ctx.workspace, gone)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    remaining = ctx.environments()
    ctx.select_env(remaining[0].name if remaining else envs.DEFAULT_ENV, create=not remaining)
    return C.badge(f"Environment {gone} deleted, now in {ctx.env}", "bad")


def build(ctx: AppContext) -> gr.Tab:
    with gr.Tab("Admin", id="admin") as tab:
        msg = gr.HTML()
        gr.Markdown(
            "Changes apply right away. A split can be undone by a merge; deletions cannot.",
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
        if not ctx.read_only:
            with gr.Accordion("Danger zone: delete data", open=False, elem_classes="rd-danger"):
                gr.Markdown(
                    "These act on the environment selected in the bar above and cannot be undone.",
                    elem_classes="rd-hint",
                )
                with gr.Row(equal_height=False):
                    with gr.Column():
                        gr.Markdown("### Delete a person")
                        del_who = gr.Dropdown([], label="Person")
                        keep = gr.Checkbox(label="Keep their sightings as unidentified")
                        del_ok = gr.Checkbox(label="I understand this cannot be undone")
                        del_btn = gr.Button("Delete person", variant="stop")
                    with gr.Column():
                        gr.Markdown("### Delete a run")
                        del_run = gr.Dropdown([], label="Run")
                        run_ok = gr.Checkbox(label="I understand this cannot be undone")
                        run_btn = gr.Button("Delete run", variant="stop")
                with gr.Row(equal_height=False):
                    with gr.Column():
                        gr.Markdown("### Empty this environment")
                        wipe_typed = gr.Textbox(label="Type the environment name to confirm")
                        wipe_btn = gr.Button("Delete all its data", variant="stop")
                    with gr.Column():
                        gr.Markdown("### Delete this environment")
                        env_typed = gr.Textbox(label="Type the environment name to confirm")
                        env_btn = gr.Button("Delete the environment", variant="stop")

    dropdowns = [a, b, who, sighting]
    gr.on(
        envbar.refresh_on(ctx, tab.select, ctx.ui["demo"].load),
        lambda: people_update(ctx),
        None,
        dropdowns,
    )
    who.change(lambda g: tracklet_choices(ctx, g), who, sighting)
    outs = [msg, *dropdowns]
    recluster_btn.click(lambda: do_recluster(ctx), None, outs)
    merge_btn.click(lambda x, y: do_merge(ctx, x, y), [a, b], outs)
    split_btn.click(lambda t: do_split(ctx, t), sighting, outs)
    rename_btn.click(lambda g, t: do_rename(ctx, g, t), [who, label], outs)
    if ctx.read_only:
        return tab

    def pickers():
        return people_update(ctx)[0], _run_choices(ctx)

    gr.on(
        envbar.refresh_on(ctx, tab.select, ctx.ui["demo"].load), pickers, None, [del_who, del_run]
    )
    del_outs = [*outs, del_run]
    del_btn.click(lambda g, k, o: do_delete_person(ctx, g, k, o), [del_who, keep, del_ok], del_outs)
    run_btn.click(lambda r, o: do_delete_run(ctx, r, o), [del_run, run_ok], del_outs)
    wipe_btn.click(lambda t: do_wipe(ctx, t), wipe_typed, del_outs)

    def after_env_delete(typed: str):
        return (do_delete_env(ctx, typed), *envbar.refreshed(ctx))

    env_btn.click(
        after_env_delete,
        env_typed,
        [msg, ctx.ui["env_dropdown"], ctx.ui["env_version"], ctx.ui["env_message"]],
    )
    return tab
