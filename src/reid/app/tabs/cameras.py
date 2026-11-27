"""Cameras tab: annotated videos, the synchronized mosaic and camera hand-offs."""

from __future__ import annotations

import gradio as gr

from reid import analytics
from reid.app import components as C
from reid.app import envbar, videos, views
from reid.app.state import AppContext
from reid.storage import db
from reid.viz.render import render_camera, render_mosaic

EVERYONE = "Everyone"  # dropdown value 0 means no highlight (gids start at 1)


def choices(ctx: AppContext):
    """(camera dropdown update, person dropdown update)."""
    with ctx.db() as conn:
        cams = [(c.name, c.id) for c in db.list_cameras(conn)]
        people = [(EVERYONE, 0), *views.person_choices(views.people_cards(ctx, conn))]
    return gr.update(choices=cams, value=cams[0][1] if cams else None), gr.update(
        choices=people, value=0
    )


def handoffs(ctx: AppContext) -> str:
    """Camera-to-camera hand-off table (everyone: it does not depend on the highlight)."""
    with ctx.db() as conn:
        names = {c.id: c.name for c in db.list_cameras(conn)}
        cams, mat = analytics.transitions(conn)
        return C.transitions_html([names.get(c, c) for c in cams], mat)


def _progress_cb(progress):
    return lambda done, total: progress(done / max(total, 1), desc="Rendering")


def video(ctx: AppContext, camera: str | None, gid: int | None, secs: float, progress):
    """Annotated video of one camera (cached by camera, person and length)."""
    if not camera:
        raise gr.Error("Choose a camera.")
    gid = gid or None
    end = float(secs) if secs and secs > 0 else None
    with ctx.db() as conn:
        cfg = ctx.render_cfg(conn, gid)  # showcase: a blurring copy of the config
        return str(
            ctx.cached(
                conn,
                "camera",
                {"cam": camera, "gid": gid, "end": end},
                ".mp4",
                lambda p: render_camera(
                    conn,
                    cfg,
                    camera,
                    p,
                    end_s=end,
                    highlight_gid=gid,
                    progress=_progress_cb(progress),
                ),
                tag=f"{camera}_gid{gid}" if gid else camera,
            )  # fmt: skip
        )


def mosaic(ctx: AppContext, gid: int | None, secs: float, progress):
    """All cameras side by side on the shared timeline (cached)."""
    gid = gid or None
    end = float(secs) if secs and secs > 0 else None
    with ctx.db() as conn:
        cfg = ctx.render_cfg(conn, gid)
        return str(
            ctx.cached(
                conn,
                "mosaic",
                {"gid": gid, "end": end},
                ".mp4",
                lambda p: render_mosaic(
                    conn,
                    cfg,
                    None,
                    p,
                    highlight_gid=gid,
                    end_s=end,
                    progress=_progress_cb(progress),
                ),
                tag=f"gid{gid}" if gid else "all",
            )  # fmt: skip
        )


def build(ctx: AppContext) -> gr.Tab:
    with gr.Tab("Cameras", id="cameras") as tab:
        with gr.Row():
            camera = gr.Dropdown([], label="Camera")
            person = gr.Dropdown([], label="Highlight a person", info="Dims everyone else.")
            secs = gr.Number(20, label="Length (seconds)", info="0 renders the whole video.")
        with gr.Row(equal_height=False):
            with gr.Column():
                cam_btn = gr.Button("Render camera video", variant="primary")
                cam_note = gr.HTML(elem_classes="rd-note")
                cam_video = gr.Video(show_label=False)
            with gr.Column():
                mos_btn = gr.Button("Render mosaic of all cameras", variant="primary")
                mos_note = gr.HTML(elem_classes="rd-note")
                mos_video = gr.Video(show_label=False)
        gr.Markdown("### Hand-offs between cameras")
        trans = gr.HTML()

    def cam_click(cam, gid, n, progress=gr.Progress()):  # noqa: B008 (Gradio's progress idiom)
        return videos.render_or_note(lambda: video(ctx, cam, gid, n, progress))

    def mos_click(gid, n, progress=gr.Progress()):  # noqa: B008
        return videos.render_or_note(lambda: mosaic(ctx, gid, n, progress))

    gr.on(
        envbar.refresh_on(ctx, tab.select, ctx.ui["demo"].load),
        lambda: choices(ctx),
        None,
        [camera, person],
    )
    gr.on(envbar.refresh_on(ctx, tab.select), lambda: handoffs(ctx), None, trans)

    def notes() -> tuple[str, str]:
        note = videos.availability_note(ctx)
        return note, note

    gr.on(
        envbar.refresh_on(ctx, tab.select),
        notes,
        None,
        [cam_note, mos_note],
    )
    ctx.ui["env_version"].change(lambda: (None, None), None, [cam_video, mos_video])
    cam_btn.click(
        cam_click, [camera, person, secs], [cam_video, cam_note],
        concurrency_limit=1, concurrency_id="render",
    )  # fmt: skip
    mos_btn.click(
        mos_click, [person, secs], [mos_video, mos_note],
        concurrency_limit=1, concurrency_id="render",
    )  # fmt: skip
    return tab
