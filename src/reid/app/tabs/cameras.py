"""Cameras tab: annotated videos, the synchronized mosaic, heatmaps and camera hand-offs."""

from __future__ import annotations

import gradio as gr

from reid import analytics
from reid.app import components as C
from reid.app import views
from reid.app.state import AppContext
from reid.storage import db
from reid.viz.heatmap import render_heatmap
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


def overview(ctx: AppContext, gid: int | None):
    gid = gid or None
    """(heatmap gallery items, transitions html), optionally for one person."""
    with ctx.db() as conn:
        names = {c.id: c.name for c in db.list_cameras(conn)}
        maps = []
        for cid, name in names.items():
            path = ctx.cached(
                conn, "heat", {"cam": cid, "gid": gid}, ".jpg",
                lambda p, cid=cid: render_heatmap(conn, ctx.cfg, cid, p, gid),
            )  # fmt: skip
            maps.append((str(path), name))
        cams, mat = analytics.transitions(conn)
        return maps, C.transitions_html([names.get(c, c) for c in cams], mat)


def _progress_cb(progress):
    return lambda done, total: progress(done / max(total, 1), desc="Rendering")


def video(ctx: AppContext, camera: str | None, gid: int | None, secs: float, progress):
    """Annotated video of one camera (cached by camera, person and length)."""
    if not camera:
        raise gr.Error("Choose a camera.")
    gid = gid or None
    end = float(secs) if secs and secs > 0 else None
    with ctx.db() as conn:
        return str(
            ctx.cached(
                conn,
                "camera",
                {"cam": camera, "gid": gid, "end": end},
                ".mp4",
                lambda p: render_camera(
                    conn,
                    ctx.cfg,
                    camera,
                    p,
                    end_s=end,
                    highlight_gid=gid,
                    progress=_progress_cb(progress),
                ),
            )  # fmt: skip
        )


def mosaic(ctx: AppContext, gid: int | None, secs: float, progress):
    """All cameras side by side on the shared timeline (cached)."""
    gid = gid or None
    end = float(secs) if secs and secs > 0 else None
    with ctx.db() as conn:
        return str(
            ctx.cached(
                conn,
                "mosaic",
                {"gid": gid, "end": end},
                ".mp4",
                lambda p: render_mosaic(
                    conn,
                    ctx.cfg,
                    None,
                    p,
                    highlight_gid=gid,
                    end_s=end,
                    progress=_progress_cb(progress),
                ),
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
                cam_video = gr.Video(show_label=False)
            with gr.Column():
                mos_btn = gr.Button("Render mosaic of all cameras", variant="primary")
                mos_video = gr.Video(show_label=False)
        gr.Markdown("### Heatmaps")
        heat = gr.Gallery(
            show_label=False, columns=2, height=360, object_fit="contain", buttons=["fullscreen"]
        )
        gr.Markdown("### Hand-offs between cameras")
        trans = gr.HTML()

    def cam_click(cam, gid, n, progress=gr.Progress()):  # noqa: B008 (Gradio's progress idiom)
        return video(ctx, cam, gid, n, progress)

    def mos_click(gid, n, progress=gr.Progress()):  # noqa: B008
        return mosaic(ctx, gid, n, progress)

    gr.on([tab.select, ctx.ui["demo"].load], lambda: choices(ctx), None, [camera, person])
    gr.on([tab.select, person.change], lambda g: overview(ctx, g), person, [heat, trans])
    cam_btn.click(cam_click, [camera, person, secs], cam_video, concurrency_limit=1,
                  concurrency_id="render")  # fmt: skip
    mos_btn.click(
        mos_click, [person, secs], mos_video, concurrency_limit=1, concurrency_id="render"
    )
    return tab
