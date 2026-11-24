"""Process tab: pick footage, choose a preset, run the pipeline with live progress."""

from __future__ import annotations

import shutil
from pathlib import Path

import gradio as gr
import pandas as pd

from reid import analytics
from reid.app import components as C
from reid.app import views
from reid.app.state import AppContext
from reid.config import CameraSet, CameraSpec, load_cameras
from reid.identity import IdentityEngine
from reid.pipeline import run

PRESETS = ["lite", "balanced", "accurate", "max"]
PRESET_HELP = {
    "lite": "Small and quick. Fits a 3 GB GPU or runs on CPU.",
    "balanced": "Good accuracy on a Colab T4 (default).",
    "accurate": "Separate detector and pose models. Slower, a bit better in crowds.",
    "max": "Heaviest models. Needs a T4 with room to spare.",
}
# (config key, label, help, min, max, step) for the advanced sliders
SLIDERS = [
    (
        "association.min_cos",
        "Body match threshold",
        "Minimum similarity to join a person.",
        0.5,
        0.95,
        0.01,
    ),
    (
        "association.tau",
        "Match confidence ratio",
        "How far the best person must beat the runner-up.",
        1.0,
        4.0,
        0.1,
    ),
    (
        "consolidation.merge_threshold",
        "Merge threshold",
        "Distance below which people merge at the end.",
        0.05,
        0.6,
        0.01,
    ),
]
VIDEO_TYPES = [".mp4", ".mov", ".avi", ".mkv", ".webm"]


def _get(cfg, dotted: str):
    for part in dotted.split("."):
        cfg = getattr(cfg, part)
    return cfg


def camera_yamls(ctx: AppContext) -> list[str]:
    found = sorted(Path("configs/cameras").glob("*.yaml")) + sorted(ctx.data_dir.glob("*.yaml"))
    return [str(p) for p in found]


def uploaded_set(ctx: AppContext, paths: list[str], face_names: list[str]) -> CameraSet:
    """Camera set from uploaded videos, copied to ``data_dir/uploads`` so renders keep working."""
    dest = ctx.data_dir / "uploads"
    dest.mkdir(exist_ok=True)
    cams = []
    for i, src in enumerate(paths, 1):
        name = Path(src).name
        target = dest / name
        if not target.exists():
            shutil.copy2(src, target)
        cams.append(
            CameraSpec(
                id=f"cam_{i:02d}", name=Path(name).stem, video_path=str(target),
                is_face_cam=name in face_names,
            )
        )  # fmt: skip
    return CameraSet(cameras=cams)


def stage_table(stats: dict) -> pd.DataFrame:
    ms = stats.get("ms_per_frame", {})
    total = sum(ms.values()) or 1.0
    rows = [
        {"Stage": k, "ms per frame": round(v, 1), "Share": f"{100 * v / total:.0f}%"}
        for k, v in ms.items()
        if v > 0
    ]
    return pd.DataFrame(rows, columns=["Stage", "ms per frame", "Share"])


def start(ctx, source, yaml_path, videos, face_names, preset, stride, max_s, sliders, progress):
    """Run the pipeline; returns (status html, KPI tiles, stage table)."""
    if source == "Upload videos":
        if not videos:
            raise gr.Error("Upload at least one video, or choose a camera set.")
        cameras = uploaded_set(ctx, videos, face_names or [])
    else:
        if not yaml_path:
            raise gr.Error("Choose a camera set YAML.")
        cameras = load_cameras(yaml_path)
    overrides = {path: value for (path, *_), value in zip(SLIDERS, sliders, strict=True)}
    overrides["runtime.frame_stride"] = int(stride)
    overrides["runtime.end_s"] = float(max_s) if max_s else None
    cfg = ctx.derive_cfg(preset, overrides)
    ids = [c.id for c in cameras.cameras]

    def on_progress(cam: str, done: int, total: int) -> None:
        frac = (ids.index(cam) + (done / total if total else 0.0)) / len(ids)
        progress(frac, desc=f"{cam}: frame {done}" + (f" of {total}" if total else ""))

    progress(0.0, desc="Loading models (first run downloads weights)")
    with ctx.db() as conn:
        engine = IdentityEngine(cfg, conn, ctx.vectors(), cameras)
        res = run(cfg, cameras, engine, on_progress, models=ctx.models(cfg), conn=conn)
        stats = res.to_dict()
        ctx.save_run_stats(res.run_id, stats)
        summary = analytics.run_summary(conn, res.run_id)
    status = C.badge("Run finished", "ok") + C.badge(f"run #{res.run_id}", "dim")
    table = gr.update(value=stage_table(stats), visible=True)
    return status, views.summary_tiles(summary, stats), table


def build(ctx: AppContext) -> gr.Tab:
    cfg = ctx.cfg
    with gr.Tab("Process", id="process") as tab:
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                source = gr.Radio(
                    ["Camera set", "Upload videos"], value="Camera set", label="Footage"
                )
                yamls = camera_yamls(ctx)
                yaml_in = gr.Dropdown(
                    yamls, value=yamls[0] if yamls else None, allow_custom_value=True,
                    label="Camera set file", info="A YAML that lists the videos.",
                )  # fmt: skip
                videos = gr.File(
                    label="Videos", file_count="multiple", file_types=VIDEO_TYPES,
                    type="filepath", visible=False,
                )  # fmt: skip
                faces = gr.CheckboxGroup(
                    [], label="Face cameras", visible=False,
                    info="Cameras that see faces up close, such as an entrance.",
                )  # fmt: skip
            with gr.Column(scale=2):
                preset = gr.Dropdown(PRESETS, value="lite", label="Model preset")
                hint = gr.Markdown(PRESET_HELP["lite"], elem_classes="rd-hint")
                with gr.Accordion("Advanced settings", open=False):
                    sliders = [
                        gr.Slider(lo, hi, _get(cfg, key), step=step, label=label, info=info)
                        for key, label, info, lo, hi, step in SLIDERS
                    ]
                    stride = gr.Slider(
                        1, 10, cfg.runtime.frame_stride, step=1, label="Process every n-th frame"
                    )
                    max_s = gr.Number(
                        cfg.runtime.end_s,
                        label="Stop after (seconds)",
                        info="Empty processes whole videos.",
                    )
        go = gr.Button("Process footage", variant="primary", size="lg")
        status = gr.HTML()
        kpis = gr.HTML()
        stages = gr.Dataframe(label="Time per frame by stage", interactive=False, visible=False)

    def switch(src):
        up = src == "Upload videos"
        return gr.update(visible=not up), gr.update(visible=up), gr.update(visible=up)

    source.change(switch, source, [yaml_in, videos, faces])
    videos.change(lambda f: gr.update(choices=[Path(x).name for x in f or []], visible=bool(f)),
                  videos, faces)  # fmt: skip
    preset.change(lambda p: PRESET_HELP.get(p, ""), preset, hint)
    inputs = [source, yaml_in, videos, faces, preset, stride, max_s, *sliders]

    def go_click(*args, progress=gr.Progress()):  # noqa: B008 (Gradio's progress idiom)
        return start(ctx, *args[:7], args[7:], progress)

    go.click(
        go_click, inputs, [status, kpis, stages], concurrency_limit=1, concurrency_id="gpu",
        show_progress_on=[status],
    )  # fmt: skip
    return tab
