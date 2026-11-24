"""Metrics tab: run summary, stage timings, GPU memory and evaluation results."""

from __future__ import annotations

import json

import gradio as gr

from reid import analytics
from reid.app import components as C
from reid.app import views
from reid.app.state import AppContext
from reid.app.tabs.process import stage_table
from reid.storage import db


def run_choices(ctx: AppContext):
    with ctx.db() as conn:
        runs = [(f"Run {r['id']} ({r['status']})", int(r["id"])) for r in db.list_runs(conn)]
    return gr.update(choices=runs, value=runs[-1][1] if runs else None)


def run_view(ctx: AppContext, run_id: int | None):
    """(KPI tiles, stage table update, memory html, fragmentation html)."""
    if run_id is None:
        return (
            C.empty_state("No runs yet", "Process footage to see metrics."),
            gr.update(visible=False),
            "",
            "",
        )
    with ctx.db() as conn:
        summary = analytics.run_summary(conn, int(run_id))
    stats = ctx.load_run_stats(int(run_id))
    table = gr.update(value=stage_table(stats), visible=True) if stats else gr.update(visible=False)
    note = (
        ""
        if stats
        else C.empty_state("No timings for this run", "Runs started in the app record them.")
    )
    return (
        views.summary_tiles(summary, stats),
        table,
        memory_html(ctx, stats) + note,
        fragmentation(summary),
    )


def memory_html(ctx: AppContext, stats: dict | None) -> str:
    items = []
    peaks = (stats or {}).get("vram_peak_mb") or {}
    for stage, mb in peaks.items():
        if stage != "total_peak":
            items.append(C.metric_tile(f"{stage} stage", f"{mb:.0f} MB", "peak GPU memory"))
    live = views.vram_now()
    if live:
        items.append(
            C.metric_tile(
                "GPU in use now",
                f"{live[0]:.1f} GB",
                f"of {live[1]:.1f} GB",
                100 * live[0] / live[1],
            )
        )
    return C.tiles(items) if items else ""


def fragmentation(summary: dict) -> str:
    """Sightings per person as bars: a person seen in many pieces may be several people."""
    per = summary["tracklets_per_identity"]
    if not per:
        return ""
    top = max(per.values())
    rows = "".join(
        f'<div class="rd-cand" style="--pc:#8b8bff"><span>Person {g}</span>'
        f'<div class="rd-meter"><i style="width:{100 * n / top:.0f}%"></i></div><b>{n}</b></div>'
        for g, n in per.items()
    )
    return f'<div class="rd-cands">{rows}</div>'


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def eval_tiles(data: dict) -> str:
    """Headline numbers of a ``reid.eval`` result (missing parts are skipped)."""
    mtmc, rt = data.get("mtmc") or {}, data.get("retrieval") or {}
    frag = data.get("fragmentation") or {}
    items = []
    if mtmc:
        items += [
            C.metric_tile(
                "Cross-camera IDF1", _pct(mtmc.get("idf1")), "same person across cameras"
            ),
            C.metric_tile("MOTA", _pct(mtmc.get("mota")), "tracking accuracy"),
            C.metric_tile(
                "ID switches", str(int(mtmc.get("id_switches") or 0)), "identity changes"
            ),
        ]
    if rt:
        items += [
            C.metric_tile("Rank-1", _pct(rt.get("rank1")), "best match is right"),
            C.metric_tile("mAP", _pct(rt.get("mAP")), "retrieval quality"),
        ]
    if frag:
        items.append(
            C.metric_tile("Purity", _pct(frag.get("gid_purity")), "one person per identity")
        )
    return C.tiles(items)


def eval_results(ctx: AppContext):
    """(tiles html, json) of the newest file in ``data_dir/eval``."""
    found = sorted((ctx.data_dir / "eval").glob("*.json"))
    if not found:
        hint = "Run `reid evaluate` with ground truth, then reload."
        return C.empty_state("No evaluation yet", hint), None
    data = json.loads(found[-1].read_text())
    return eval_tiles(data) or C.empty_state(found[-1].name), data


def build(ctx: AppContext) -> gr.Tab:
    with gr.Tab("Metrics", id="metrics") as tab:
        run = gr.Dropdown([], label="Run")
        kpis = gr.HTML()
        mem = gr.HTML()
        stages = gr.Dataframe(label="Time per frame by stage", interactive=False, visible=False)
        gr.Markdown("### Sightings per person")
        frag = gr.HTML()
        gr.Markdown("### Evaluation against ground truth")
        ev_tiles = gr.HTML()
        with gr.Accordion("Full evaluation result", open=False):
            ev_json = gr.JSON(show_label=False)

    gr.on([tab.select, ctx.ui["demo"].load], lambda: run_choices(ctx), None, run)
    run.change(lambda r: run_view(ctx, r), run, [kpis, stages, mem, frag])
    gr.on([tab.select, ctx.ui["demo"].load], lambda: eval_results(ctx), None, [ev_tiles, ev_json])
    return tab
