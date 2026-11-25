"""Score the benchmark environments of a workspace with the official AI City 2024 evaluator.

Collects every ``aicity24_<scene>_<preset>`` scene (packed or not), concatenates their
``predictions.txt`` and official ``gt_official.txt``, runs NVIDIA's evaluator overall and once
per scene, and writes
``official_<preset>.json`` plus a short ``RESULTS_<preset>.md`` into the workspace.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reid.eval.aicity_bench_io import read_benchmark
from reid.eval.aicity_official import (
    LEADERBOARD_2024,
    fetch_official_eval,
    run_official_eval,
    run_official_per_scene,
    write_scene_map,
)
from reid.eval.aicity_pack import RESULTS_DIR, scene_files_dir

log = logging.getLogger(__name__)

METRICS = ("HOTA", "DetA", "AssA", "LocA")


@dataclass
class SceneInput:
    """One scored scene: its files folder (``results/<env>`` or ``<env>/aicity``)."""

    scene: str
    name: str
    files_dir: Path
    cameras: list[int]


def collect_scenes(workspace: str | Path, preset: str) -> list[SceneInput]:
    """Scenes ``aicity24_*_<preset>`` with exported predictions, sorted by scene.

    Looks in ``results/`` (packed scenes) and in unpacked environment folders. Scenes without
    predictions (failed or unfinished) are skipped and said so.
    """
    ws = Path(workspace)
    pattern = f"aicity24_*_{preset}"
    names = {p.name for p in (ws / RESULTS_DIR).glob(pattern)} | {p.name for p in ws.glob(pattern)}
    out = []
    for name in sorted(names):
        d = scene_files_dir(ws, name)
        if d is None or not (d / "meta.json").exists():
            msg = f"{name}: no exported predictions (scene failed or not finished), not scored"
            log.warning(msg)
            print(msg, file=sys.stderr, flush=True)
            continue
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        cams = sorted(int(c) for c in m["camera_numbers"].values())
        out.append(SceneInput(str(m["scene"]), name, d, cams))
    return sorted(out, key=lambda s: s.scene)


def _concat(parts: list[Path], dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out)
            if p.stat().st_size and not _ends_with_newline(p):
                out.write(b"\n")
    return dst


def _ends_with_newline(p: Path) -> bool:
    with open(p, "rb") as f:
        f.seek(-1, 2)
        return f.read(1) == b"\n"


def score_workspace(
    workspace: str | Path,
    preset: str = "balanced",
    num_cores: int = 4,
    *,
    per_scene: bool = True,
    eval_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the official evaluator on every scored environment; returns and saves the results."""
    ws = Path(workspace)
    scenes = collect_scenes(ws, preset)
    if not scenes:
        raise FileNotFoundError(f"no aicity24_*_{preset} environment with predictions in {ws}")
    work = ws / ".official" / preset
    pred = _concat([s.files_dir / "predictions.txt" for s in scenes], work / "pred.txt")
    gt = _concat([s.files_dir / "gt_official.txt" for s in scenes], work / "gt.txt")
    names = {f"scene_{s.scene}": s.cameras for s in scenes}
    ev = Path(eval_dir) if eval_dir else fetch_official_eval(ws / ".official" / "evaluator")
    t0 = time.perf_counter()
    per: dict[str, dict[str, float]] = {}
    if per_scene or len(scenes) == 1:
        per = run_official_per_scene(ev, pred, gt, names, work, num_cores)
    overall = (
        next(iter(per.values()))
        if len(scenes) == 1
        else run_official_eval(
            ev, pred, gt, write_scene_map(names, work / "scenes.json"), num_cores
        )
    )
    bench = read_benchmark(ws).get("runs", {})
    result = {
        "preset": preset,
        "date": time.strftime("%Y-%m-%d"),
        "protocol": "official AI City 2024 Track 1 evaluator (HOTA, 3D floor location, "
        "zero-similarity distance 2.0 m); overall = mean over the scenes",
        "scenes": {
            s.scene: {"cameras": len(s.cameras), **per.get(f"scene_{s.scene}", {})} for s in scenes
        },
        "overall": overall,
        "evaluator_seconds": round(time.perf_counter() - t0, 1),
        "leaderboard_2024": LEADERBOARD_2024,
        "runs": {s.scene: bench.get(s.name, {}) for s in scenes},
    }
    (ws / f"official_{preset}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (ws / f"RESULTS_{preset}.md").write_text(results_markdown(result), encoding="utf-8")
    return result


def _fmt(v: Any, pct: bool = False, nd: int = 1) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.{nd}f}%" if pct else f"{v:.{nd}f}"


def results_markdown(r: dict[str, Any]) -> str:
    """Short Markdown report of a :func:`score_workspace` result."""
    runs = r.get("runs", {})
    any_run = next((v for v in runs.values() if v), {})
    lines = [
        f"# AI City 2024 Track 1 benchmark, preset `{r['preset']}`",
        "",
        f"Scored on {r['date']} with NVIDIA's official evaluator (HOTA on floor positions). "
        f"Frame stride {any_run.get('frame_stride', '?')}, processing at "
        f"{any_run.get('max_side', '?')} px long side, GPU: {any_run.get('gpu') or 'unknown'}, "
        f"commit: {any_run.get('commit') or 'unknown'}.",
        "",
        "## Official metrics (percent)",
        "",
        "| scene | cameras | HOTA | DetA | AssA | LocA |",
        "|---|---|---|---|---|---|",
    ]
    for scene, m in r["scenes"].items():
        lines.append(
            f"| {scene} | {m['cameras']} | " + " | ".join(_fmt(m.get(k)) for k in METRICS) + " |"
        )
    o = r["overall"]
    lines.append("| **overall** | | " + " | ".join(f"**{_fmt(o.get(k))}**" for k in METRICS) + " |")
    lines += [
        "",
        "## Our 2D metrics, speed (per scene)",
        "",
        "| scene | combined IDF1 | MOTA | identities / GT people | Rank-1 | fps | GPU memory |",
        "|---|---|---|---|---|---|---|",
    ]
    for scene, run in runs.items():
        m = run.get("metrics_2d") or {}
        gm = (run.get("gpu_memory") or {}).get("required_mb")
        lines.append(
            f"| {scene} | {_fmt(m.get('idf1'), True)} | {_fmt(m.get('mota'), True)} | "
            f"{m.get('identities', 'n/a')} / {m.get('gt_ids', 'n/a')} | "
            f"{_fmt((m.get('retrieval') or {}).get('rank1'), True)} | {_fmt(run.get('fps'))} | "
            f"{f'{gm / 1024:.2f} GB' if gm else 'n/a'} |"
        )
    lb = r["leaderboard_2024"]
    lines += [
        "",
        "## Context: the 2024 leaderboard",
        "",
        f"Published HOTA of the AI City Challenge 2024 Track 1 winners: 1st {lb['1st']}, "
        f"2nd {lb['2nd']}, 3rd {lb['3rd']} (team papers, CVPRW 2024; online entries received a "
        "10% bonus). Those numbers are on all 30 test scenes, ours are on the scenes listed "
        "above, so they are context and not a like-for-like comparison.",
        "",
    ]
    return "\n".join(lines)
