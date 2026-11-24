"""Markdown table and JSON persistence of an evaluation result."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from reid.storage import files


def save_result(data_dir: str | Path, result: dict[str, Any]) -> Path:
    """Write ``result`` to ``data_dir/eval/<timestamp>.json`` and return the path."""
    name = time.strftime("%Y%m%d-%H%M%S", time.localtime(result["timestamp"]))
    path = files.eval_path(data_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return path


def _f(v: Any, pct: bool = False, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer() and not pct):
        return str(int(v))
    return f"{v * 100:.1f}%" if pct else f"{v:.{nd}f}"


def _mot_table(rows: dict[str, dict], title: str) -> list[str]:
    lines = [
        f"### {title}", "",
        "| camera | IDF1 | MOTA | IDSW | precision | recall | FP | FN | GT ids |",
        "|---|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    for cam, m in rows.items():
        lines.append(
            f"| {cam} | {_f(m['idf1'], True)} | {_f(m['mota'], True)} | {_f(m['id_switches'])} "
            f"| {_f(m['precision'], True)} | {_f(m['recall'], True)} | {_f(m['fp'])} "
            f"| {_f(m['fn'])} | {_f(m['num_unique_objects'])} |"
        )
    return [*lines, ""]


def to_markdown(r: dict[str, Any]) -> str:
    """Render an :func:`reid.eval.evaluate` result as markdown tables."""
    p, c, m, fr = r["protocol"], r["counts"], r["mtmc"], r["fragmentation"]
    out = [
        f"## Evaluation of run {r['run_id']} (preset: {r['preset'] or 'unknown'})", "",
        f"IoU >= {p['iou']}, GT boxes shorter than {p['min_gt_height']:g} px ignored, "
        f"frame stride {p['frame_stride']}. {c['gt_ids']} GT people, {c['identities']} system "
        f"identities, {c['tracklets']} tracklets ({c['unidentified_tracklets']} unidentified).", "",
        "### Cross-camera (CityFlow-style combined IDF1)", "",
        "| IDF1 | IDP | IDR | MOTA | IDSW | precision | recall |", "|---|---|---|---|---|---|---|",
        f"| {_f(m['idf1'], True)} | {_f(m['idp'], True)} | {_f(m['idr'], True)} "
        f"| {_f(m['mota'], True)} | {_f(m['id_switches'])} | {_f(m['precision'], True)} "
        f"| {_f(m['recall'], True)} |", "",
    ]  # fmt: skip
    out += _mot_table(r["per_camera"], "Per camera, hypothesis = system gid")
    out += _mot_table(r["per_camera_tracklet"], "Per camera, tracker only (hypothesis = tracklet)")
    out += [
        "### Fragmentation and purity", "",
        "| metric | value |", "|---|---|",
        f"| GT people with a gid | {fr['gt_with_match']} / {fr['n_gt_ids']} |",
        f"| GT people split over >1 gid | {fr['gt_fragmented']} |",
        f"| mean gids per GT person | {_f(fr['gids_per_gt_mean'], nd=2)} |",
        f"| gids covering >1 GT person | {fr['gid_impure']} / {fr['n_system_gids']} |",
        f"| gid purity (box-weighted) | {_f(fr['gid_purity'], True)} |",
        f"| GT completeness (box-weighted) | {_f(fr['gt_completeness'], True)} |",
        f"| matched boxes unidentified | {fr['matched_boxes_unidentified']} |", "",
    ]  # fmt: skip
    rt = r.get("retrieval")
    if rt:
        out += [
            "### Held-out retrieval (body)", "",
            "| queries | held out | Rank-1 | Rank-5 | mAP | answered | answered precision |",
            "|---|---|---|---|---|---|---|",
            f"| {rt['evaluated']} (+{rt['unretrievable']} unretrievable) "
            f"| {_f(rt['held_out_fraction'], True)} | {_f(rt['rank1'], True)} "
            f"| {_f(rt['rank5'], True)} | {_f(rt['mAP'], True)} "
            f"| {rt['search_answered']}/{rt['queries']} "
            f"| {_f(rt['search_answered_precision'], True)} |", "",
        ]  # fmt: skip
    return "\n".join(out)
