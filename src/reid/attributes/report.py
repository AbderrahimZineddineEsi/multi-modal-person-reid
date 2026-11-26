"""``reid attributes report``: test-set metrics, per-label table and the voting curve.

PA-100K is scored on its 26 original attributes (comparable with published numbers); PETA on the
labels it knows, with our person-disjoint split (indicative comparison only). Reported numbers
use threshold 0.5; thresholds tuned on validation are shown as an extra. Labels that were not
trained (too few positives) are listed and excluded from mA. The test split is read here only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from reid.attributes import store
from reid.attributes.head import HeadParams, load_head, predict
from reid.attributes.report_plots import plot_training, plot_voting
from reid.attributes.schema import LABELS, PA100K_COLUMNS
from reid.eval.par_metrics import instance_metrics, label_stats, mean_accuracy, voting_curve

log = logging.getLogger(__name__)
UI_MIN_MA, UI_MIN_POS = 0.70, 30
REFERENCES = {
    "pa100k": "Rethinking_of_PAR ResNet-50 on PA-100K: mA 80.21, F1 87.40",
    "peta": "Rethinking_of_PAR ResNet-50 on PETAzs: mA 71.43, F1 72.04 (other split, attributes)",
}


def _pct(v: float) -> str:
    return "n/a" if not np.isfinite(v) else f"{100 * v:.2f}"


def dataset_metrics(head: HeadParams, emb_dir: Path, name: str) -> dict | None:
    """Metrics of one dataset's test split, or None when it was not embedded."""
    if "test" not in store.available_splits(emb_dir, name):
        return None
    sp = store.load_split(emb_dir, name, "test", include_flips=False)
    probs, t = predict(head, sp.x), sp.targets
    if name == "pa100k":
        cols = np.array(PA100K_COLUMNS)
    else:
        cols = np.flatnonzero((t != -1).any(0))
    used = cols[head.trained_labels[cols]]
    st = label_stats(probs, t, 0.5)
    tuned = label_stats(probs, t, head.thresholds)
    out = {
        "n_images": int(len(t)),
        "n_labels": int(len(cols)),
        "labels_untrained": [LABELS[j] for j in cols if not head.trained_labels[j]],
        "ma": mean_accuracy(probs, t, 0.5, used),
        "ma_tuned_thresholds": mean_accuracy(probs, t, head.thresholds, used),
        **instance_metrics(probs, t, 0.5, used),
        "instance_tuned_thresholds": instance_metrics(probs, t, head.thresholds, used),
        "labels": {
            LABELS[j]: {
                "n_pos": int(st["n_pos"][j]),
                "n_neg": int(st["n_neg"][j]),
                "pos_recall": float(st["pos_recall"][j]),
                "neg_recall": float(st["neg_recall"][j]),
                "ma": float(st["ma_term"][j]),
                "ma_tuned": float(tuned["ma_term"][j]),
            }
            for j in cols
        },
    }
    if name == "peta" and len(sp.persons):
        out["voting"] = voting_curve(probs, t, sp.persons, cols=used)
    return out


def ui_candidates(per_ds: dict[str, dict]) -> set[str]:
    """Labels with mA >= 70 % and >= 30 test positives in every dataset that has that many."""
    cand = {}
    for res in per_ds.values():
        for lab, m in res["labels"].items():
            if m["n_pos"] >= UI_MIN_POS and m["n_neg"] > 0:
                cand[lab] = cand.get(lab, True) and m["ma"] >= UI_MIN_MA
    return {lab for lab, ok in cand.items() if ok}


def _label_table(head: HeadParams, per_ds: dict[str, dict], ui: set[str]) -> list[str]:
    names = list(per_ds)
    rows = ["| label | group | trained | " + " | ".join(f"{n} pos / mA" for n in names) + " | UI |"]
    rows.append("|---|---|---|" + "---|" * len(names) + "---|")
    for j, lab in enumerate(LABELS):
        cells = []
        for n in names:
            m = per_ds[n]["labels"].get(lab)
            cells.append("-" if m is None else f"{m['n_pos']} / {_pct(m['ma'])}")
        if all(c == "-" for c in cells):
            continue
        tr = "yes" if head.trained_labels[j] else "NO"
        rows.append(
            f"| {lab} | {head.groups[j]} | {tr} | "
            + " | ".join(cells)
            + f" | {'x' if lab in ui else ''} |"
        )
    return rows


def _summary(per_ds: dict[str, dict]) -> list[str]:
    rows = ["| dataset | test images | labels | mA | Acc | Prec | Rec | F1 | mA (tuned thr.) |"]
    rows.append("|---|---|---|---|---|---|---|---|---|")
    for n, r in per_ds.items():
        rows.append(
            f"| {n} | {r['n_images']} | {r['n_labels'] - len(r['labels_untrained'])} "
            f"| {_pct(r['ma'])} | {_pct(r['acc'])} | {_pct(r['prec'])} | {_pct(r['rec'])} "
            f"| {_pct(r['f1'])} | {_pct(r['ma_tuned_thresholds'])} |"
        )
    return rows


def _voting_md(curve: list[dict]) -> list[str]:
    rows = ["| k | persons | mA (k images) | mA (1 image, same persons) |", "|---|---|---|---|"]
    rows += [
        f"| {r['k']} | {r['n_persons']} | {_pct(r['ma'])} +- {100 * r['ma_std']:.2f} "
        f"| {_pct(r['ma_k1'])} |"
        for r in curve
    ]
    return rows


def run_report(
    emb_dir: str | Path, head_path: str | Path, out: str | Path, tag: str = "default"
) -> Path:
    """Write ``RESULTS_attributes_<tag>.md``, ``metrics_<tag>.json``, PNGs; returns the .md."""
    emb_dir, out = Path(emb_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    head = load_head(head_path)
    per_ds = {}
    for name in ("pa100k", "peta"):
        res = dataset_metrics(head, emb_dir, name)
        if res is not None:
            per_ds[name] = res
    if not per_ds:
        raise FileNotFoundError(f"no test embeddings in {emb_dir}")
    ui = ui_candidates(per_ds)
    voting = per_ds.get("peta", {}).get("voting", [])
    train_png = plot_training(head.train_log, out / f"training_{tag}.png")
    vote_png = plot_voting(voting, out / f"voting_{tag}.png")
    (out / f"metrics_{tag}.json").write_text(
        json.dumps(
            {"tag": tag, "ui_candidates": sorted(ui), "datasets": per_ds, "hparams": head.hparams},
            indent=1,
        )
    )
    md = [
        f"# Attribute head: results ({tag})", "",
        f"Body model `{head.body_model}` (neck `{head.neck_feat}`), "
        f"hidden {head.hparams.get('hidden')}, "
        f"best validation mA {_pct(head.hparams.get('best_val_ma', float('nan')))}. "
        "Threshold 0.5; "
        "mA over trained labels with both classes in the test split.", "",
        "## Summary", "", *_summary(per_ds), "",
    ]  # fmt: skip
    for n, r in per_ds.items():
        md.append(
            f"- {n}: {REFERENCES[n]}. "
            f"Not trained (too few positives): {r['labels_untrained'] or 'none'}."
        )
    md += [
        "",
        "PETA uses our person-disjoint 50/10/40 % split (seed 0), so it is indicative only.",
        "",
    ]
    md += ["## Per label (test positives / mA %)", "", *_label_table(head, per_ds, ui), ""]
    md += [
        "UI candidate: mA >= 70 % in every dataset that has >= 30 test positives of the label.",
        "",
    ]
    if voting:
        md += ["## Voting over images of one person (PETA test)", "", *_voting_md(voting), ""]
        md += ["A tracklet's samples are more correlated than random images of a person, so this "
               "is an upper bound on the gain inside the pipeline.", ""]  # fmt: skip
    for png in (train_png, vote_png):
        if png:
            md.append(f"![{png.stem}]({png.name})")
    path = out / f"RESULTS_attributes_{tag}.md"
    path.write_text("\n".join(md) + "\n")
    return path
