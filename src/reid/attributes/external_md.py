"""Markdown tables of an external (zero-shot) attribute evaluation, see :mod:`.external`."""

from __future__ import annotations

import numpy as np

from reid.attributes.external import INSTANCE_KEYS


def pct(v: float | None) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{100 * v:.1f}"


def pm(s: dict[str, float]) -> str:
    return f"{100 * s['mean']:.1f} ± {100 * s['std']:.1f}"


def same_label_comparison(res: dict, ref: dict[str, dict]) -> list[dict]:
    """Mean mA of the labels shared with each in-domain dataset: in-domain vs this dataset."""
    rows = []
    for name, r in ref.items():
        labs = [lab for lab in r["labels"] if lab in res["used_labels"]]
        if not labs:
            continue
        rows.append(
            {
                "dataset": name,
                "n_labels": len(labs),
                "in_domain": float(np.mean([r["labels"][lab] for lab in labs])),
                "image": float(np.mean([res["labels"][lab]["ma"] for lab in labs])),
                "identity": float(np.mean([res["identity"]["labels"][lab]["ma"] for lab in labs])),
            }
        )
    return rows


def headline(res: dict) -> list[str]:
    im, idt = res["image"], res["identity"]["summary"]
    rows = ["| level | mA | Acc | Prec | Rec | F1 | mA (val-tuned thr.) | mA (all mapped) |"]
    rows.append("|---|---|---|---|---|---|---|---|")
    rows.append(
        f"| image ({res['n_images']}) | {pct(im['ma'])} | "
        + " | ".join(pct(im[k]) for k in INSTANCE_KEYS)
        + f" | {pct(im['ma_tuned'])} | {pct(im['ma_all_mapped'])} |"
    )
    n = res["identity"]
    rows.append(
        f"| identity, {n['k']} crops ({n['n_ids']}, {n['repeats']} draws) | {pm(idt['ma'])} | "
        + " | ".join(pm(idt[k]) for k in INSTANCE_KEYS)
        + f" | {pm(idt['ma_tuned'])} | {pm(idt['ma_all'])} |"
    )
    ci_i, ci_d = im["ma_ci95_identities"], n["ma_ci95_identities"]
    rows += [
        "",
        "95 % interval of mA over resampled identities (the independent units; "
        f"{ci_i['n_boot']} resamples): image level {pct(ci_i['lo'])} to {pct(ci_i['hi'])}, "
        f"identity level {pct(ci_d['lo'])} to {pct(ci_d['hi'])}.",
    ]
    return rows


def per_label(res: dict, ref: dict[str, dict]) -> list[str]:
    names = list(ref)
    head = "| label | pos / neg ids | pos / neg images | image mA | pos rec | neg rec | AUC "
    head += "| identity mA | " + " | ".join(f"{n} mA" for n in names) + " | drop |"
    rows = [head, "|---|---|---|---|---|---|---|---|" + "---|" * (len(names) + 1)]
    for lab, m in res["labels"].items():
        if lab not in res["used_labels"]:
            continue
        idm = res["identity"]["labels"][lab]
        refs = [ref[n]["labels"].get(lab) for n in names]
        known = [v for v in refs if v is not None]
        drop = f"{100 * (np.mean(known) - m['ma']):+.1f}" if known else "-"
        rows.append(
            f"| {lab} | {m['n_pos_ids']} / {m['n_neg_ids']} | {m['n_pos_images']} / "
            f"{m['n_neg_images']} | {pct(m['ma'])} | {pct(m['pos_recall'])} | "
            f"{pct(m['neg_recall'])} | {pct(m['auc'])} | {pct(idm['ma'])} | "
            + " | ".join("-" if v is None else pct(v) for v in refs)
            + f" | {drop} |"
        )
    return rows


def voting(res: dict) -> list[str]:
    rows = [
        "| k crops | identities with >= k crops | mA (k crops) | mA (1 crop, same identities) |"
    ]
    rows.append("|---|---|---|---|")
    for r in res["voting"]:
        rows.append(
            f"| {r['k']} | {r['n_persons']} | {pct(r['ma'])} ± {100 * r['ma_std']:.1f} "
            f"| {pct(r['ma_k1'])} |"
        )
    return rows


def consistency(res: dict) -> list[str]:
    head = "| label | positive ids: crops predicted positive | positive ids with mixed crops "
    head += "| negative ids: crops predicted positive | negative ids with mixed crops |"
    rows = [head]
    rows.append("|---|---|---|---|---|")
    for lab in res["used_labels"]:
        c = res["consistency"][lab]
        rows.append(
            f"| {lab} | {pct(c['pos_rate'])} | {pct(c['pos_mixed'])} | {pct(c['neg_rate'])} "
            f"| {pct(c['neg_mixed'])} |"
        )
    return rows


def render(res: dict, ref: dict[str, dict], meta: dict) -> str:
    """The ``RESULTS_<dataset>.md`` text."""
    md = [
        f"# Zero-shot evaluation on {meta['dataset']}", "",
        f"Head `{meta['head']}` (body `{meta['body_model']}`), threshold 0.5. "
        f"{res['n_images']} images of {res['n_identities']} identities; labels are per identity. "
        f"mA over {len(res['used_labels'])} of {len(res['mapped_labels'])} mapped labels "
        f"(labels with fewer than {res['min_pos']} positive or negative identities are excluded).",
        "", "## Headline", "", *headline(res), "",
        "Identity level: probabilities of up to k random crops per identity are averaged "
        "(identities with fewer crops use all of them); mean ± std over the draws.", "",
        "## Same labels, in-domain vs zero-shot (mean per-label mA, %)", "",
        "| in-domain dataset | labels in common | in-domain | zero-shot, image | "
        "zero-shot, identity |", "|---|---|---|---|---|",
    ]  # fmt: skip
    for r in same_label_comparison(res, ref):
        md.append(
            f"| {r['dataset']} | {r['n_labels']} | {pct(r['in_domain'])} | {pct(r['image'])} "
            f"| {pct(r['identity'])} |"
        )
    md += ["", "## Per label", "", *per_label(res, ref), "", "drop = mean in-domain mA minus "
           "image-level zero-shot mA (points; positive = worse zero-shot).", ""]  # fmt: skip
    md += ["## Excluded mapped labels", ""]
    md += [f"- `{k}`: {v}" for k, v in res["excluded"].items()] or ["- none"]
    md += ["", "## Voting curve", "", *voting(res), ""]
    md += ["## Agreement between crops of one identity (threshold 0.5)", ""]
    md += [*consistency(res), ""]
    return "\n".join(md) + "\n"
