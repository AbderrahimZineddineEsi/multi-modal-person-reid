"""Zero-shot evaluation of the attribute head on a dataset it never saw (Market-1501).

Evaluation only: nothing is trained or tuned here. Threshold 0.5 is the headline (as in
:mod:`reid.attributes.report`); the validation-tuned thresholds stored in the head are an extra.
Labels of an external dataset are per identity, so the independent sample is the identity, not
the image: the label filter (``min_pos``) counts identities, and the identity-level result
averages the probabilities of up to ``k`` random crops of each identity (the pipeline's tracklet
vote). Identities with fewer than ``k`` crops use all of them.
"""

from __future__ import annotations

import warnings

import numpy as np

from reid.attributes.datasets import group_by_person
from reid.attributes.head import HeadParams
from reid.attributes.schema import LABELS, N_LABELS
from reid.eval.par_metrics import (
    instance_metrics,
    label_auc,
    label_stats,
    mean_accuracy,
    voting_curve,
)

MIN_POS = 30
VOTE_K = 16
REPEATS = 20
VOTE_KS = (1, 2, 4, 8, 16)
INSTANCE_KEYS = ("acc", "prec", "rec", "f1")


def _nanmean(a: np.ndarray) -> np.ndarray:
    """Mean over axis 0 ignoring NaN; all-NaN columns (label without both classes) stay NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, 0)


def _ms(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, float)
    return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr))}


def _id_targets(targets: np.ndarray, groups: dict[str, list[int]]) -> np.ndarray:
    """One target row per identity (labels are per identity: the first image's row)."""
    return np.stack([targets[v[0]] for v in groups.values()])


def vote(probs: np.ndarray, groups: dict[str, list[int]], k: int, rng: np.random.Generator):
    """``(n_ids, labels)`` mean probabilities of up to ``k`` random crops of each identity."""
    return np.stack(
        [probs[rng.choice(v, min(k, len(v)), replace=False)].mean(0) for v in groups.values()]
    )


def label_split(
    head: HeadParams, cols: np.ndarray, n_pos_id: np.ndarray, n_neg_id: np.ndarray, min_pos: int
) -> tuple[np.ndarray, dict[str, str]]:
    """Labels used for mA, and the reason for every excluded mapped label."""
    used, why = [], {}
    for j in cols:
        if not head.trained_labels[j]:
            why[LABELS[j]] = "not trained (too few positives in training)"
        elif min(n_pos_id[j], n_neg_id[j]) < min_pos:
            why[LABELS[j]] = (
                f"{n_pos_id[j]} positive / {n_neg_id[j]} negative identities (< {min_pos})"
            )
        else:
            used.append(j)
    return np.array(used, int), why


def _per_label(st: dict, tuned: dict, auc: np.ndarray, j: int, n_id: tuple[int, int]) -> dict:
    return {
        "n_pos_images": int(st["n_pos"][j]), "n_neg_images": int(st["n_neg"][j]),
        "n_pos_ids": n_id[0], "n_neg_ids": n_id[1],
        "pos_recall": float(st["pos_recall"][j]), "neg_recall": float(st["neg_recall"][j]),
        "ma": float(st["ma_term"][j]), "ma_tuned": float(tuned["ma_term"][j]),
        "auc": float(auc[j]),
    }  # fmt: skip


def consistency(
    probs: np.ndarray, targets: np.ndarray, groups: dict[str, list[int]], cols: np.ndarray
) -> dict[str, dict[str, float]]:
    """How often the crops of one identity disagree (0.5 threshold), per label.

    ``pos_rate``: mean over positive identities of the share of crops predicted positive;
    ``neg_rate``: same over negative identities (false positive rate per crop); ``pos_mixed`` /
    ``neg_mixed``: share of identities whose crops are neither all positive nor all negative.
    """
    pred = probs > 0.5
    rows = []
    for v in groups.values():
        rows.append(pred[v].mean(0))
    rate = np.stack(rows)
    tid = _id_targets(targets, groups)
    out = {}
    for j in cols:
        pos, neg = tid[:, j] == 1, tid[:, j] == 0
        mixed = (rate[:, j] > 0) & (rate[:, j] < 1)
        out[LABELS[j]] = {
            "pos_rate": float(rate[pos, j].mean()) if pos.any() else float("nan"),
            "neg_rate": float(rate[neg, j].mean()) if neg.any() else float("nan"),
            "pos_mixed": float(mixed[pos].mean()) if pos.any() else float("nan"),
            "neg_mixed": float(mixed[neg].mean()) if neg.any() else float("nan"),
        }
    return out


def bootstrap_ma(
    probs: np.ndarray,
    targets: np.ndarray,
    groups: dict[str, list[int]],
    used: np.ndarray,
    *,
    n_boot: int = 500,
    seed: int = 0,
    per_identity: bool = False,
) -> dict[str, float]:
    """95 % interval of mA when the *identities* are resampled with replacement.

    Labels are per identity, so identities (not images) are the independent units. With
    ``per_identity`` each identity contributes one row (``probs`` / ``targets`` are then the
    voted probabilities and identity targets); otherwise all its images, as in the image-level mA.
    """
    members = list(groups.values()) if not per_identity else [[i] for i in range(len(groups))]
    pred, t = (np.asarray(probs) > 0.5)[:, used], np.asarray(targets)[:, used]
    cnt = np.stack(
        [
            np.stack(
                [(t[v] == 1).sum(0), (pred[v] & (t[v] == 1)).sum(0)]
                + [(t[v] == 0).sum(0), (~pred[v] & (t[v] == 0)).sum(0)]
            )
            for v in members
        ]
    )  # (ids, [pos, tp, neg, tn], used)
    rng = np.random.default_rng(seed)
    mas = []
    for _ in range(n_boot):
        c = rng.multinomial(len(members), np.full(len(members), 1 / len(members))) @ cnt.reshape(
            len(members), -1
        )
        pos, tp, neg, tn = c.reshape(4, -1)
        with np.errstate(invalid="ignore", divide="ignore"):
            term = np.where((pos > 0) & (neg > 0), (tp / pos + tn / neg) / 2, np.nan)
        mas.append(np.nanmean(term))
    lo, hi = np.percentile(mas, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi), "n_boot": n_boot}


def identity_level(
    head: HeadParams, probs, targets, groups, used, cols, k: int, repeats: int, seed: int
) -> dict:
    """Identity-level metrics over ``repeats`` random draws of ``k`` crops per identity."""
    tid = _id_targets(targets, groups)
    rng = np.random.default_rng(seed)
    runs: dict[str, list] = {key: [] for key in ("ma", "ma_tuned", "ma_all", *INSTANCE_KEYS)}
    runs["inst_tuned_f1"], terms, aucs = [], [], []
    for _ in range(repeats):
        p = vote(probs, groups, k, rng)
        st = label_stats(p, tid, 0.5)
        terms.append(np.stack([st["pos_recall"], st["neg_recall"], st["ma_term"]]))
        aucs.append(label_auc(p, tid))
        runs["ma"].append(mean_accuracy(p, tid, 0.5, used))
        runs["ma_tuned"].append(mean_accuracy(p, tid, head.thresholds, used))
        runs["ma_all"].append(mean_accuracy(p, tid, 0.5, cols))
        inst = instance_metrics(p, tid, 0.5, used)
        for key in INSTANCE_KEYS:
            runs[key].append(inst[key])
        runs["inst_tuned_f1"].append(instance_metrics(p, tid, head.thresholds, used)["f1"])
    boot = bootstrap_ma(
        vote(probs, groups, k, np.random.default_rng(seed)),
        tid,
        groups,
        used,
        seed=seed,
        per_identity=True,
    )
    mean = np.full((3, N_LABELS), np.nan)
    mean[:, cols] = _nanmean(np.stack(terms)[:, :, cols])
    auc = np.full(N_LABELS, np.nan)
    auc[cols] = _nanmean(np.stack(aucs)[:, cols])
    return {
        "k": k, "repeats": repeats, "n_ids": len(groups), "ma_ci95_identities": boot,
        "summary": {key: _ms(v) for key, v in runs.items()},
        "labels": {
            LABELS[j]: {"pos_recall": float(mean[0, j]), "neg_recall": float(mean[1, j]),
                        "ma": float(mean[2, j]), "auc": float(auc[j])}
            for j in cols
        },
    }  # fmt: skip


def evaluate_external(
    head: HeadParams,
    probs: np.ndarray,
    targets: np.ndarray,
    persons: list[str],
    *,
    min_pos: int = MIN_POS,
    k: int = VOTE_K,
    repeats: int = REPEATS,
    seed: int = 0,
) -> dict:
    """Image-level, identity-level and voting-curve metrics of ``head`` on one external split.

    ``probs`` are the head's probabilities per image (no flips), ``targets`` the unified
    targets, ``persons`` the identity of each image.
    """
    probs, targets = np.asarray(probs), np.asarray(targets)
    groups = group_by_person(list(persons))
    tid = _id_targets(targets, groups)
    cols = np.flatnonzero((targets != -1).any(0))
    n_pos_id, n_neg_id = (tid == 1).sum(0), (tid == 0).sum(0)
    used, excluded = label_split(head, cols, n_pos_id, n_neg_id, min_pos)
    st, tuned = label_stats(probs, targets, 0.5), label_stats(probs, targets, head.thresholds)
    auc = label_auc(probs, targets)
    inst, inst_tuned = (
        instance_metrics(probs, targets, 0.5, used),
        instance_metrics(probs, targets, head.thresholds, used),
    )
    return {
        "n_images": int(len(targets)), "n_identities": len(groups), "min_pos": min_pos,
        "mapped_labels": [LABELS[j] for j in cols], "used_labels": [LABELS[j] for j in used],
        "excluded": excluded,
        "image": {
            "ma": mean_accuracy(probs, targets, 0.5, used),
            "ma_tuned": mean_accuracy(probs, targets, head.thresholds, used),
            "ma_all_mapped": mean_accuracy(probs, targets, 0.5, cols),
            "ma_ci95_identities": bootstrap_ma(probs, targets, groups, used, seed=seed),
            **inst, "instance_tuned": inst_tuned,
        },
        "labels": {
            LABELS[j]: _per_label(st, tuned, auc, j, (int(n_pos_id[j]), int(n_neg_id[j])))
            for j in cols
        },
        "identity": identity_level(head, probs, targets, groups, used, cols, k, repeats, seed),
        "voting": voting_curve(
            probs, targets, list(persons), ks=VOTE_KS, draws=repeats, seed=seed, cols=used
        ),
        "consistency": consistency(probs, targets, groups, cols),
    }  # fmt: skip


def in_domain(metrics: dict, labels: list[str], min_pos: int = MIN_POS) -> dict[str, dict]:
    """In-domain numbers of ``labels`` from an existing ``metrics_<tag>.json`` (not recomputed).

    Per reference dataset: the per-label test mA (labels with >= ``min_pos`` test positives) and
    the mean over the labels that dataset has in common with ``labels``.
    """
    out = {}
    for name, res in metrics.get("datasets", {}).items():
        per = {
            lab: float(res["labels"][lab]["ma"])
            for lab in labels
            if lab in res["labels"]
            and res["labels"][lab]["n_pos"] >= min_pos
            and np.isfinite(res["labels"][lab]["ma"])
        }
        if per:
            out[name] = {"labels": per, "mean": float(np.mean(list(per.values())))}
    return out
