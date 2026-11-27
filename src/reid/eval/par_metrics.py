"""Pedestrian attribute recognition metrics with unknown-label masks, and a voting curve.

Targets are ``int8`` with 1 / 0 / -1 (unknown, ignored everywhere). Predictions are
probabilities; a label is predicted positive when ``prob > threshold`` (0.5 by default, the
standard protocol).

* Label-based: per label positive recall, negative recall and their mean (the mA term); mA is the
  mean over labels that have at least one positive and one negative among the known targets.
* ``label_auc``: per-label ROC AUC (threshold free), NaN without both classes.
* Instance-based (Li et al. 2015; Rethinking_of_PAR convention): per image, over its known labels,
  ``Acc = |Y & P| / |Y | P|``, ``Prec = |Y & P| / |P|``, ``Rec = |Y & P| / |Y|``; images with no
  known positive and no predicted positive are skipped. ``F1 = 2 Prec Rec / (Prec + Rec)`` of the
  averaged precision and recall (the convention of the published tables); ``f1_per_image`` is the
  mean of the per-image F1.
"""

from __future__ import annotations

import numpy as np

UNKNOWN = -1


def label_stats(
    probs: np.ndarray, targets: np.ndarray, threshold: float | np.ndarray = 0.5
) -> dict[str, np.ndarray]:
    """Per-label masked counts and recalls; ``ma_term`` is NaN where undefined."""
    pred = np.asarray(probs) > threshold
    t = np.asarray(targets)
    pos, neg = t == 1, t == 0
    n_pos, n_neg = pos.sum(0), neg.sum(0)
    tp, tn = (pred & pos).sum(0), (~pred & neg).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pr = np.where(n_pos > 0, tp / np.maximum(n_pos, 1), np.nan)
        nr = np.where(n_neg > 0, tn / np.maximum(n_neg, 1), np.nan)
    ma = np.where((n_pos > 0) & (n_neg > 0), (pr + nr) / 2, np.nan)
    return {"n_pos": n_pos, "n_neg": n_neg, "pos_recall": pr, "neg_recall": nr, "ma_term": ma}


def label_auc(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-label ROC AUC over the known targets (rank statistic); NaN without both classes."""
    from scipy.stats import rankdata

    p, t = np.asarray(probs), np.asarray(targets)
    out = np.full(p.shape[1], np.nan)
    for j in range(p.shape[1]):
        known = t[:, j] != UNKNOWN
        y = t[known, j] == 1
        n_pos, n_neg = int(y.sum()), int((~y).sum())
        if n_pos and n_neg:
            ranks = rankdata(p[known, j])
            out[j] = (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return out


def mean_accuracy(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float | np.ndarray = 0.5,
    cols: np.ndarray | list[int] | None = None,
) -> float:
    """mA over ``cols`` (default all labels); NaN when no label qualifies."""
    c = slice(None) if cols is None else np.asarray(cols)
    ma = label_stats(np.asarray(probs)[:, c], np.asarray(targets)[:, c], _thr(threshold, c))[
        "ma_term"
    ]
    return float(np.nanmean(ma)) if np.isfinite(ma).any() else float("nan")


def _thr(threshold: float | np.ndarray, cols: slice | np.ndarray) -> float | np.ndarray:
    return threshold if np.isscalar(threshold) else np.asarray(threshold)[cols]


def instance_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float | np.ndarray = 0.5,
    cols: np.ndarray | list[int] | None = None,
) -> dict[str, float]:
    """Instance-based Acc / Prec / Rec / F1 over the known labels of each image."""
    c = slice(None) if cols is None else np.asarray(cols)
    t = np.asarray(targets)[:, c]
    known = t != UNKNOWN
    pred = (np.asarray(probs)[:, c] > _thr(threshold, c)) & known
    gt = t == 1
    tp = (pred & gt).sum(1).astype(np.float64)
    n_pred, n_gt = pred.sum(1), gt.sum(1)
    keep = (n_pred + n_gt) > 0
    if not keep.any():
        return {k: float("nan") for k in ("acc", "prec", "rec", "f1", "f1_per_image")}
    tp, n_pred, n_gt = tp[keep], n_pred[keep], n_gt[keep]
    acc = tp / (n_pred + n_gt - tp)
    prec = np.where(n_pred > 0, tp / np.maximum(n_pred, 1), 0.0)
    rec = np.where(n_gt > 0, tp / np.maximum(n_gt, 1), 0.0)
    p, r = float(prec.mean()), float(rec.mean())
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {
        "acc": float(acc.mean()),
        "prec": p,
        "rec": r,
        "f1": f1,
        "f1_per_image": float((2 * tp / (n_pred + n_gt)).mean()),
    }


def voting_curve(
    probs: np.ndarray,
    targets: np.ndarray,
    persons: np.ndarray | list[str],
    ks: tuple[int, ...] = (1, 2, 4, 8),
    draws: int = 5,
    seed: int = 0,
    threshold: float | np.ndarray = 0.5,
    cols: np.ndarray | list[int] | None = None,
    min_persons: int = 30,
) -> list[dict[str, float]]:
    """mA when the probabilities of ``k`` random images of one person are averaged.

    Only persons with at least ``k`` images are used (their number is reported). ``ma_k1`` is
    the single-image mA on the same persons (random image), so the gain does not mix with the
    change of the person set. The largest ``k`` with ``min_persons`` persons is added to ``ks``.
    Targets are taken from the person's first image (PETA labels are per person).
    """
    probs, targets = np.asarray(probs), np.asarray(targets)
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(np.asarray(persons).tolist()):
        groups.setdefault(p, []).append(i)
    sizes = np.array(sorted(len(v) for v in groups.values()))
    feasible = [
        int(k) for k in range(1, int(sizes.max(initial=0)) + 1) if (sizes >= k).sum() >= min_persons
    ]
    all_ks = sorted({k for k in ks} | ({max(feasible)} if feasible else set()))
    rng = np.random.default_rng(seed)
    out = []
    for k in all_ks:
        members = [v for v in groups.values() if len(v) >= k]
        if len(members) < 2:
            continue
        tgt = np.stack([targets[v[0]] for v in members])
        mas, base = [], []
        for _ in range(draws):
            avg = np.stack([probs[rng.choice(v, k, replace=False)].mean(0) for v in members])
            one = np.stack([probs[rng.choice(v)] for v in members])
            mas.append(mean_accuracy(avg, tgt, threshold, cols))
            base.append(mean_accuracy(one, tgt, threshold, cols))
        out.append(
            {
                "k": k,
                "n_persons": len(members),
                "ma": float(np.nanmean(mas)),
                "ma_std": float(np.nanstd(mas)),
                "ma_k1": float(np.nanmean(base)),
            }
        )
    return out
