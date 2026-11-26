"""Train the attribute head on stored embeddings: masked, weighted BCE (frozen backbone).

Loss: per label ``j`` with positive ratio ``r_j`` over the known training targets, positives are
weighted ``exp(1 - r_j)`` and negatives ``exp(r_j)`` (``ratio2weight`` of Rethinking_of_PAR, Tan et
al. AAAI 2020 lineage); unknown targets (-1) are masked and the sum is divided by the number of
known targets. Labels with fewer than ``min_pos`` training positives are not trained (masked out
entirely). Early stopping, the model choice and the HPO objective use the **validation score**:
the mean over the datasets of the validation mA at threshold 0.5, computed over the trained
labels **excluding the three ``view_*`` labels**, so that runs with and without viewpoint
training are compared on the same label set (step 1 included the view labels). The test split is
never read here. Per-label thresholds maximise balanced accuracy on validation.
"""

from __future__ import annotations

import copy
import logging
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

from reid.attributes import store
from reid.attributes.head import HeadParams, save_head
from reid.attributes.mlp import MLPHead
from reid.attributes.schema import GROUPS, LABELS, N_LABELS
from reid.eval.par_metrics import mean_accuracy

log = logging.getLogger(__name__)
DATASETS = ("pa100k", "peta")
VIEW_COLS = np.array([j for j, g in enumerate(GROUPS) if g == "view"])
WEIGHTINGS = ("ratio", "none")


@dataclass
class TrainConfig:
    """Hyperparameters (all are CLI flags and are written into ``head.npz``)."""

    hidden: int = 512
    dropout: float = 0.3
    lr: float = 1e-3
    wd: float = 1e-4
    batch: int = 512
    epochs: int = 40
    patience: int = 8
    peta_weight: float = 1.0  # loss multiplier of PETA rows (1.0 = natural mixing)
    seed: int = 0
    min_pos: int = 10
    weighting: str = "ratio"  # "ratio" = exp(1 - r) / exp(r); "none" = all weights 1
    use_flips: bool = True  # False drops the flipped training rows
    train_view: bool = True  # False masks the view_* labels in training (viewpoint ablation)


@dataclass
class TrainData:
    """Rows (embedding, targets, dataset index) of train, and per-dataset val images."""

    x: np.ndarray  # float16 (M, D)
    t: np.ndarray  # int8 (M, 67)
    ds: np.ndarray  # int (M,) index into DATASETS
    val: dict[str, tuple[np.ndarray, np.ndarray]]  # dataset -> (x float16, targets)
    train_pos: np.ndarray  # positives per label over original (unflipped) train images
    flip: np.ndarray  # bool (M,) True for the horizontally flipped copies


def _clean(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Targets with all labels unknown for zero vectors (unreadable images)."""
    bad = np.linalg.norm(x.astype(np.float32), axis=1) < 1e-3
    if bad.any():
        t = t.copy()
        t[bad] = -1
    return t


def load_train_data(emb_dir: str | Path) -> TrainData:
    """Concatenate the train (original + flipped) rows of the datasets found; val separately."""
    xs, ts, dss, fls, val = [], [], [], [], {}
    pos = np.zeros(N_LABELS, np.int64)
    for di, name in enumerate(DATASETS):
        avail = store.available_splits(emb_dir, name)
        if "train" not in avail or "val" not in avail:
            log.warning("%s: train/val embeddings missing, dataset skipped", name)
            continue
        tr = store.load_split(emb_dir, name, "train")
        t = _clean(tr.x, tr.row_targets)
        xs.append(tr.x)
        ts.append(t)
        dss.append(np.full(len(tr.x), di))
        fls.append(tr.flip)
        pos += (t[~tr.flip] == 1).sum(0)
        va = store.load_split(emb_dir, name, "val", include_flips=False)
        val[name] = (va.x, _clean(va.x, va.row_targets))
    if not xs:
        raise FileNotFoundError(f"no train/val embeddings in {emb_dir}")
    x, t, ds, flip = (np.concatenate(a) for a in (xs, ts, dss, fls))
    return TrainData(x, t, ds, val, pos, flip)


def label_weights(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positive / negative weights ``exp(1 - r)`` / ``exp(r)`` from the known-target ratio."""
    known = (t != -1).sum(0)
    r = np.where(known > 0, (t == 1).sum(0) / np.maximum(known, 1), 0.0)
    return np.exp(1.0 - r).astype(np.float32), np.exp(r).astype(np.float32)


def masked_bce(
    logits: torch.Tensor,
    t: torch.Tensor,
    pos_w: torch.Tensor,
    neg_w: torch.Tensor,
    trained: torch.Tensor,
    sample_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """``sum(w * bce * mask) / sum(mask)``; ``t`` is int (-1 unknown), ``trained`` bool (labels)."""
    mask = ((t != -1) & trained).float()
    if sample_w is not None:
        mask = mask * sample_w[:, None]
    tf = (t == 1).float()
    w = torch.where(t == 1, pos_w, neg_w)
    bce = F.binary_cross_entropy_with_logits(logits, tf, reduction="none")
    return (w * bce * mask).sum() / mask.sum().clamp_min(1.0)


def tune_thresholds(
    probs: np.ndarray, targets: np.ndarray, trained: np.ndarray, grid: np.ndarray | None = None
) -> np.ndarray:
    """Per label threshold maximising balanced accuracy (ties: closest to 0.5); 0.5 if undefined."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    out = np.full(probs.shape[1], 0.5, np.float32)
    for j in np.flatnonzero(trained):
        y, p = targets[:, j], probs[:, j]
        pos, neg = p[y == 1], p[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        bal = ((pos[:, None] > grid).mean(0) + (neg[:, None] <= grid).mean(0)) / 2
        best = np.flatnonzero(bal >= bal.max() - 1e-12)
        out[j] = grid[best[np.argmin(np.abs(grid[best] - 0.5))]]
    return out


@torch.no_grad()
def _predict(model: MLPHead, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(x), 4096):
        xb = torch.from_numpy(x[s : s + 4096]).to(device).float()
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, N_LABELS), np.float32)


def _val_ma(model, data: TrainData, cols: np.ndarray, device) -> dict[str, float]:  # noqa: ANN001
    return {
        n: mean_accuracy(_predict(model, x, device), t, 0.5, cols) for n, (x, t) in data.val.items()
    }


def val_columns(trained: np.ndarray) -> np.ndarray:
    """Labels of the validation score: trained labels without the ``view_*`` ones."""
    return np.flatnonzero(trained & ~np.isin(np.arange(N_LABELS), VIEW_COLS))


def train_head(
    data: TrainData,
    cfg: TrainConfig,
    body_model: str = "",
    neck_feat: str = "",
    on_epoch: Callable[[int, float], None] | None = None,
) -> HeadParams:
    """Train with early stopping on the validation score and return the best model.

    The validation score is the mean over datasets of the validation mA (threshold 0.5) over the
    trained labels excluding ``view_*`` (see the module docstring). ``on_epoch(epoch, score)`` is
    called after every epoch; an exception raised from it (Optuna's ``TrialPruned``) stops the
    training and propagates. With ``cfg.train_view`` False the view labels are masked in training
    and flagged as not trained in the returned head.
    """
    if cfg.weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {cfg.weighting!r}")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)  # noqa: NPY002
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained = data.train_pos >= cfg.min_pos
    if not cfg.train_view:
        trained = trained & ~np.isin(np.arange(N_LABELS), VIEW_COLS)
    cols = val_columns(trained)
    keep = slice(None) if cfg.use_flips else ~data.flip
    xk, tk, dk = data.x[keep], data.t[keep], data.ds[keep]
    log.info("training %d of %d labels (%d rows)", int(trained.sum()), N_LABELS, len(xk))
    if cfg.weighting == "ratio":
        pw, nw = (torch.from_numpy(a).to(device) for a in label_weights(tk))
    else:
        pw = nw = torch.ones(N_LABELS, device=device)
    tr_mask = torch.from_numpy(trained).to(device)
    x = torch.from_numpy(xk).to(device)
    t = torch.from_numpy(tk.astype(np.int64)).to(device)
    sw = torch.where(
        torch.from_numpy(dk).to(device) == DATASETS.index("peta"), cfg.peta_weight, 1.0
    ).float()
    model = MLPHead(x.shape[1], N_LABELS, cfg.hidden, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    gen = torch.Generator().manual_seed(cfg.seed)
    best, best_state, best_epoch, best_vma, bad, history = -1.0, None, -1, {}, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        order = torch.randperm(len(x), generator=gen).to(device)
        total, nb = 0.0, 0
        for s in range(0, len(order), cfg.batch):
            idx = order[s : s + cfg.batch]
            if len(idx) < 2:
                continue
            loss = masked_bce(model(x[idx].float()), t[idx], pw, nw, tr_mask, sw[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, nb = total + loss.item(), nb + 1
        sched.step()
        vma = _val_ma(model, data, cols, device)
        score = float(np.nanmean(list(vma.values()))) if vma else float("nan")
        history.append(
            {"epoch": epoch, "loss": total / max(nb, 1), "val_ma": vma, "val_ma_mean": score}
        )
        log.info("epoch %d loss %.4f val mA %s mean %.4f", epoch, total / max(nb, 1), vma, score)
        if on_epoch is not None:
            on_epoch(epoch, score)
        if score > best:
            best, bad, best_state = score, 0, copy.deepcopy(model.state_dict())
            best_epoch, best_vma = epoch, vma
        else:
            bad += 1
            if bad >= cfg.patience:
                log.info("early stop at epoch %d (best %.4f)", epoch, best)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    vx = np.concatenate([v[0] for v in data.val.values()])
    vt = np.concatenate([v[1] for v in data.val.values()])
    thr = tune_thresholds(_predict(model, vx, device), vt, trained)
    hp = {
        **asdict(cfg), "in_dim": int(x.shape[1]), "best_val_ma": best, "best_epoch": best_epoch,
        "best_val_ma_by_dataset": best_vma, "device": str(device),
    }  # fmt: skip
    return model.to_params(
        labels=list(LABELS), groups=list(GROUPS), thresholds=thr, trained_labels=trained,
        body_model=body_model, neck_feat=neck_feat, hparams=hp, train_log=history,
    )  # fmt: skip


def run_train(emb_dir: str | Path, out: str | Path, cfg: TrainConfig) -> Path:
    """``reid attributes train``: load embeddings, train, write ``<out>/head.npz``."""
    data = load_train_data(emb_dir)
    idx = store.read_index(
        store.split_dir(emb_dir, "pa100k" if "pa100k" in data.val else "peta", "train")
    )
    head = train_head(data, cfg, (idx or {}).get("model", ""), (idx or {}).get("neck_feat", ""))
    path = save_head(out, head)
    under = [n for n, ok in zip(LABELS, head.trained_labels, strict=True) if not ok]
    log.info("wrote %s; labels not trained (< %d positives): %s", path, cfg.min_pos, under)
    return path
