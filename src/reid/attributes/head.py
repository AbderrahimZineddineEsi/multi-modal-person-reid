"""Trained attribute head: ``head.npz`` round trip and pure-NumPy inference (no torch needed).

The head is ``Linear -> BatchNorm1d (eval) -> ReLU -> Linear -> sigmoid`` (dropout is a no-op at
inference). BatchNorm is stored unfolded (mean, var, weight, bias, eps). ``hparams`` and
``train_log`` are JSON strings inside the archive; everything loads with ``allow_pickle=False``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HEAD_FILE = "head.npz"
_ARRAYS = ("W1", "b1", "bn_mean", "bn_var", "bn_weight", "bn_bias", "W2", "b2")


@dataclass
class HeadParams:
    """Weights (torch layout: ``W1 (hidden, in)``, ``W2 (labels, hidden)``) and metadata."""

    W1: np.ndarray
    b1: np.ndarray
    bn_mean: np.ndarray
    bn_var: np.ndarray
    bn_weight: np.ndarray
    bn_bias: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    labels: list[str]
    groups: list[str]
    thresholds: np.ndarray  # (labels,) tuned on validation
    trained_labels: np.ndarray  # bool (labels,)
    body_model: str = ""
    neck_feat: str = ""
    bn_eps: float = 1e-5
    hparams: dict = field(default_factory=dict)
    train_log: list = field(default_factory=list)

    @property
    def in_dim(self) -> int:
        return int(self.W1.shape[1])


def head_file(path: str | Path) -> Path:
    """``path`` itself if it is a file, else ``path/head.npz``."""
    p = Path(path)
    return p if p.suffix == ".npz" else p / HEAD_FILE


def save_head(path: str | Path, head: HeadParams) -> Path:
    """Write ``head.npz``; returns its path."""
    p = head_file(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(getattr(head, k), np.float32) for k in _ARRAYS}
    np.savez(
        p,
        **arrays,
        bn_eps=np.float64(head.bn_eps),
        labels=np.array(head.labels, dtype=str),
        groups=np.array(head.groups, dtype=str),
        thresholds=np.asarray(head.thresholds, np.float32),
        trained_labels=np.asarray(head.trained_labels, bool),
        body_model=np.array(head.body_model, dtype=str),
        neck_feat=np.array(head.neck_feat, dtype=str),
        hparams=np.array(json.dumps(head.hparams), dtype=str),
        train_log=np.array(json.dumps(head.train_log), dtype=str),
    )
    return p


def load_head(path: str | Path) -> HeadParams:
    """Read a ``head.npz`` (file or its folder)."""
    with np.load(head_file(path), allow_pickle=False) as z:
        return HeadParams(
            **{k: z[k].astype(np.float32) for k in _ARRAYS},
            labels=[str(x) for x in z["labels"]],
            groups=[str(x) for x in z["groups"]],
            thresholds=z["thresholds"].astype(np.float32),
            trained_labels=z["trained_labels"].astype(bool),
            body_model=str(z["body_model"]),
            neck_feat=str(z["neck_feat"]),
            bn_eps=float(z["bn_eps"]),
            hparams=json.loads(str(z["hparams"])),
            train_log=json.loads(str(z["train_log"])),
        )


def predict(head: HeadParams, embs: np.ndarray) -> np.ndarray:
    """Probabilities ``(N, labels)`` float32 for embeddings ``(N, in_dim)`` (fp16 or fp32)."""
    x = np.asarray(embs, np.float32)
    h = x @ head.W1.T + head.b1
    h = (h - head.bn_mean) / np.sqrt(head.bn_var + head.bn_eps) * head.bn_weight + head.bn_bias
    logits = np.maximum(h, 0.0) @ head.W2.T + head.b2
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))).astype(np.float32)
