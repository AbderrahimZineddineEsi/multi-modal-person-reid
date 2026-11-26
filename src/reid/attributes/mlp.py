"""The torch MLP head used for training (inference lives in :mod:`reid.attributes.head`)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from reid.attributes.head import HeadParams


class MLPHead(nn.Module):
    """``Linear(in, hidden) -> BatchNorm1d -> ReLU -> Dropout(p) -> Linear(hidden, labels)``.

    Outputs logits; probabilities are ``sigmoid(logits)``.
    """

    def __init__(self, in_dim: int, n_labels: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, n_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(torch.relu(self.bn(self.fc1(x)))))

    def to_params(self, **meta) -> HeadParams:  # noqa: ANN003
        """Copy the weights into a NumPy :class:`HeadParams` (``meta`` fills the other fields)."""

        def n(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy().astype(np.float32)

        return HeadParams(
            W1=n(self.fc1.weight),
            b1=n(self.fc1.bias),
            bn_mean=n(self.bn.running_mean),
            bn_var=n(self.bn.running_var),
            bn_weight=n(self.bn.weight),
            bn_bias=n(self.bn.bias),
            W2=n(self.fc2.weight),
            b2=n(self.fc2.bias),
            bn_eps=float(self.bn.eps),
            **meta,
        )
