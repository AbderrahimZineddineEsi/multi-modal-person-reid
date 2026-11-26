"""PNG plots of the attribute report: training curves and the voting curve (matplotlib, Agg)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_training(log: list[dict], path: Path) -> Path | None:
    """Loss and validation mA per epoch; None when there is no log."""
    if not log:
        return None
    ep = [r["epoch"] for r in log]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    a.plot(ep, [r["loss"] for r in log])
    a.set(xlabel="epoch", ylabel="train loss", title="Masked weighted BCE")
    for name in sorted({k for r in log for k in r["val_ma"]}):
        b.plot(ep, [100 * r["val_ma"].get(name, float("nan")) for r in log], label=name)
    b.plot(ep, [100 * r["val_ma_mean"] for r in log], "k--", label="mean (selection)")
    b.set(xlabel="epoch", ylabel="validation mA (%)", title="Validation mA")
    b.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_voting(curve: list[dict], path: Path) -> Path | None:
    """mA against the number of averaged images per person; None for an empty curve."""
    if not curve:
        return None
    k = [r["k"] for r in curve]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.errorbar(k, [100 * r["ma"] for r in curve], [100 * r["ma_std"] for r in curve], marker="o")
    ax.plot(k, [100 * r["ma_k1"] for r in curve], "--", color="grey", label="1 image, same persons")
    ax.set(xscale="log", xlabel="images averaged per person (k)", ylabel="PETA test mA (%)")
    ax.set_xticks(k, [str(x) for x in k])
    ax.set_title("Voting over k images of the same person")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
