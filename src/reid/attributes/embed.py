"""Embed the attribute datasets once with the frozen body ReID model (resumable shards).

Images are decoded, flipped (train / val copies) and resized to the model input in
``DataLoader`` workers; the model call is :meth:`BodyReID.embed` itself, so the vectors are
identical to what the pipeline stores. Pre-resizing to the model size is exact because
``preprocess_crops`` resizes to that same size again (a copy). Logs ``ETA |`` lines like the
AI City benchmark. See :mod:`reid.attributes.store` for the layout and the resume rule.
"""

from __future__ import annotations

import logging
import os
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from reid.attributes import store
from reid.attributes.datasets import ParSplit
from reid.eval.aicity_progress import fmt_remaining

log = logging.getLogger(__name__)

FLIP_SPLITS = ("train", "val")  # flipped copies only here: no test image or mirror is in train


class Embedder(Protocol):
    """What embedding needs from a body model (:class:`reid.models.body_reid.BodyReID`)."""

    name: str
    neck_feat: str
    dim: int
    h: int
    w: int

    def embed(self, crops: list[np.ndarray]) -> np.ndarray: ...


class _ShardItems:
    """Map-style dataset of one shard: item -> uint8 ``(h, w, 3)`` BGR (+ read-ok flag)."""

    def __init__(self, paths: list[Path], flips: list[bool], h: int, w: int) -> None:
        self.paths, self.flips, self.h, self.w = paths, flips, h, w

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[np.ndarray, bool]:
        img = cv2.imread(str(self.paths[i]), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return np.zeros((self.h, self.w, 3), np.uint8), False
        if self.flips[i]:
            img = cv2.flip(img, 1)
        return cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR), True


class EtaLog:
    """Logs ``ETA | ...`` lines (images/s and remaining time) at most every ``interval_s``."""

    def __init__(self, total: int, interval_s: float = 30.0, clock=time.monotonic) -> None:  # noqa: ANN001
        self.total, self.done, self.interval_s, self._clock = total, 0, interval_s, clock
        self._t0 = clock()
        self._last = -1e18
        self.last_line = ""

    def update(self, n: int, where: str, force: bool = False) -> None:
        self.done += n
        now = self._clock()
        if not force and now - self._last < self.interval_s:
            return
        self._last = now
        speed = self.done / max(now - self._t0, 1e-9)
        left = (self.total - self.done) / speed if speed > 0 else 0.0
        pct = 100.0 * self.done / max(self.total, 1)
        self.last_line = (
            f"ETA | {where} | {self.done}/{self.total} items ({pct:.0f}%) | {speed:.1f} img/s "
            f"| ~{fmt_remaining(left)} left"
        )
        log.info(self.last_line)

    @property
    def rate(self) -> float:
        return self.done / max(self._clock() - self._t0, 1e-9)


def build_embedder(preset: str, batch: int) -> Any:
    """:class:`BodyReID` of a preset (default balanced: CLIP-ReID MSMT17)."""
    from reid.config import load_config
    from reid.models.body_reid import BodyReID

    cfg = load_config(None, {"runtime.batch_size": batch}, preset=preset)
    return BodyReID(cfg)


def _reid_version() -> str:
    try:
        return metadata.version("reid")
    except metadata.PackageNotFoundError:
        return "unknown"


def _embed_shard(
    emb: Embedder, items: _ShardItems, batch: int, workers: int
) -> tuple[np.ndarray, int]:
    """Embed one shard's items; returns ``(float32 (n, D), unreadable images)``."""
    from torch.utils.data import DataLoader

    loader = DataLoader(
        items,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=4 if workers > 0 else None,
    )
    out, failed = [], 0
    for imgs, ok in loader:
        vec = np.asarray(emb.embed(list(imgs.numpy())), np.float32)
        ok_np = ok.numpy().astype(bool)
        vec[~ok_np] = 0.0  # an unreadable image gets a zero vector, masked at training time
        failed += int((~ok_np).sum())
        out.append(vec)
    return np.concatenate(out) if out else np.zeros((0, emb.dim), np.float32), failed


def embed_split(
    emb: Embedder,
    sp: ParSplit,
    out_dir: str | Path,
    *,
    batch: int = 64,
    workers: int | None = None,
    shard_size: int = store.DEFAULT_SHARD,
    eta: EtaLog | None = None,
    part: tuple[int, int] = (0, 1),
) -> int:
    """Embed one split into shards (skipping shards already on disk); returns unreadable count.

    ``part = (i, n)`` embeds only the shards ``k`` with ``k % n == i`` (one process per GPU). A
    partial run never marks the split complete: a final run with ``part=(0, 1)`` finds every
    shard on disk, embeds nothing and marks it complete. The unreadable-image count is then
    approximate (parts overwrite each other's count); unreadable images are zero vectors and are
    masked at training time anyway.
    """
    workers = min(4, os.cpu_count() or 1) if workers is None else workers
    n = len(sp)
    flip = sp.split in FLIP_SPLITS
    total = store.n_items(n, flip)
    sdir = store.split_dir(out_dir, sp.dataset, sp.split)
    index = {
        "dataset": sp.dataset, "split": sp.split, "images": sp.images, "with_flip": flip,
        "n_images": n, "n_items": total, "shard_size": shard_size, "model": emb.name,
        "neck_feat": emb.neck_feat, "fp16": True, "dim": emb.dim, "reid_version": _reid_version(),
        "complete": False, "failed": 0,
    }  # fmt: skip
    resumed = store.prepare_split(sdir, index, sp.targets, sp.persons)
    index["failed"] = (store.read_index(sdir) or {}).get("failed", 0)
    bounds = store.shard_bounds(total, shard_size)
    done = store.done_shards(sdir, total, shard_size) if resumed else set()
    paths = sp.paths()
    log.info(
        "%s/%s: %d items in %d shards (%d done)",
        sp.dataset,
        sp.split,
        total,
        len(bounds),
        len(done),
    )
    mine = [k for k in range(len(bounds)) if k % part[1] == part[0]]
    for k in mine:
        a, b = bounds[k]
        if k in done:
            continue
        ids = np.arange(a, b)
        items = _ShardItems([paths[i % n] for i in ids], [bool(i >= n) for i in ids], emb.h, emb.w)
        vec, failed = _embed_shard(emb, items, batch, workers)
        store.write_shard(sdir, k, vec)
        index["failed"] += failed
        store.write_index(sdir, index)  # keeps the unreadable count across a crash
        if eta:
            eta.update(b - a, f"{sp.dataset}/{sp.split} shard {k + 1}/{len(bounds)}", force=True)
    if part[1] > 1 and len(store.done_shards(sdir, total, shard_size)) < len(bounds):
        log.info("%s/%s: part %d/%d done", sp.dataset, sp.split, part[0], part[1])
        return int(index["failed"])
    index["complete"] = True
    store.write_index(sdir, index)
    if index["failed"]:
        log.warning(
            "%s/%s: %d unreadable images (zero vectors)", sp.dataset, sp.split, index["failed"]
        )
    return int(index["failed"])


def pending_items(
    sp: ParSplit, out_dir: str | Path, shard_size: int, part: tuple[int, int] = (0, 1)
) -> int:
    """Items of ``sp`` (in this ``part``) not yet embedded, for the ETA; assumes resumability."""
    total = store.n_items(len(sp), sp.split in FLIP_SPLITS)
    bounds = store.shard_bounds(total, shard_size)
    sdir = store.split_dir(out_dir, sp.dataset, sp.split)
    old = store.read_index(sdir)
    stale = old is None or old.get("shard_size") != shard_size or old.get("images") != sp.images
    done = set() if stale else store.done_shards(sdir, total, shard_size)
    return sum(b - a for k, (a, b) in enumerate(bounds) if k % part[1] == part[0] and k not in done)


def embed_datasets(
    emb: Embedder,
    datasets: dict[str, dict[str, ParSplit]],
    out_dir: str | Path,
    *,
    batch: int = 64,
    workers: int | None = None,
    shard_size: int = store.DEFAULT_SHARD,
    part: tuple[int, int] = (0, 1),
) -> dict[str, float]:
    """Embed every split of every dataset (or one ``part``); logs ETA lines and the throughput."""
    splits = [sp for per in datasets.values() for sp in per.values()]
    eta = EtaLog(sum(pending_items(sp, out_dir, shard_size, part) for sp in splits))
    for sp in splits:
        embed_split(
            emb, sp, out_dir, batch=batch, workers=workers, shard_size=shard_size, eta=eta,
            part=part,
        )  # fmt: skip
    log.info("embedding done: %d items, %.1f img/s", eta.done, eta.rate)
    return {"items": float(eta.done), "img_per_s": eta.rate}
