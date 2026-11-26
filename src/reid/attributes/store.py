"""On-disk embeddings of the attribute datasets: resumable fp16 shards plus targets.

Layout ``<emb_dir>/<dataset>/<split>/``: ``shard_00000.npy`` ... (fp16, ``(n, D)``),
``index.json``, ``targets.npy`` (int8 ``(N, 67)``, one row per image) and ``persons.npy`` (PETA
person keys, empty for PA-100K). The embedded items are the ``N`` original images followed, for
train and val, by their ``N`` horizontally flipped copies; item ``k`` belongs to image
``k % N`` and the flip flag is ``k >= N``. Shards hold ``shard_size`` consecutive items and are
written atomically, so a crash loses at most the shard being written.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_SHARD = 4096


def split_dir(emb_dir: str | Path, dataset: str, split: str) -> Path:
    """Folder of one dataset split."""
    return Path(emb_dir) / dataset / split


def n_items(n_images: int, with_flip: bool) -> int:
    """Number of embedded items of a split."""
    return n_images * (2 if with_flip else 1)


def shard_bounds(total: int, shard_size: int) -> list[tuple[int, int]]:
    """``[start, stop)`` item ranges of the shards."""
    return [(s, min(s + shard_size, total)) for s in range(0, total, shard_size)]


def shard_path(sdir: Path, k: int) -> Path:
    return sdir / f"shard_{k:05d}.npy"


def _atomic_write(path: Path, writer) -> None:  # noqa: ANN001
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # per process: parallel parts
    with open(tmp, "wb") as f:
        writer(f)
    os.replace(tmp, path)


def read_index(sdir: Path) -> dict | None:
    """``index.json`` of a split, or None."""
    p = sdir / "index.json"
    return json.loads(p.read_text()) if p.is_file() else None


def write_index(sdir: Path, index: dict) -> None:
    _atomic_write(sdir / "index.json", lambda f: f.write(json.dumps(index, indent=1).encode()))


def prepare_split(sdir: Path, index: dict, targets: np.ndarray, persons: list[str]) -> bool:
    """Create the split folder or validate it for resuming; returns True when resuming.

    Shards of a different image list / model / shard size are deleted and the split restarts.
    """
    sdir.mkdir(parents=True, exist_ok=True)
    old = read_index(sdir)
    keys = ("images", "model", "neck_feat", "shard_size", "with_flip")
    resume = old is not None and all(old.get(k) == index.get(k) for k in keys)
    if not resume:
        for p in sdir.glob("shard_*.npy"):
            p.unlink()
    else:
        index = {**index, "failed": old.get("failed", 0)}
    write_index(sdir, index)
    _atomic_write(sdir / "targets.npy", lambda f: np.save(f, targets.astype(np.int8)))
    pers = np.array(persons, dtype=str) if persons else np.array([], dtype="<U1")
    _atomic_write(sdir / "persons.npy", lambda f: np.save(f, pers))
    return resume


def write_shard(sdir: Path, k: int, emb: np.ndarray) -> None:
    """Atomically save shard ``k`` as float16."""
    _atomic_write(shard_path(sdir, k), lambda f: np.save(f, emb.astype(np.float16)))


def done_shards(sdir: Path, total: int, shard_size: int) -> set[int]:
    """Indices of complete shards (right row count) already on disk."""
    done = set()
    for k, (a, b) in enumerate(shard_bounds(total, shard_size)):
        p = shard_path(sdir, k)
        if p.is_file():
            try:
                if np.load(p, mmap_mode="r").shape[0] == b - a:
                    done.add(k)
            except (OSError, ValueError):
                p.unlink()
    return done


@dataclass
class SplitEmb:
    """Loaded embeddings of one split: ``x[k]`` belongs to image ``base[k]``."""

    x: np.ndarray  # float16 (M, D)
    base: np.ndarray  # int (M,) image index of each row
    flip: np.ndarray  # bool (M,)
    targets: np.ndarray  # int8 (N, 67), one row per image
    persons: np.ndarray  # str (N,) or (0,)
    index: dict

    @property
    def row_targets(self) -> np.ndarray:
        """Targets expanded to one row per embedding (flipped copies repeat their image)."""
        return self.targets[self.base]


def load_split(
    emb_dir: str | Path, dataset: str, split: str, include_flips: bool = True
) -> SplitEmb:
    """Load a finished split (all shards present). ``include_flips=False`` keeps originals."""
    sdir = split_dir(emb_dir, dataset, split)
    index = read_index(sdir)
    if index is None or not index.get("complete"):
        raise FileNotFoundError(f"embeddings of {dataset}/{split} are missing or incomplete")
    targets = np.load(sdir / "targets.npy")
    persons = np.load(sdir / "persons.npy")
    n = len(targets)
    total = n_items(n, index["with_flip"])
    parts = [
        np.load(shard_path(sdir, k)) for k in range(len(shard_bounds(total, index["shard_size"])))
    ]
    x = np.concatenate(parts) if parts else np.zeros((0, index["dim"]), np.float16)
    if len(x) != total:
        raise ValueError(f"{sdir}: {len(x)} embeddings for {total} items")
    base = np.arange(total) % max(n, 1)
    flip = np.arange(total) >= n
    if not include_flips:
        x, base, flip = x[:n], base[:n], flip[:n]
    return SplitEmb(x, base, flip, targets, persons, index)


def available_splits(emb_dir: str | Path, dataset: str) -> list[str]:
    """Complete splits of a dataset."""
    out = []
    for s in ("train", "val", "test"):
        idx = read_index(split_dir(emb_dir, dataset, s))
        if idx and idx.get("complete"):
            out.append(s)
    return out
