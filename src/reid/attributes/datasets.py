"""PA-100K and PETA loaders: image lists, unified targets and (PETA) person keys per split.

PA-100K keeps its official 80k / 10k / 10k split. PETA has no official split: persons
``(subset, ID)`` are split 50 / 10 / 40 % (train / val / test) with a seeded permutation, so no
person appears in two splits (a random image split would leak identities into the test set).
The i-LID subset is skipped (it needs a separate application and licence).
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from reid.attributes.schema import check_pa100k_names, pa100k_targets, peta_targets

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
PETA_SKIP = ("i-lid",)
PETA_SPLIT_FRACTIONS = (0.5, 0.1, 0.4)
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass
class ParSplit:
    """One split of one dataset: image paths (relative to ``root``) and unified targets."""

    dataset: str
    split: str
    root: Path
    images: list[str]
    targets: np.ndarray  # int8 (N, 67)
    persons: list[str] = field(default_factory=list)  # "subset/ID" per image (PETA), else []

    def __len__(self) -> int:
        return len(self.images)

    def paths(self) -> list[Path]:
        """Absolute image paths."""
        return [self.root / r for r in self.images]


def _subdirs(root: Path, max_depth: int) -> list[Path]:
    """``root`` and its directories down to ``max_depth`` (breadth first, no file listing)."""
    out, level = [root], [root]
    for _ in range(max_depth):
        nxt: list[Path] = []
        for d in level:
            try:
                with os.scandir(d) as it:
                    nxt += [Path(e.path) for e in it if e.is_dir(follow_symlinks=True)]
            except OSError:
                continue
        out += nxt
        level = nxt
    return out


def find_file(root: Path, name: str, max_depth: int = 4) -> Path | None:
    """First file called ``name`` in ``root`` or its sub-folders (depth limited)."""
    for d in _subdirs(Path(root), max_depth):
        if (d / name).is_file():
            return d / name
    return None


# ------------------------------------------------------------------------------- PA-100K
def _names(arr: np.ndarray) -> list[str]:
    return [str(np.asarray(x).ravel()[0]) for x in np.asarray(arr).ravel()]


def find_pa100k_images(root: Path, first_name: str) -> Path | None:
    """Folder that holds the PA-100K images (the one containing ``first_name``)."""
    for d in _subdirs(Path(root), 4):
        if (d / first_name).is_file():
            return d
    return None


def load_pa100k(root: str | Path, require_images: bool = True) -> dict[str, ParSplit]:
    """Load PA-100K from a folder with ``annotation.mat`` and the images (anywhere below it).

    With ``require_images=False`` (annotation only) paths are relative to ``root/data``.
    """
    from scipy.io import loadmat

    root = Path(root)
    mat_path = find_file(root, "annotation.mat")
    if mat_path is None:
        raise FileNotFoundError(f"annotation.mat not found under {root}")
    mat = loadmat(str(mat_path))
    check_pa100k_names(_names(mat["attributes"]))
    names = {s: _names(mat[f"{s}_images_name"]) for s in SPLITS}
    img_dir = find_pa100k_images(root, names["train"][0])
    if img_dir is None:
        if require_images:
            raise FileNotFoundError(f"PA-100K images ({names['train'][0]}) not found under {root}")
        img_dir = root / "data"
    out = {}
    for s in SPLITS:
        rel = [str((img_dir / n).relative_to(root)) for n in names[s]]
        out[s] = ParSplit("pa100k", s, root, rel, pa100k_targets(mat[f"{s}_label"]))
    return out


# ------------------------------------------------------------------------------------ PETA
def find_peta_root(root: str | Path) -> Path:
    """Folder whose children are the PETA subsets (``<subset>/archive/Label.txt``)."""
    for d in _subdirs(Path(root), 3):
        if any(c.is_dir() and (c / "archive" / "Label.txt").is_file() for c in d.iterdir()):
            return d
    raise FileNotFoundError(f"PETA subsets (<subset>/archive/Label.txt) not found under {root}")


def _read_labels(path: Path) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if parts:
            labels[parts[0]] = parts[1:]
    return labels


def _person_id(file_name: str, labels: dict[str, list[str]]) -> str | None:
    """Label key of an image: its file name (CUHK), else the prefix before the first ``_``."""
    stem = file_name.rsplit(".", 1)[0]
    for cand in (file_name, stem, file_name.split("_")[0], stem.split("_")[0]):
        if cand in labels:
            return cand
    return None


def split_persons(
    persons: list[str], seed: int = 0, fractions: tuple[float, float, float] = PETA_SPLIT_FRACTIONS
) -> dict[str, str]:
    """Seeded person-disjoint split: ``person key -> 'train' | 'val' | 'test'``."""
    uniq = sorted(set(persons))
    order = np.random.default_rng(seed).permutation(len(uniq))
    n_train = round(fractions[0] * len(uniq))
    n_val = round(fractions[1] * len(uniq))
    out = {}
    for rank, idx in enumerate(order):
        out[uniq[idx]] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
    return out


def scan_peta(root: str | Path) -> tuple[Path, list[str], list[list[str]], list[str]]:
    """All PETA images without i-LID: ``(root, relative paths, tokens, person keys)``."""
    peta = find_peta_root(root)
    rel: list[str] = []
    tokens: list[list[str]] = []
    persons: list[str] = []
    unmatched = 0
    for sub in sorted(p for p in peta.iterdir() if p.is_dir()):
        label_file = sub / "archive" / "Label.txt"
        if sub.name.lower() in PETA_SKIP or not label_file.is_file():
            continue
        labels = _read_labels(label_file)
        for img in sorted((sub / "archive").iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            pid = _person_id(img.name, labels)
            if pid is None:
                unmatched += 1
                continue
            rel.append(str(img.relative_to(peta)))
            tokens.append(labels[pid])
            persons.append(f"{sub.name}/{pid}")
    if unmatched:
        log.warning("PETA: %d images without a Label.txt entry were skipped", unmatched)
    return peta, rel, tokens, persons


def load_peta(root: str | Path, seed: int = 0) -> dict[str, ParSplit]:
    """Load PETA (without i-LID) with a seeded person-disjoint 50 / 10 / 40 % split."""
    peta, rel, tokens, persons = scan_peta(root)
    which = split_persons(persons, seed)
    targets = peta_targets(tokens)
    out = {}
    for s in SPLITS:
        idx = [i for i, p in enumerate(persons) if which[p] == s]
        out[s] = ParSplit(
            "peta", s, peta, [rel[i] for i in idx], targets[idx], [persons[i] for i in idx]
        )
    n_persons = {s: len({p for p in out[s].persons}) for s in SPLITS}
    log.info("PETA: %d images, persons per split %s", len(rel), n_persons)
    return out


def positive_counts(split: ParSplit) -> dict[str, tuple[int, int]]:
    """Per unified label ``(positives, negatives)`` of a split (unknown excluded)."""
    from reid.attributes.schema import LABELS

    t = split.targets
    return {n: (int((t[:, j] == 1).sum()), int((t[:, j] == 0).sum())) for j, n in enumerate(LABELS)}


def group_by_person(persons: list[str]) -> dict[str, list[int]]:
    """Indices of each person's images."""
    out: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(persons):
        out[p].append(i)
    return dict(out)
