"""Named collections of L2-normalized float32 vectors with int keys and int labels (gid).

Persistence per collection ``<name>``: ``<name>.npy`` (N x D float32 vectors) and
``<name>.ids.npy`` (N x 2 int64: key, label). A NumPy brute-force backend is the default; an
optional ``faiss`` flat inner-product backend is used if importable (``backend="auto"`` or
``"faiss"``). Both sit behind the same :class:`VectorCollection` interface.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np

Backend = Literal["numpy", "faiss", "auto"]
NO_LABEL = -1

try:  # optional accelerator
    import faiss  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - depends on environment
    faiss = None


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


class VectorCollection:
    """One collection: add / remove / relabel / query / best_per_label."""

    def __init__(self, dim: int | None = None, backend: Backend = "numpy") -> None:
        if backend == "faiss" and faiss is None:
            raise ImportError("faiss is not installed (pip install faiss-cpu)")
        self._use_faiss = backend == "faiss" or (backend == "auto" and faiss is not None)
        self.dim = dim
        self._vecs = np.zeros((0, dim or 0), dtype=np.float32)
        self._keys = np.zeros(0, dtype=np.int64)
        self._labels = np.zeros(0, dtype=np.int64)
        self._row: dict[int, int] = {}
        self._index = None  # lazily built faiss index
        self.dirty = False

    # ---- introspection
    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: int) -> bool:
        return int(key) in self._row

    @property
    def keys(self) -> np.ndarray:
        return self._keys.copy()

    @property
    def labels(self) -> np.ndarray:
        return self._labels.copy()

    @property
    def backend(self) -> str:
        return "faiss" if self._use_faiss else "numpy"

    def vectors(self, keys: list[int] | None = None) -> np.ndarray:
        """Stored vectors (all, or those of ``keys`` in that order)."""
        if keys is None:
            return self._vecs.copy()
        return self._vecs[[self._row[int(k)] for k in keys]]

    def label_of(self, key: int) -> int:
        return int(self._labels[self._row[int(key)]])

    def keys_for_label(self, label: int) -> np.ndarray:
        return self._keys[self._labels == label]

    # ---- mutation
    def add(self, keys, vectors, labels=None) -> None:
        """Add vectors (re-normalized). Existing keys are replaced. ``labels`` default to -1."""
        keys = np.atleast_1d(np.asarray(keys, dtype=np.int64))
        vecs = _normalize(np.atleast_2d(vectors))
        if len(keys) != len(vecs):
            raise ValueError("keys and vectors differ in length")
        if len(set(keys.tolist())) != len(keys):
            raise ValueError("duplicate keys in one add() call")
        lab = (
            np.full(len(keys), NO_LABEL, dtype=np.int64)
            if labels is None
            else np.atleast_1d(np.asarray(labels, dtype=np.int64))
        )
        if len(lab) != len(keys):
            raise ValueError("labels and keys differ in length")
        if len(keys) == 0:
            return
        if self.dim is None or self._vecs.shape[1] == 0:
            self.dim = vecs.shape[1]
            self._vecs = np.zeros((len(self), self.dim), dtype=np.float32)
        if vecs.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vecs.shape[1]}")
        existing = [int(k) for k in keys if int(k) in self._row]
        if existing:
            self.remove(existing)
        start = len(self)
        self._vecs = np.concatenate([self._vecs, vecs]) if start else vecs
        self._keys = np.concatenate([self._keys, keys])
        self._labels = np.concatenate([self._labels, lab])
        for i, k in enumerate(keys.tolist()):
            self._row[k] = start + i
        self._touch()

    def remove(self, keys) -> int:
        """Remove by key; unknown keys are ignored. Returns the number removed."""
        drop = {int(k) for k in np.atleast_1d(keys)} & self._row.keys()
        if not drop:
            return 0
        keep = np.fromiter((int(k) not in drop for k in self._keys), dtype=bool, count=len(self))
        self._vecs, self._keys, self._labels = (
            self._vecs[keep],
            self._keys[keep],
            self._labels[keep],
        )
        self._row = {int(k): i for i, k in enumerate(self._keys)}
        self._touch()
        return len(drop)

    def remove_label(self, label: int) -> int:
        """Remove every vector carrying ``label``."""
        return self.remove(self.keys_for_label(label))

    def relabel(
        self, mapping: dict[int, int] | None = None, *, keys=None, label: int | None = None
    ) -> None:
        """Change labels: ``relabel({key: new_label})`` or ``relabel(keys=[...], label=7)``."""
        if mapping is not None:
            for k, lab in mapping.items():
                self._labels[self._row[int(k)]] = lab
        if keys is not None:
            if label is None:
                raise ValueError("label required with keys")
            for k in np.atleast_1d(keys):
                self._labels[self._row[int(k)]] = label
        self.dirty = True  # labels are not part of the faiss index, so it stays valid

    def merge_labels(self, old: int, new: int) -> None:
        """Move every vector labelled ``old`` to ``new`` (e.g. after a gid merge)."""
        self._labels[self._labels == old] = new
        self.dirty = True

    def _touch(self) -> None:
        self.dirty = True
        self._index = None

    # ---- search
    def _search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, rows) of the k best rows, sorted by score descending."""
        n = len(self)
        k = min(k, n)
        if k <= 0:
            return np.zeros(0, np.float32), np.zeros(0, np.int64)
        if self._use_faiss:
            if self._index is None:
                self._index = faiss.IndexFlatIP(self.dim)
                self._index.add(self._vecs)
            s, r = self._index.search(q[None, :], k)
            return s[0], r[0].astype(np.int64)
        scores = self._vecs @ q
        if k < n:
            rows = np.argpartition(-scores, k - 1)[:k]
            rows = rows[np.argsort(-scores[rows], kind="stable")]
        else:
            rows = np.argsort(-scores, kind="stable")
        return scores[rows], rows

    def _prep(self, vec) -> np.ndarray:
        q = _normalize(np.asarray(vec, dtype=np.float32).reshape(-1))
        if self.dim is not None and q.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {q.shape[0]}")
        return q

    def query(
        self, vec, top_k: int = 10, exclude_labels: set[int] | None = None
    ) -> list[tuple[int, int, float]]:
        """Top-k neighbours as ``(key, label, score)`` sorted by cosine score descending."""
        if len(self) == 0:
            return []
        q = self._prep(vec)
        if exclude_labels:
            ok = ~np.isin(self._labels, list(exclude_labels))
            rows_ok = np.nonzero(ok)[0]
            if len(rows_ok) == 0:
                return []
            scores = self._vecs[rows_ok] @ q
            order = np.argsort(-scores, kind="stable")[:top_k]
            rows, sc = rows_ok[order], scores[order]
        else:
            sc, rows = self._search(q, top_k)
        return [
            (int(self._keys[r]), int(self._labels[r]), float(s))
            for r, s in zip(rows, sc, strict=True)
        ]

    def best_per_label(
        self, vec, top_k: int | None = None, exclude_labels: set[int] | None = None
    ) -> list[tuple[int, int, float]]:
        """Best vector of each label as ``(label, key, score)``, sorted by score descending.

        Unlabelled vectors (label -1) are ignored. ``top_k`` limits the number of labels.
        """
        if len(self) == 0:
            return []
        q = self._prep(vec)
        sc, rows = self._search(q, len(self))
        labs = self._labels[rows]
        mask = labs != NO_LABEL
        if exclude_labels:
            mask &= ~np.isin(labs, list(exclude_labels))
        sc, rows, labs = sc[mask], rows[mask], labs[mask]
        _, first = np.unique(labs, return_index=True)  # first occurrence == best score
        first.sort()
        out = [(int(labs[i]), int(self._keys[rows[i]]), float(sc[i])) for i in first]
        return out[:top_k] if top_k else out

    # ---- persistence
    def save(self, path_stem: str | Path) -> None:
        """Write ``<stem>.npy`` and ``<stem>.ids.npy`` atomically."""
        stem = Path(path_stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        _save_npy(stem.with_name(stem.name + ".ids.npy"), np.stack([self._keys, self._labels], 1))
        _save_npy(stem.with_name(stem.name + ".npy"), self._vecs)
        self.dirty = False

    @classmethod
    def load(cls, path_stem: str | Path, backend: Backend = "numpy") -> VectorCollection:
        stem = Path(path_stem)
        vecs = np.load(stem.with_name(stem.name + ".npy"))
        ids = np.load(stem.with_name(stem.name + ".ids.npy"))
        col = cls(dim=vecs.shape[1] if vecs.ndim == 2 else None, backend=backend)
        if len(ids):
            col._vecs = np.ascontiguousarray(vecs, dtype=np.float32)
            col._keys = ids[:, 0].astype(np.int64)
            col._labels = ids[:, 1].astype(np.int64)
            col._row = {int(k): i for i, k in enumerate(col._keys)}
        return col


class VectorStore:
    """Named collections persisted under ``root`` (e.g. ``data_dir/vectors``)."""

    def __init__(self, root: str | Path, backend: Backend = "numpy") -> None:
        self.root = Path(root)
        self.backend = backend
        self._cols: dict[str, VectorCollection] = {}

    def names(self) -> list[str]:
        """Names of collections in memory or on disk."""
        on_disk = (
            {p.name[: -len(".ids.npy")] for p in self.root.glob("*.ids.npy")}
            if self.root.exists()
            else set()
        )
        return sorted(on_disk | self._cols.keys())

    def get(self, name: str, dim: int | None = None) -> VectorCollection:
        """Return the collection ``name``, loading it from disk or creating it empty."""
        if name not in self._cols:
            stem = self.root / name
            if stem.with_name(name + ".ids.npy").exists():
                self._cols[name] = VectorCollection.load(stem, self.backend)
            else:
                self._cols[name] = VectorCollection(dim, self.backend)
        return self._cols[name]

    def drop(self, name: str) -> None:
        """Delete a collection from memory and disk."""
        self._cols.pop(name, None)
        for suffix in (".npy", ".ids.npy"):
            (self.root / (name + suffix)).unlink(missing_ok=True)

    def save(self) -> None:
        """Persist every modified collection."""
        for name, col in self._cols.items():
            if col.dirty:
                col.save(self.root / name)
