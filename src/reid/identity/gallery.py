"""Self-adaptive gallery per identity (pure NumPy).

Implements Casao, Azagra, Murillo, Montijano, "A Self-Adaptive Gallery Construction Method for
Open-World Person Re-Identification", Sensors 23(5):2662, 2023:

* Sec. 3.1/3.2: each class keeps at most ``m`` (50) descriptors; when over budget the sample
  ``x* = argmax[gamma * H(x)/log N - (1 - gamma) * D_i(x)]`` is dropped (Eq. 10), where
  ``D_i(x) = min_j (1 - x.x_j)`` is the intra-class diversity (Eq. 5) and ``H`` the Shannon
  entropy of the softmax over class centroids (Eq. 8).
* Eq. 5-7: a new sample ``x_q`` is inserted only if ``D_i(x_q) >= D(C_i)`` (diversity gate).
* Sec. 3.3 + Technique 4: class centroids are visibility-weighted means; the softmax-ratio test
  ``p1/p2 >= tau`` (tau = 2, temperature v = 0.1, i.e. scores x10) accepts a match.

Deviations (and why):

* The entropy normaliser is ``log N`` (N = current number of classes). The paper prints
  ``log(1/N)``, which is negative and would flip the sign; the text says it normalises the
  entropy to [0, 1]. With N = 1 the entropy term is 0.
* Centroid weights are our per-sample ``quality`` (pose visibility x detection confidence,
  computed upstream), a superset of the paper's visibility ratio.
* ``D(C_i)`` defaults to the paper's Eq. 6 (smallest pairwise distance). ``diversity_ref="mean"``
  (mean nearest-neighbour distance, PLAN section 3.3's reading) is a stricter option.
* Applied to tracklet samples, never to raw consecutive frames, and only after the tracklet's
  identity is confirmed (verify-before-update); see ``associate.py``.
* The softmax test excludes cannot-linked gids (they get probability 0, as in v1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9


def l2n(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows (float32); zero vectors stay zero."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def weighted_mean(embs: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Normalized weighted mean of L2-normalized rows (visibility/quality-weighted centroid)."""
    if weights is None or float(np.sum(weights)) <= 0:
        weights = np.ones(len(embs), dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    return l2n((embs * w[:, None]).sum(0))


def nn_distance(embs: np.ndarray) -> np.ndarray:
    """``D_i(x)`` for every member: 1 - max cosine to another member (inf if alone)."""
    n = len(embs)
    if n < 2:
        return np.full(n, np.inf, dtype=np.float32)
    s = embs @ embs.T
    np.fill_diagonal(s, -np.inf)
    return 1.0 - s.max(axis=1)


def class_diversity(embs: np.ndarray, ref: str = "min") -> float:
    """``D(C_i)``: smallest pairwise distance (``ref="min"``) or mean NN distance (``"mean"``)."""
    if len(embs) < 2:
        return float("inf")
    d = nn_distance(embs)
    return float(d.min() if ref == "min" else d.mean())


def passes_diversity_gate(embs: np.ndarray, x: np.ndarray, ref: str = "min") -> bool:
    """Casao Eq. 7: insert ``x`` only if ``D_i(x) >= D(C_i)``. Always true below 2 members."""
    if len(embs) < 2:
        return True
    d_x = 1.0 - float(np.max(embs @ x))
    return d_x >= class_diversity(embs, ref) - 1e-7


def softmax(scores: np.ndarray, temp: float) -> np.ndarray:
    """Row-wise softmax of ``scores * temp`` (``temp = 1/v``)."""
    z = np.asarray(scores, dtype=np.float64) * temp
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def normalized_entropy(probs: np.ndarray, n_classes: int) -> np.ndarray:
    """Shannon entropy per row divided by ``log N`` (0 when N <= 1)."""
    if n_classes <= 1:
        return np.zeros(len(probs))
    h = -(probs * np.log(probs + EPS)).sum(axis=-1)
    return h / np.log(n_classes)


def drop_index(embs: np.ndarray, centroids: np.ndarray, gamma: float, temp: float) -> int:
    """Index of the sample to drop: ``argmax[gamma*H/log N - (1-gamma)*D_i]`` (Casao Eq. 10)."""
    h = normalized_entropy(softmax(embs @ centroids.T, temp), len(centroids))
    d = nn_distance(embs)
    d = np.where(np.isfinite(d), d, 0.0)
    return int(np.argmax(gamma * h - (1.0 - gamma) * d))


@dataclass
class BodyDecision:
    """Outcome of the softmax-ratio test."""

    gid: int | None  # best candidate (even when rejected), None if no candidate
    cos: float = 0.0
    ratio: float = 0.0  # p1/p2 (inf with a single candidate)
    accepted: bool = False


def softmax_ratio_test(
    x: np.ndarray,
    gids: list[int],
    centroids: np.ndarray,
    *,
    blocked: set[int] | frozenset[int] = frozenset(),
    temp: float,
    tau: float,
    min_cos: float,
) -> BodyDecision:
    """Casao Sec. 3.3 acceptance: ``p1/p2 >= tau`` and ``cos >= min_cos`` (v1 floor).

    Blocked (cannot-linked) gids are removed before the softmax. A single remaining candidate
    has ratio inf and is accepted on the cosine floor alone (as in v1).
    """
    keep = [i for i, g in enumerate(gids) if g not in blocked]
    if not keep:
        return BodyDecision(None)
    sims = centroids[keep] @ x
    order = np.argsort(-sims, kind="stable")
    best = keep[int(order[0])]
    cos = float(sims[order[0]])
    if len(keep) == 1:
        ratio = float("inf")
    else:
        p = softmax(sims, temp)
        ratio = float(p[order[0]] / (p[order[1]] + EPS))
    ok = ratio >= tau and cos >= min_cos
    return BodyDecision(gids[best], cos, ratio, ok)


@dataclass
class Gallery:
    """Members of one identity's gallery (rows aligned with ``ids`` and ``weights``)."""

    embs: np.ndarray
    weights: np.ndarray
    ids: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)

    def centroid(self) -> np.ndarray:
        return weighted_mean(self.embs, self.weights)


class GalleryBank:
    """All identity galleries with Casao's insertion gate and budgeted pruning."""

    def __init__(
        self, budget: int = 50, gamma: float = 0.6, temp: float = 10.0, diversity_ref: str = "min"
    ) -> None:
        self.budget, self.gamma, self.temp, self.ref = budget, gamma, temp, diversity_ref
        self.galleries: dict[int, Gallery] = {}
        self._cent: dict[int, np.ndarray] = {}

    # ---- queries
    def __contains__(self, gid: int) -> bool:
        return gid in self.galleries

    def gids(self) -> list[int]:
        return sorted(self.galleries)

    def centroid(self, gid: int) -> np.ndarray:
        if gid not in self._cent:
            self._cent[gid] = self.galleries[gid].centroid()
        return self._cent[gid]

    def centroids(self, gids: list[int] | None = None) -> tuple[list[int], np.ndarray]:
        """``(gids, matrix[G, D])`` of visibility/quality-weighted centroids."""
        gids = self.gids() if gids is None else gids
        if not gids:
            return [], np.zeros((0, 0), dtype=np.float32)
        return gids, np.stack([self.centroid(g) for g in gids])

    # ---- mutation
    def _touch(self, gid: int) -> None:
        self._cent.pop(gid, None)

    def restore(self, gid: int, embs: np.ndarray, weights: np.ndarray, ids: list[int]) -> None:
        """Install a gallery as stored (no gate, no pruning), members sorted by id."""
        order = np.argsort(ids, kind="stable")
        self.galleries[gid] = Gallery(
            l2n(embs)[order], np.asarray(weights, np.float32)[order], [int(ids[i]) for i in order]
        )
        self._touch(gid)

    def remove(self, gid: int) -> None:
        self.galleries.pop(gid, None)
        self._touch(gid)

    def _prune_one(self, gid: int) -> int:
        g = self.galleries[gid]
        gids = self.gids()
        cents = np.stack([g.centroid() if k == gid else self.centroid(k) for k in gids])
        i = drop_index(g.embs, cents, self.gamma, self.temp)
        dropped = g.ids[i]
        keep = np.arange(len(g)) != i
        self.galleries[gid] = Gallery(
            g.embs[keep], g.weights[keep], [x for x in g.ids if x != dropped]
        )
        self._touch(gid)
        return dropped

    def prune(self, gid: int) -> list[int]:
        """Drop samples (Eq. 10) until the gallery is within budget; returns dropped ids."""
        out: list[int] = []
        while gid in self.galleries and len(self.galleries[gid]) > self.budget:
            out.append(self._prune_one(gid))
        return out

    def insert(self, gid: int, emb: np.ndarray, weight: float, sid: int) -> tuple[bool, int | None]:
        """Gate + insert + budget drop. Returns ``(inserted, dropped_id)``.

        ``inserted`` is False when the diversity gate rejects the sample; ``dropped_id`` may be
        the sample just inserted (as in the paper).
        """
        x = l2n(emb.reshape(1, -1))[0]
        if gid not in self.galleries:
            self.galleries[gid] = Gallery(x[None], np.array([weight], np.float32), [int(sid)])
            self._touch(gid)
            return True, None
        g = self.galleries[gid]
        if not passes_diversity_gate(g.embs, x, self.ref):
            return False, None
        ids = [*g.ids, int(sid)]
        order = np.argsort(ids, kind="stable")  # members stay sorted by id (reload-stable)
        self.galleries[gid] = Gallery(
            np.vstack([g.embs, x[None]])[order],
            np.append(g.weights, np.float32(weight))[order],
            [ids[i] for i in order],
        )
        self._touch(gid)
        dropped = self.prune(gid)
        return True, (dropped[0] if dropped else None)

    def insert_many(
        self, gid: int, embs: np.ndarray, weights: np.ndarray, ids: list[int]
    ) -> tuple[list[int], list[int]]:
        """Insert samples best-quality first. Returns ``(added_ids, removed_ids)``.

        ``added_ids`` are the samples that are members afterwards and were not before;
        ``removed_ids`` the previous members that were dropped.
        """
        before = set(self.galleries[gid].ids) if gid in self.galleries else set()
        for i in np.argsort(-np.asarray(weights), kind="stable"):
            self.insert(gid, embs[i], float(weights[i]), ids[i])
        after = set(self.galleries[gid].ids) if gid in self.galleries else set()
        return sorted(after - before), sorted(before - after)

    def merge(self, keep: int, absorb: int) -> tuple[list[int], list[int]]:
        """Union ``absorb`` into ``keep`` and re-prune; returns ``(moved_ids, dropped_ids)``.

        ``moved_ids`` are the members of ``absorb`` that survive in ``keep``.
        """
        a = self.galleries.pop(absorb, None)
        self._touch(absorb)
        if a is None:
            return [], []
        if keep not in self.galleries:
            self.galleries[keep] = a
            self._touch(keep)
            return list(a.ids), []
        k = self.galleries[keep]
        ids = [*k.ids, *a.ids]
        order = np.argsort(ids, kind="stable")
        self.galleries[keep] = Gallery(
            np.vstack([k.embs, a.embs])[order],
            np.concatenate([k.weights, a.weights])[order],
            [ids[i] for i in order],
        )
        self._touch(keep)
        dropped = self.prune(keep)
        survivors = set(self.galleries[keep].ids)
        return [i for i in a.ids if i in survivors], dropped
