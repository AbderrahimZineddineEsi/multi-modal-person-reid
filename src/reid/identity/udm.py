"""Unknown Data Manager over pooled tracklet descriptors.

Implements Casao et al. 2023, Sec. 3.3 (Unknown Data Manager):

* unmatched data are pooled and clustered with DBSCAN (density based, handles an unbounded
  number of classes and noise);
* a cluster is only considered once it holds at least ``l`` samples (l = 20);
* the cluster is compared with the gallery through the *k = 3 most frequent classes among the
  k nearest centroids of its samples*; its distance to the gallery is the minimum over those
  classes' samples of ``1 - x.x_j``; if that distance exceeds ``alpha = 0.1`` the cluster becomes
  a new identity, otherwise it merges into the class of the closest sample.

Deviations (and why):

* DBSCAN runs on **tracklet descriptors** (not frames), so ``min_samples`` is 1 tracklet; the
  paper's ``l`` counts the *samples* (<= 16 per tracklet) of the pooled tracklets. The paper does
  not give ``eps``; we use 0.15 cosine distance (v1's value).
* Cannot-links are honoured: pairs that violate a constraint get distance 2 before DBSCAN and each
  cluster is then split greedily into constraint-compatible groups (DBSCAN alone could chain
  two simultaneous people through a third); gids blocked for any member are not merge targets.
* ``flush=True`` (``end_run``) drops the size threshold ``l`` so nobody stays unidentified (v1 B8);
  the new/merge distance test still applies.
* The paper's Q = 20 "initialization" is handled in ``associate.py`` (bootstrap rule).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN

from reid.config import AssociationCfg
from reid.identity.constraints import Span, face_conflict
from reid.identity.gallery import GalleryBank


@dataclass
class PoolItem:
    """An unmatched tracklet waiting in the pool."""

    span: Span
    descriptor: np.ndarray
    embs: np.ndarray
    weights: np.ndarray
    sample_ids: list[int]
    face: np.ndarray | None = None
    n_face_obs: int = 0


@dataclass
class UdmResolution:
    """Decision for one cluster: merge into ``gid`` or mint a new identity (``gid is None``)."""

    items: list[PoolItem]
    gid: int | None
    distance: float
    n_samples: int = 0
    reason: str = ""
    extra: dict = field(default_factory=dict)


class UnknownDataManager:
    """Pool of unmatched tracklets plus the clustering / cluster-test logic."""

    def __init__(self, cfg: AssociationCfg) -> None:
        self.cfg = cfg
        self.pool: list[PoolItem] = []

    def __len__(self) -> int:
        return len(self.pool)

    def add(self, item: PoolItem) -> None:
        self.pool.append(item)

    # ---- clustering
    def clusters(self, violates: Callable[[Span, Span], bool]) -> list[list[PoolItem]]:
        """DBSCAN on cosine distance, then split into constraint-compatible groups."""
        n = len(self.pool)
        if n == 0:
            return []
        d = np.stack([p.descriptor for p in self.pool])
        dist = np.clip(1.0 - d @ d.T, 0.0, 2.0)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.pool[i], self.pool[j]
                if violates(a.span, b.span) or (
                    a.n_face_obs >= self.cfg.face_min_obs
                    and b.n_face_obs >= self.cfg.face_min_obs
                    and face_conflict(a.face, b.face, self.cfg.face_conflict)
                ):
                    dist[i, j] = dist[j, i] = 2.0
        labels = DBSCAN(eps=self.cfg.udm_eps, min_samples=1, metric="precomputed").fit_predict(dist)
        out: list[list[PoolItem]] = []
        for lab in sorted(set(labels.tolist())):
            members = [i for i in range(n) if labels[i] == lab]
            members.sort(key=lambda i: -len(self.pool[i].sample_ids))  # stable
            groups: list[list[int]] = []
            for i in members:
                for g in groups:
                    if all(dist[i, j] < 2.0 for j in g):
                        g.append(i)
                        break
                else:
                    groups.append([i])
            out.extend([[self.pool[i] for i in sorted(g)] for g in groups])
        return out

    # ---- cluster test
    def cluster_test(
        self, group: list[PoolItem], bank: GalleryBank, blocked: set[int]
    ) -> tuple[int | None, float]:
        """Casao's new-class test. Returns ``(gid or None for new, distance)``."""
        gids = [g for g in bank.gids() if g not in blocked]
        if not gids:
            return None, float("inf")
        x = np.concatenate([p.embs for p in group])
        _, cents = bank.centroids(gids)
        k = min(self.cfg.udm_k, len(gids))
        sims = x @ cents.T
        near = np.argsort(-sims, axis=1, kind="stable")[:, :k]
        votes = np.bincount(near.ravel(), minlength=len(gids))
        tot = np.zeros(len(gids))  # similarity mass breaks ties between equally voted classes
        np.add.at(tot, near.ravel(), np.take_along_axis(sims, near, 1).ravel())
        cand = sorted(range(len(gids)), key=lambda i: (-votes[i], -tot[i]))[:k]
        best_gid, best_d = None, float("inf")
        for i in cand:
            dmin = float((1.0 - x @ bank.galleries[gids[i]].embs.T).min())
            if dmin < best_d:
                best_gid, best_d = gids[i], dmin
        if best_d > self.cfg.udm_new_dist:
            return None, best_d
        return best_gid, best_d

    def process(
        self,
        bank: GalleryBank,
        violates: Callable[[Span, Span], bool],
        blocked_for: Callable[[PoolItem], set[int]],
        flush: bool = False,
    ) -> list[UdmResolution]:
        """Evaluate every eligible cluster; resolved items leave the pool."""
        out: list[UdmResolution] = []
        done: set[int] = set()
        for group in self.clusters(violates):
            total = sum(len(p.sample_ids) for p in group)
            if not flush and total < self.cfg.udm_min_samples:
                continue
            blocked: set[int] = set()
            for p in group:
                blocked |= blocked_for(p)
            gid, dist = self.cluster_test(group, bank, blocked)
            out.append(UdmResolution(group, gid, dist, total, "flush" if flush else "udm"))
            done |= {id(p) for p in group}
        self.pool = [p for p in self.pool if id(p) not in done]
        return out


def total_samples(items: Iterable[PoolItem]) -> int:
    """Total samples of a set of pooled tracklets."""
    return sum(len(p.sample_ids) for p in items)
