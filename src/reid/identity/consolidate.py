"""Step 6: constrained agglomerative consolidation of identities, plus manual operations.

Not in Casao et al. 2023 (their gallery never repairs fragmentation). It replaces v1's
retroactive "identity healing" (v1_review 1.9, B4-B7) and DBSCAN-only minting:

* Agglomerative clustering (Lance-Williams update; ``average`` linkage by default) over the
  visibility/quality-weighted **gid centroids** of the galleries, stopped at ``merge_threshold``.
* **Must-links** (face evidence, see ``constraints.must_links``) are applied first; any pair
  that would violate a cannot-link is skipped. **Cannot-links** (same-camera overlap, topology,
  face conflict, persisted manual splits) are never violated, also transitively.
* The merged identity keeps the **oldest** gid; tracklets are relabelled with
  ``db.set_gids(source='consolidate')`` so every change is recorded in ``tracklet_gid_history``
  (reversible with :func:`split_tracklet`); galleries are unioned and re-pruned to the budget
  (fixes v1 B7); vector labels follow, so indices never go stale (B5/B6).

Manual operations (:func:`merge_identities`, :func:`split_tracklet`, :func:`rename_identity`)
and :func:`recluster` (used by the UI) live here too. Deviations from the plan: none.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from reid.config import CameraSet, Config
from reid.identity.constraints import gid_cannot_links, must_links
from reid.identity.prototypes import rebuild_prototypes
from reid.identity.store import IdentityStore
from reid.storage import db
from reid.storage.vectors import VectorStore

log = logging.getLogger(__name__)


@dataclass
class Merge:
    """One identity folded into another."""

    kept: int
    absorbed: int
    reason: str  # 'must_link' | 'agglomerative' | 'manual'
    distance: float = 0.0


@dataclass
class ConsolidationResult:
    """What a consolidation pass did."""

    merges: list[Merge] = field(default_factory=list)
    n_before: int = 0
    n_after: int = 0
    must_links: list[tuple[int, int]] = field(default_factory=list)
    skipped_must_links: list[tuple[int, int]] = field(default_factory=list)

    @property
    def touched(self) -> set[int]:
        return {m.kept for m in self.merges}


def oldest_gid(conn: sqlite3.Connection, gids: list[int]) -> int:
    """The oldest identity (earliest ``created_at``, then smallest gid)."""

    def key(g: int) -> tuple[float, int]:
        ident = db.get_identity(conn, g)
        return (ident.created_at if ident else float("inf"), g)

    return min(gids, key=key)


def _lw(linkage: str, ni: int, nj: int, di: np.ndarray, dj: np.ndarray) -> np.ndarray:
    if linkage == "complete":
        return np.maximum(di, dj)
    if linkage == "single":
        return np.minimum(di, dj)
    return (ni * di + nj * dj) / (ni + nj)


def consolidate(store: IdentityStore, restrict_gids: set[int] | None = None) -> ConsolidationResult:
    """Run constrained agglomerative clustering and apply the merges through ``store``."""
    cfg = store.cfg
    a, c = cfg.association, cfg.consolidation
    bank = store.bank
    gids = [
        g
        for g in bank.gids()
        if len(bank.galleries[g]) >= c.min_samples and (restrict_gids is None or g in restrict_gids)
    ]
    res = ConsolidationResult(n_before=len(gids), n_after=len(gids))
    if len(gids) < 2:
        return res
    _, cents = bank.centroids(gids)
    dist = np.clip(1.0 - cents @ cents.T, 0.0, 2.0).astype(np.float64)
    faces = {g: f for g, f in store.gid_faces().items() if g in gids} if c.use_faces else {}
    cl = gid_cannot_links(
        store.spans(), gids, a, store.cameras, faces, c.use_faces, db.get_cannot_links(store.conn)
    )
    idx = {g: i for i, g in enumerate(gids)}
    n = len(gids)
    m = dist.copy()
    np.fill_diagonal(m, np.inf)
    members: list[list[int]] = [[i] for i in range(n)]
    active = np.ones(n, dtype=bool)
    owner = list(range(n))
    kinds: list[set[str]] = [set() for _ in range(n)]

    def merge(i: int, j: int, reason: str) -> None:
        di, dj = m[i].copy(), m[j].copy()
        new = _lw(c.linkage, len(members[i]), len(members[j]), di, dj)
        m[i, :], m[:, i] = new, new
        m[i, i] = np.inf
        m[j, :], m[:, j] = np.inf, np.inf
        cl[i, :] |= cl[j, :]
        cl[:, i] |= cl[:, j]
        cl[j, :] = cl[:, j] = False
        cl[i, i] = False
        for g in members[j]:
            owner[g] = i
        members[i] += members[j]
        members[j] = []
        active[j] = False
        kinds[i] |= kinds[j] | {reason}

    if c.use_faces:
        ml = sorted(must_links(faces, a))
        res.must_links = ml
        for ga, gb in ml:
            i, j = owner[idx[ga]], owner[idx[gb]]
            if i == j:
                continue
            if cl[i, j]:
                res.skipped_must_links.append((ga, gb))
                continue
            merge(min(i, j), max(i, j), "must_link")
    while True:
        masked = np.where(cl, np.inf, m)
        flat = int(np.argmin(masked))
        i, j = divmod(flat, n)
        if not np.isfinite(masked[i, j]) or masked[i, j] > c.merge_threshold:
            break
        merge(min(i, j), max(i, j), "agglomerative")

    for k in range(n):
        if len(members[k]) < 2:
            continue
        kept = oldest_gid(store.conn, [gids[g] for g in members[k]])
        for g in members[k]:
            if gids[g] != kept:
                kind = "must_link" if kinds[k] == {"must_link"} else "agglomerative"
                d = float(dist[idx[kept], g])
                store.merge_gids(kept, gids[g], reason=f"consolidate:{kind}")
                res.merges.append(Merge(kept, gids[g], kind, d))
    res.n_after = n - len(res.merges)
    return res


def _finish(store: IdentityStore, touched: set[int]) -> None:
    rebuild_prototypes(store.conn, store.vectors, store.cfg, sorted(touched))
    db.refresh_identity_stats(store.conn)
    store.save()


def recluster(
    conn: sqlite3.Connection,
    vectors: VectorStore,
    cfg: Config,
    cameras: CameraSet | None = None,
    restrict_gids: set[int] | None = None,
) -> ConsolidationResult:
    """Consolidate all identities in the data dir (UI button): merge, rebuild prototypes."""
    store = IdentityStore(cfg, conn, vectors, cameras)
    res = consolidate(store, restrict_gids)
    _finish(store, res.touched)
    return res


def merge_identities(
    conn: sqlite3.Connection, vectors: VectorStore, cfg: Config, a: int, b: int
) -> int:
    """Manually merge two identities (constraints ignored); returns the kept (oldest) gid."""
    if a == b:
        raise ValueError("cannot merge an identity with itself")
    for g in (a, b):
        if db.get_identity(conn, g) is None:
            raise KeyError(f"unknown identity {g}")
    store = IdentityStore(cfg, conn, vectors)
    kept = oldest_gid(conn, [a, b])
    absorbed = b if kept == a else a
    store.merge_gids(kept, absorbed, reason="manual merge")
    _finish(store, {kept})
    return kept


def split_tracklet(
    conn: sqlite3.Connection, vectors: VectorStore, cfg: Config, tracklet_id: int
) -> int:
    """Move one tracklet into a brand new identity; returns the new gid.

    A persistent cannot-link between old and new gid stops consolidation from re-merging them.
    """
    t = db.get_tracklet(conn, tracklet_id)
    if t is None:
        raise KeyError(f"unknown tracklet {tracklet_id}")
    if t.gid is None:
        raise ValueError(f"tracklet {tracklet_id} has no identity to split from")
    if len(db.get_tracklets(conn, gid=t.gid)) < 2:
        raise ValueError("cannot split the only tracklet of an identity")
    store = IdentityStore(cfg, conn, vectors)
    old, new = t.gid, store.mint()
    store.assign([(tracklet_id, new, None, "manual")], reason="manual split")
    db.add_cannot_link(conn, old, new, "manual split")
    store.rebuild_gallery(old)
    store.rebuild_gallery(new)
    _finish(store, {old, new})
    return new


def rename_identity(conn: sqlite3.Connection, gid: int, label: str | None) -> None:
    """Set the display label of an identity."""
    db.update_identity(conn, gid, label=label)
