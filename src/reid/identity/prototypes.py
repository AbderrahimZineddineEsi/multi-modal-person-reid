"""Step 7: K prototypes per identity for search (``body_proto`` collection).

Two methods behind ``association.proto_method``:

``alpha_fps`` (in-house, v1_review 1.12, named after Al Muzaddid & Beksi 2025 but not their
formula): greedy farthest-point sampling seeded by the highest-quality sample, scoring
``alpha * q_n + (1 - alpha) * d_n``. Deviation from v1: ``q`` (quality) and ``d`` (cosine
distance to the nearest selected prototype) are min-max normalized to [0, 1] each step before
mixing; v1 added raw terms (q in 0.4-0.9 with tiny spread vs d in 0-0.3), so quality dominated and
the diversity term was nearly inert. Prototypes are real gallery samples.

``gcp`` (Al Muzaddid & Beksi, "Person Re-Identification via Generalized Class Prototypes",
arXiv:2510.17043, Algorithm 1): ``P = {centroid}``; repeatedly take the sample ``x`` farthest
from ``P``, remove it from ``X``, move it toward its closest prototype ``p`` with
``x <- x + alpha (p - x)`` and add it to ``P``. Deviations: the paper's loop yields ``N + 1``
points (centroid + N); we return exactly ``K`` (centroid + K-1 interpolated points) so both
methods give the same index size; cosine distance on L2-normalized features (the paper does not
fix the metric); interpolated points are re-normalized; ``alpha`` is not stated in the paper, we
default to 0.5 (``gcp_alpha``).
"""

from __future__ import annotations

import logging
import sqlite3

import numpy as np

from reid.config import AssociationCfg, Config
from reid.identity.gallery import l2n, weighted_mean
from reid.identity.store import PROTO_STRIDE
from reid.storage import db
from reid.storage.vectors import VectorStore

log = logging.getLogger(__name__)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.ones_like(x)


def alpha_fps_select(embs: np.ndarray, quality: np.ndarray, k: int, alpha: float) -> list[int]:
    """Indices of ``k`` prototypes (all samples if ``len(embs) <= k``), normalized terms."""
    n = len(embs)
    if n <= k:
        return list(range(n))
    e = l2n(embs)
    q = np.asarray(quality, dtype=np.float64)
    sel = [int(np.argmax(q))]
    dist = 1.0 - e @ e[sel[0]]
    for _ in range(k - 1):
        comb = alpha * _minmax(q) + (1.0 - alpha) * (dist / max(float(dist.max()), 1e-12))
        comb[sel] = -np.inf
        nxt = int(np.argmax(comb))
        sel.append(nxt)
        dist = np.minimum(dist, 1.0 - e @ e[nxt])
    return sel


def gcp_prototypes(
    embs: np.ndarray, k: int, alpha: float, weights: np.ndarray | None = None
) -> np.ndarray:
    """Al Muzaddid & Beksi Algorithm 1: centroid + ``k-1`` interpolated farthest points."""
    x = l2n(embs)
    protos = [weighted_mean(x, weights)]
    remaining = list(range(len(x)))
    while len(protos) < k and remaining:
        p = np.stack(protos)
        d = 1.0 - x[remaining] @ p.T  # (r, |P|)
        far = int(np.argmax(d.min(axis=1)))  # farthest from the set P
        nearest = int(np.argmin(d[far]))  # its closest prototype
        xi = x[remaining.pop(far)]
        protos.append(l2n(xi + alpha * (p[nearest] - xi)))
    return np.stack(protos)


def compute_prototypes(embs: np.ndarray, quality: np.ndarray, cfg: AssociationCfg) -> np.ndarray:
    """Prototype matrix ``(<=K, D)`` for one identity using ``cfg.proto_method``."""
    if len(embs) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if cfg.proto_method == "gcp":
        return gcp_prototypes(embs, cfg.proto_k, cfg.gcp_alpha, quality)
    sel = alpha_fps_select(embs, quality, cfg.proto_k, cfg.proto_alpha)
    return l2n(embs)[sel]


def rebuild_prototypes(
    conn: sqlite3.Connection,
    vectors: VectorStore,
    cfg: Config,
    gids: list[int] | None = None,
) -> int:
    """(Re)build ``body_proto`` for ``gids`` (all identities if None); returns identities done.

    Source samples are the identity's gallery members (all its samples if the gallery is
    empty). Identities that no longer have tracklets lose their prototypes. Idempotent.
    """
    col = vectors.get("body_proto")
    samples = vectors.get("body_samples")
    live = {
        r["gid"] for r in conn.execute("SELECT DISTINCT gid FROM tracklets WHERE gid IS NOT NULL")
    }
    targets = sorted(live) if gids is None else sorted(set(gids))
    for g in {int(x) for x in col.labels if x != -1} - live:
        col.remove_label(g)
    done = 0
    for gid in targets:
        col.remove_label(gid)
        if gid not in live:
            continue
        rows = db.get_samples(conn, gid=gid, in_gallery=True) or db.get_samples(conn, gid=gid)
        if not rows:
            continue
        ids = [r["id"] for r in rows]
        quality = np.array([r["quality"] for r in rows], dtype=np.float32)
        protos = compute_prototypes(samples.vectors(ids), quality, cfg.association)
        col.add([gid * PROTO_STRIDE + i for i in range(len(protos))], protos, [gid] * len(protos))
        done += 1
    return done
