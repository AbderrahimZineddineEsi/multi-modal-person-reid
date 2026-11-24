"""Query-by-image search over the identity space (PLAN section 3.4).

Replaces v1 ``search_person`` / ``search_face`` (Pinecone, v1_review 1.13-1.14) with queries on
the local ``body_proto`` (K prototypes per gid) and ``face_tracklet`` collections.

* **Body**: embed the query (or take a precomputed embedding), best prototype score per gid,
  UNKNOWN below ``body_min_cos``, then the Casao-style softmax ambiguity check on the top-2 gids:
  ``ratio = p1/p2 = exp(temp * (s1 - s2))``; ``ratio < tau`` is AMBIGUOUS. Deviation from v1: the
  temperature is the paper's 10 (v1 15); the check is otherwise the same top-2 test. If the
  prototypes were never built the tracklet descriptors are searched instead.
* **Face**: largest face of the query (or a pre-cropped face, gray-padded by ``face_pad`` inside
  ``FaceModel.embed_aligned``, v1's trick), best
  face-tracklet score per gid, UNKNOWN below ``face_min_cos``; deviation from v1 (no margin):
  AMBIGUOUS when the runner-up is within ``face_margin``.
* **Both**: a face MATCH wins; otherwise the body result is the fallback. v1 silently let the
  face override the body gid; here the returned ``modality`` says which one decided.

Queries are either an embedding (1-D array) or an image (H x W x 3 BGR array) plus a ``models``
object (``reid.models.registry.Models``-like: ``body.embed``, ``face.analyze``,
``face.embed_aligned``; if omitted, ``build_models(cfg)`` is imported lazily). Tests therefore
need no models.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Protocol

import numpy as np

from reid import analytics
from reid.config import Config
from reid.identity.gallery import l2n
from reid.storage.vectors import VectorStore
from reid.types import SearchHit, SearchStatus

log = logging.getLogger(__name__)


class QueryModels(Protocol):
    """What search needs from ``reid.models.registry.Models``: ``body.embed(crops)`` and, for
    faces, ``face.analyze(img)`` / ``face.embed_aligned(chip)`` (``face`` may be None)."""

    body: Any
    face: Any


def _models(models: Any, cfg: Config) -> Any:
    if models is not None:
        return models
    from reid.models.registry import build_models  # lazy: needs the [models] extra

    return build_models(cfg)


def _embed_body(models: Any, img: np.ndarray) -> np.ndarray:
    return np.asarray(models.body.embed([img])[0], dtype=np.float32)


def _embed_face(models: Any, img: np.ndarray, is_crop: bool) -> np.ndarray | None:
    """Embedding of the largest face of ``img`` (or of a pre-cropped face chip), else None."""
    if models.face is None:
        return None
    if is_crop:  # FaceModel.embed_aligned gray-pads by search.face_pad (v1 embed_face_crop)
        emb = models.face.embed_aligned(img)
    else:
        found = models.face.analyze(img)
        if not found:
            return None
        emb = max(found, key=lambda r: float((r[0][2] - r[0][0]) * (r[0][3] - r[0][1])))[3]
    return None if emb is None else np.asarray(emb, dtype=np.float32)


def _is_embedding(q: Any) -> bool:
    return isinstance(q, np.ndarray) and q.ndim == 1


def _finish(hit: SearchHit, conn: sqlite3.Connection) -> SearchHit:
    if hit.gid is not None and hit.status in (SearchStatus.MATCH, SearchStatus.AMBIGUOUS):
        hit.journey = analytics.journey(conn, hit.gid)
    return hit


def search_body(
    query: np.ndarray,
    conn: sqlite3.Connection,
    vectors: VectorStore,
    cfg: Config,
    *,
    models: Any = None,
) -> SearchHit:
    """Body search: MATCH, AMBIGUOUS (top-2 too close) or UNKNOWN."""
    s = cfg.search
    q = l2n(query.reshape(1, -1))[0] if _is_embedding(query) else None
    if q is None:
        q = l2n(_embed_body(_models(models, cfg), query).reshape(1, -1))[0]
    col = vectors.get("body_proto")
    if len(col) == 0:
        col = vectors.get("body_tracklet")
    hits = col.best_per_label(q, top_k=s.top_k) if len(col) else []
    cands = [(g, sc) for g, _k, sc in hits]
    if not hits or hits[0][2] < s.body_min_cos:
        return SearchHit(
            SearchStatus.UNKNOWN, None, hits[0][2] if hits else 0.0, "body", candidates=cands
        )
    gid, _k, s1 = hits[0]
    second_gid, s2, ratio = None, None, None
    status = SearchStatus.MATCH
    if len(hits) > 1:
        second_gid, s2 = hits[1][0], hits[1][2]
        ratio = float(np.exp(s.softmax_temp * (s1 - s2)))
        if ratio < s.tau:
            status = SearchStatus.AMBIGUOUS
    return _finish(
        SearchHit(status, gid, s1, "body", second_gid, s2, ratio, candidates=cands), conn
    )


def search_face(
    query: np.ndarray,
    conn: sqlite3.Connection,
    vectors: VectorStore,
    cfg: Config,
    *,
    models: Any = None,
    is_crop: bool = False,
) -> SearchHit:
    """Face search: MATCH, AMBIGUOUS (runner-up within margin), UNKNOWN or NO_FACE."""
    s = cfg.search
    if _is_embedding(query):
        q = l2n(query.reshape(1, -1))[0]
    else:
        emb = _embed_face(_models(models, cfg), query, is_crop)
        if emb is None:
            return SearchHit(SearchStatus.NO_FACE, modality="face")
        q = l2n(emb.reshape(1, -1))[0]
    col = vectors.get("face_tracklet")
    hits = col.best_per_label(q, top_k=s.face_top_k) if len(col) else []
    cands = [(g, sc) for g, _k, sc in hits]
    if not hits or hits[0][2] < s.face_min_cos:
        return SearchHit(
            SearchStatus.UNKNOWN, None, hits[0][2] if hits else 0.0, "face", candidates=cands
        )
    gid, _k, s1 = hits[0]
    second_gid, s2 = (hits[1][0], hits[1][2]) if len(hits) > 1 else (None, None)
    status = SearchStatus.MATCH
    if s2 is not None and s1 - s2 < s.face_margin:
        status = SearchStatus.AMBIGUOUS
    return _finish(SearchHit(status, gid, s1, "face", second_gid, s2, candidates=cands), conn)


def search_both(
    query: np.ndarray | None,
    conn: sqlite3.Connection,
    vectors: VectorStore,
    cfg: Config,
    *,
    models: Any = None,
    body_embedding: np.ndarray | None = None,
    face_embedding: np.ndarray | None = None,
    is_crop: bool = False,
) -> SearchHit:
    """Face first (a MATCH wins), body as the fallback. Pass an image as ``query`` or the two
    precomputed embeddings."""
    face_q = face_embedding if face_embedding is not None else query
    body_q = body_embedding if body_embedding is not None else query
    face: SearchHit | None = None
    if face_q is not None:
        face = search_face(face_q, conn, vectors, cfg, models=models, is_crop=is_crop)
        if face.status == SearchStatus.MATCH:
            return face
    if body_q is None:
        return face or SearchHit(SearchStatus.UNKNOWN)
    body = search_body(body_q, conn, vectors, cfg, models=models)
    if body.status == SearchStatus.UNKNOWN and face and face.status == SearchStatus.AMBIGUOUS:
        return face  # no body evidence: the face's best guess is more informative than UNKNOWN
    return body
