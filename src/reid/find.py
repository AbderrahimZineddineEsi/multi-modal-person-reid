"""Attribute filter search over annotated tracklets ("find a woman in red with a backpack").

Pure functions over the database (no Gradio, no models): the probabilities written by
:mod:`reid.attributes.annotate` are ranked against a :class:`FindQuery`.

* **Score** of a tracklet: the mean over the query terms of a log-likelihood. A required label
  contributes ``log p``, a forbidden one ``log(1 - p)`` and an any-of group (e.g. upper colour red
  OR orange) ``log(max p)``; p is clipped to [1e-4, 1 - 1e-4]. ``match_pct = 100 * exp(score)`` is
  the geometric mean of the term probabilities, a 0-100 number that is easy to read.
* **Ranked mode** (default): every tracklet that passes the camera and time filters is returned,
  best first, so a rare combination still shows its closest candidates. **Strict mode**: only
  tracklets where every term passes its label's tuned threshold (stored with the head; an any-of
  group passes when one member does). ``Hit.passes`` is that test in both modes, and
  ``FindResult.n_matching`` counts it, so the header count means the same thing either way.
* **People**: :func:`group_by_person` collapses appearances by identity; the person score is the
  best tracklet's score (a person who matches once matches). Tracklets without an identity stay
  separate entries.

Heavy work is one vectorised pass over a cached :class:`AttributeTable` (``load_table``); a
query on 2,000 tracklets takes a few milliseconds. Callers that issue many queries (the app) keep
the table and pass it to :func:`find`; reload it after identity edits or annotation.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from reid.storage import attributes_db as adb

_EPS = 1e-4


@dataclass
class FindQuery:
    """What to look for. Labels are the head's label names (``reid.attributes.schema.LABELS``)."""

    cameras: list[str] | None = None  # None = all cameras
    t_from: float | None = None  # keep tracklets overlapping [t_from, t_to] (epoch seconds)
    t_to: float | None = None
    require: dict[str, bool] = field(default_factory=dict)  # label -> must have / must not have
    any_of: list[list[str]] = field(default_factory=list)  # each group: at least one of these
    strict: bool = False
    limit_people: int | None = None  # keep only the N best people (or unidentified tracklets)

    @property
    def n_terms(self) -> int:
        return len(self.require) + len(self.any_of)


@dataclass
class Hit:
    """One tracklet that matched (or, in ranked mode, was ranked)."""

    tracklet_id: int
    gid: int | None
    camera_id: str
    start_ts: float
    end_ts: float
    score: float  # mean log-likelihood of the query terms (0 without terms)
    match_pct: float  # 100 * exp(score)
    passes: bool  # every term passes its label threshold (the strict-mode test)
    probs: np.ndarray  # float32 (n_labels,) in head label order
    n_samples: int = 0
    crop_path: str | None = None  # best crop, relative to data_dir


@dataclass
class FindResult:
    """Hits best first plus the counts for the header ("12 people, 37 appearances of 1,787")."""

    head: adb.HeadRecord | None
    total_tracklets: int  # annotated tracklets of the environment (the searchable ones)
    n_in_scope: int  # ... that pass the camera and time filters
    n_matching: int  # ... that also pass every term's threshold (strict-mode test)
    hits: list[Hit]  # ranked: all in scope; strict: only the passing ones; best first


@dataclass
class PersonHit:
    """All appearances of one person among the hits (``gid`` None = one unidentified tracklet)."""

    gid: int | None
    best: Hit
    hits: list[Hit]  # best first
    cameras: list[str]  # in order of first appearance
    score: float  # = best.score


@dataclass
class AttributeTable:
    """Attribute probabilities and tracklet metadata of one environment, loaded once.

    Row ``i`` of every array is the same tracklet. ``gids`` and the other metadata are as of
    load time: reload after identity edits (merge / split / delete) or new annotations.
    """

    head: adb.HeadRecord
    tracklet_ids: np.ndarray
    probs: np.ndarray  # float32 (N, n_labels)
    n_samples: np.ndarray
    camera_ids: np.ndarray  # str (N,)
    start_ts: np.ndarray
    end_ts: np.ndarray
    gids: list[int | None]
    crop_paths: list[str | None]
    total_tracklets: int  # all tracklets of the database, annotated or not

    def __len__(self) -> int:
        return len(self.tracklet_ids)


def load_table(conn: sqlite3.Connection, head_id: int | None = None) -> AttributeTable | None:
    """The attribute table of ``head_id`` (default: the latest head with annotations), or None."""
    head = adb.get_head(conn, head_id) if head_id is not None else adb.latest_annotated_head(conn)
    if head is None:
        return None
    ids, probs, n_samples = adb.read_tracklet_attributes(conn, head.id, len(head.labels))
    meta = {
        r["id"]: r
        for r in conn.execute(
            "SELECT id, camera_id, start_ts, end_ts, gid, best_crop_path FROM tracklets"
        )
    }
    rows = [meta[int(i)] for i in ids]  # FK guarantees presence
    total = int(conn.execute("SELECT COUNT(*) FROM tracklets").fetchone()[0])
    return AttributeTable(
        head=head,
        tracklet_ids=ids,
        probs=probs,
        n_samples=n_samples,
        camera_ids=np.array([r["camera_id"] for r in rows], dtype=str),
        start_ts=np.array([r["start_ts"] for r in rows], np.float64),
        end_ts=np.array([r["end_ts"] for r in rows], np.float64),
        gids=[r["gid"] for r in rows],
        crop_paths=[r["best_crop_path"] for r in rows],
        total_tracklets=total,
    )


def _col(index: dict[str, int], label: str) -> int:
    if label not in index:
        raise ValueError(f"unknown attribute {label!r}")
    return index[label]


def _scope(table: AttributeTable, q: FindQuery) -> np.ndarray:
    """Boolean mask of tracklets passing the camera and time filters."""
    mask = np.ones(len(table), bool)
    if q.cameras is not None:
        mask &= np.isin(table.camera_ids, list(q.cameras))
    if q.t_from is not None:
        mask &= table.end_ts >= q.t_from
    if q.t_to is not None:
        mask &= table.start_ts <= q.t_to
    return mask


def score_table(table: AttributeTable, q: FindQuery) -> tuple[np.ndarray, np.ndarray]:
    """``(score (N,), passes (N,))`` of every tracklet against the query terms."""
    n = len(table)
    if q.n_terms == 0:
        return np.zeros(n), np.ones(n, bool)
    index, thr = table.head.index, table.head.thresholds
    p = table.probs.astype(np.float64)
    pc = np.clip(p, _EPS, 1 - _EPS)
    terms: list[np.ndarray] = []
    ok = np.ones(n, bool)
    for label, want in q.require.items():
        j = _col(index, label)
        terms.append(np.log(pc[:, j]) if want else np.log1p(-pc[:, j]))
        ok &= (p[:, j] >= thr[j]) if want else (p[:, j] < thr[j])
    for group in q.any_of:
        if not group:
            raise ValueError("an any-of group needs at least one label")
        js = [_col(index, label) for label in group]
        terms.append(np.log(pc[:, js].max(1)))
        ok &= (p[:, js] >= thr[js]).any(1)
    return np.mean(terms, axis=0), ok


def find(
    conn: sqlite3.Connection, query: FindQuery, table: AttributeTable | None = None
) -> FindResult:
    """Rank the annotated tracklets of an environment against ``query``.

    ``table`` is a cached :func:`load_table` result (loaded here when omitted). Without any
    annotation the result is empty with ``head=None``. Raises ``ValueError`` for an unknown label.
    Without attribute terms the hits are ordered by time. ``limit_people`` keeps the hits of the
    N best people (an unidentified tracklet counts as one person).
    """
    table = table if table is not None else load_table(conn)
    if table is None:
        total = int(conn.execute("SELECT COUNT(*) FROM tracklets").fetchone()[0])
        return FindResult(None, total, 0, 0, [])
    scope = _scope(table, query)
    score, passes = score_table(table, query)
    n_matching = int((scope & passes).sum())
    keep = scope & passes if query.strict else scope
    rows = np.flatnonzero(keep)
    order = rows[np.lexsort((table.tracklet_ids[rows], table.start_ts[rows], -score[rows]))]
    hits = [_hit(table, int(i), float(score[i]), bool(passes[i])) for i in order]
    if query.limit_people is not None:
        hits = _limit_people(hits, query.limit_people)
    return FindResult(table.head, len(table), int(scope.sum()), n_matching, hits)


def _hit(table: AttributeTable, i: int, score: float, passes: bool) -> Hit:
    return Hit(
        tracklet_id=int(table.tracklet_ids[i]),
        gid=table.gids[i],
        camera_id=str(table.camera_ids[i]),
        start_ts=float(table.start_ts[i]),
        end_ts=float(table.end_ts[i]),
        score=score,
        match_pct=100.0 * float(np.exp(score)),
        passes=passes,
        probs=table.probs[i],
        n_samples=int(table.n_samples[i]),
        crop_path=table.crop_paths[i],
    )


def _person_key(h: Hit) -> tuple[str, int]:
    return ("g", h.gid) if h.gid is not None else ("t", h.tracklet_id)


def _limit_people(hits: list[Hit], n: int) -> list[Hit]:
    keys: list[tuple[str, int]] = []
    for h in hits:  # hits are best first, so first sight order = person ranking
        k = _person_key(h)
        if k not in keys:
            keys.append(k)
    allowed = set(keys[: max(n, 0)])
    return [h for h in hits if _person_key(h) in allowed]


def group_by_person(hits: Sequence[Hit]) -> list[PersonHit]:
    """Collapse hits by identity, best person first (score = best tracklet's score).

    Tracklets with ``gid`` None are never merged: each becomes its own "unidentified" entry.
    """
    groups: dict[tuple[str, int], list[Hit]] = {}
    for h in hits:
        groups.setdefault(_person_key(h), []).append(h)
    out = []
    for members in groups.values():
        members = sorted(members, key=lambda h: (-h.score, h.start_ts, h.tracklet_id))
        cams: list[str] = []
        for h in sorted(members, key=lambda h: h.start_ts):
            if h.camera_id not in cams:
                cams.append(h.camera_id)
        out.append(PersonHit(members[0].gid, members[0], members, cams, members[0].score))
    out.sort(key=lambda p: (-p.score, p.best.start_ts, p.best.tracklet_id))
    return out


def person_attributes(
    conn: sqlite3.Connection, gid: int, table: AttributeTable | None = None
) -> dict[str, float]:
    """Attribute probabilities of an identity: mean over its tracklets weighted by n_samples.

    Empty dict when none of its tracklets is annotated. Uses ``table`` if given (gids as of its
    load time), else the database.
    """
    if table is not None:
        rows = [i for i, g in enumerate(table.gids) if g == gid]
        head, probs, w = table.head, table.probs[rows], table.n_samples[rows]
    else:
        head = adb.latest_annotated_head(conn)
        if head is None:
            return {}
        ids = [r[0] for r in conn.execute("SELECT id FROM tracklets WHERE gid=?", (gid,))]
        _, probs, w = adb.read_tracklet_attributes(conn, head.id, len(head.labels), ids)
    if len(probs) == 0:
        return {}
    wf = np.maximum(w.astype(np.float64), 1.0)
    mean = (probs * (wf / wf.sum())[:, None]).sum(0)
    return {label: float(v) for label, v in zip(head.labels, mean, strict=True)}
