"""Repository functions of the attribute tables (schema v3): heads and per-tracklet probabilities.

``attribute_heads`` has one row per distinct head (identified by :func:`head_sha256`, a hash of the
weights and labels, not of the file bytes, so re-saving an identical head does not create a new
row). Its labels, groups and tuned thresholds are copied into the row, so an environment moved to
another machine can still be searched without the ``head.npz``. ``tracklet_attributes`` holds the
head's probabilities of a tracklet as a float32 BLOB. Both tables are plain SQL, no ORM, like
:mod:`reid.storage.db`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from reid.attributes.head import HeadParams
from reid.storage.db import transaction

_CHUNK = 500  # stay below SQLite's variable limit


@dataclass
class HeadRecord:
    """One ``attribute_heads`` row with decoded JSON columns."""

    id: int
    name: str
    sha256: str
    body_model: str
    neck_feat: str
    labels: list[str]
    groups: list[str]
    thresholds: np.ndarray  # (labels,) tuned on validation
    trained: dict = field(default_factory=dict)  # {"trained_labels": [...], "best_val_ma": ...}
    created_at: float = 0.0

    @property
    def index(self) -> dict[str, int]:
        """Label name -> column of the probability vectors."""
        return {n: i for i, n in enumerate(self.labels)}


def head_sha256(head: HeadParams) -> str:
    """Hash of a head's weights, labels, thresholds and body-model settings (not of the file)."""
    h = hashlib.sha256()
    for name in ("W1", "b1", "bn_mean", "bn_var", "bn_weight", "bn_bias", "W2", "b2", "thresholds"):
        h.update(np.ascontiguousarray(getattr(head, name), np.float32).tobytes())
    h.update(json.dumps([head.labels, head.groups, head.body_model, head.neck_feat]).encode())
    return h.hexdigest()


def _record(row: sqlite3.Row) -> HeadRecord:
    return HeadRecord(
        id=row["id"],
        name=row["name"],
        sha256=row["sha256"],
        body_model=row["body_model"],
        neck_feat=row["neck_feat"],
        labels=json.loads(row["labels_json"]),
        groups=json.loads(row["groups_json"]),
        thresholds=np.asarray(json.loads(row["thresholds_json"]), np.float32),
        trained=json.loads(row["trained_json"]),
        created_at=row["created_at"],
    )


def upsert_head(conn: sqlite3.Connection, head: HeadParams, name: str | None = None) -> HeadRecord:
    """Register ``head`` (no-op if its sha is known) and return its row."""
    sha = head_sha256(head)
    row = conn.execute("SELECT * FROM attribute_heads WHERE sha256=?", (sha,)).fetchone()
    if row is None:
        trained = {
            "trained_labels": [
                str(n) for n, t in zip(head.labels, head.trained_labels, strict=True) if t
            ],
            "best_val_ma": head.hparams.get("best_val_ma"),
        }
        conn.execute(
            "INSERT INTO attribute_heads(name, sha256, body_model, neck_feat, labels_json, "
            "groups_json, thresholds_json, trained_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                name or f"{head.body_model}-{sha[:8]}",
                sha,
                head.body_model,
                head.neck_feat,
                json.dumps(head.labels),
                json.dumps(head.groups),
                json.dumps([float(x) for x in head.thresholds]),
                json.dumps(trained),
                time.time(),
            ),
        )
        row = conn.execute("SELECT * FROM attribute_heads WHERE sha256=?", (sha,)).fetchone()
    return _record(row)


def get_head(conn: sqlite3.Connection, head_id: int) -> HeadRecord | None:
    row = conn.execute("SELECT * FROM attribute_heads WHERE id=?", (head_id,)).fetchone()
    return _record(row) if row else None


def head_by_sha(conn: sqlite3.Connection, sha: str) -> HeadRecord | None:
    """The head row with this sha (the 'active head' of a configured head file), or None."""
    row = conn.execute("SELECT * FROM attribute_heads WHERE sha256=?", (sha,)).fetchone()
    return _record(row) if row else None


def latest_annotated_head(conn: sqlite3.Connection) -> HeadRecord | None:
    """Most recent head (by creation) that has at least one annotated tracklet."""
    row = conn.execute(
        "SELECT h.* FROM attribute_heads h WHERE EXISTS "
        "(SELECT 1 FROM tracklet_attributes a WHERE a.head_id=h.id) "
        "ORDER BY h.created_at DESC, h.id DESC LIMIT 1"
    ).fetchone()
    return _record(row) if row else None


def upsert_tracklet_attributes(
    conn: sqlite3.Connection, head_id: int, rows: Iterable[tuple[int, np.ndarray, int]]
) -> int:
    """Insert or replace ``(tracklet_id, probs, n_samples)`` rows in one transaction."""
    data = [(int(t), head_id, np.asarray(p, np.float32).tobytes(), int(n)) for t, p, n in rows]
    if data:
        with transaction(conn):
            conn.executemany(
                "INSERT OR REPLACE INTO tracklet_attributes(tracklet_id, head_id, probs, "
                "n_samples) VALUES (?,?,?,?)",
                data,
            )
    return len(data)


def annotated_tracklet_ids(conn: sqlite3.Connection, head_id: int) -> set[int]:
    """Tracklets that already have attributes from this head."""
    return {
        r[0]
        for r in conn.execute(
            "SELECT tracklet_id FROM tracklet_attributes WHERE head_id=?", (head_id,)
        )
    }


def count_annotated(conn: sqlite3.Connection, head_id: int | None = None) -> int:
    """Number of annotated tracklets (of one head, or of any head when ``head_id`` is None)."""
    if head_id is None:
        sql, args = "SELECT COUNT(DISTINCT tracklet_id) FROM tracklet_attributes", ()
    else:
        sql, args = "SELECT COUNT(*) FROM tracklet_attributes WHERE head_id=?", (head_id,)
    return int(conn.execute(sql, args).fetchone()[0])


def read_tracklet_attributes(
    conn: sqlite3.Connection,
    head_id: int,
    n_labels: int,
    tracklet_ids: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(tracklet_ids int64 (N,), probs float32 (N, n_labels), n_samples int64 (N,))``.

    All annotated tracklets of the head, ordered by id, or only ``tracklet_ids`` (those that
    are annotated).
    """
    rows: list[sqlite3.Row] = []
    base = "SELECT tracklet_id, probs, n_samples FROM tracklet_attributes WHERE head_id=?"
    if tracklet_ids is None:
        rows = conn.execute(base + " ORDER BY tracklet_id", (head_id,)).fetchall()
    else:
        ids = [int(i) for i in tracklet_ids]
        for k in range(0, len(ids), _CHUNK):
            chunk = ids[k : k + _CHUNK]
            marks = ",".join("?" * len(chunk))
            rows += conn.execute(f"{base} AND tracklet_id IN ({marks})", (head_id, *chunk))
        rows.sort(key=lambda r: r["tracklet_id"])
    ids_a = np.fromiter((r["tracklet_id"] for r in rows), np.int64, count=len(rows))
    n_a = np.fromiter((r["n_samples"] for r in rows), np.int64, count=len(rows))
    probs = np.zeros((len(rows), n_labels), np.float32)
    for i, r in enumerate(rows):
        probs[i] = np.frombuffer(r["probs"], np.float32)
    return ids_a, probs, n_a
