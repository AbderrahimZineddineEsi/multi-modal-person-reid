"""Annotate tracklets with attribute probabilities (frozen body embeddings -> trained head).

Every stored body sample of a tracklet is already an embedding of the body ReID model, so the head
(:mod:`reid.attributes.head`, NumPy inference) is applied to the vectors in the ``body_samples``
collection: no video, no GPU, seconds per environment. A tracklet's attributes are the
quality-weighted mean of its samples' probabilities (the "voting over frames" the head report
measures), stored in ``tracklet_attributes`` together with the number of samples.

A head only fits embeddings of the model it was trained on: a head whose ``body_model`` /
``neck_feat`` / input size differs from the environment's body model is ignored (lite-preset
environments use OSNet, so they get no attributes) and the rest of the system is unaffected.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reid.attributes.head import HEAD_FILE, HeadParams, head_file, load_head, predict
from reid.config import Config
from reid.storage import attributes_db as adb
from reid.storage import db, files, portable
from reid.storage.vectors import VectorStore

log = logging.getLogger(__name__)
_CHUNK = 500  # ids per SQL query
_PREDICT_BATCH = 8192  # vectors per head call (8192 x 1280 floats = 42 MB)


# --------------------------------------------------------------------------- resolving the head


def default_head_path(cfg: Config) -> Path:
    """``<weights cache>/attributes/head.npz`` (where ``reid attributes use-head`` installs)."""
    from reid.models.device import weights_dir

    return weights_dir(cfg.paths.weights_dir) / "attributes" / HEAD_FILE


def configured_head_path(cfg: Config) -> Path:
    """The head file ``cfg.attributes.head`` points to (or the default cache location)."""
    return head_file(cfg.attributes.head) if cfg.attributes.head else default_head_path(cfg)


def incompatibility(head: HeadParams, body_model: str, neck_feat: str, dim: int | None) -> str:
    """Why ``head`` cannot read embeddings of this body model ('' when it can)."""
    if head.body_model and head.body_model != body_model:
        return f"head trained on {head.body_model}, environment uses {body_model}"
    if head.neck_feat and head.neck_feat != neck_feat:
        return f"head trained on neck_feat={head.neck_feat}, environment uses {neck_feat}"
    if dim is not None and head.in_dim != dim:
        return f"head expects {head.in_dim}-D embeddings, environment has {dim}-D"
    return ""


def load_configured_head(cfg: Config, *, warn: bool = True) -> HeadParams | None:
    """The head file ``cfg`` points to, whatever body model it was trained on (None if absent).

    ``attributes.enabled = false`` also gives None. Use :func:`resolve_head` to additionally
    require that it fits ``cfg.body_reid``.
    """
    if not cfg.attributes.enabled:
        return None
    path = configured_head_path(cfg)
    if not path.is_file():
        return None
    try:
        return load_head(path)
    except (OSError, ValueError, KeyError) as exc:
        if warn:
            log.warning("attribute head %s is unreadable (%s): attributes disabled", path, exc)
        return None


def resolve_head(cfg: Config, *, warn: bool = True) -> HeadParams | None:
    """The configured head if it exists and fits ``cfg.body_reid``, else None.

    A missing head is silent (attributes are optional); a head for another body model logs ONE
    warning per call (``warn=False`` silences it).
    """
    head = load_configured_head(cfg, warn=warn)
    if head is None:
        return None
    why = incompatibility(head, cfg.body_reid.model, cfg.body_reid.neck_feat, cfg.body_reid.dim)
    if why:
        if warn:
            log.warning("attribute head ignored (%s): no attributes for this run", why)
        return None
    return head


# --------------------------------------------------------------------------- annotating


def _sample_rows(conn: sqlite3.Connection, tids: Sequence[int] | None) -> list[sqlite3.Row]:
    """``(id, tracklet_id, quality)`` of the samples of ``tids`` (all tracklets when None)."""
    if tids is None:
        return conn.execute("SELECT id, tracklet_id, quality FROM samples ORDER BY id").fetchall()
    out: list[sqlite3.Row] = []
    for k in range(0, len(tids), _CHUNK):
        chunk = [int(t) for t in tids[k : k + _CHUNK]]
        marks = ",".join("?" * len(chunk))
        out += conn.execute(
            f"SELECT id, tracklet_id, quality FROM samples WHERE tracklet_id IN ({marks}) "
            "ORDER BY id",
            chunk,
        )
    return out


def annotate(
    conn: sqlite3.Connection,
    vectors: VectorStore,
    head: HeadParams,
    tracklet_ids: Sequence[int] | None = None,
    force: bool = False,
) -> int:
    """Annotate tracklets with ``head``; returns how many were written.

    ``tracklet_ids=None`` means every tracklet. Tracklets already annotated by this head are
    skipped unless ``force``; tracklets without samples (or whose sample vectors are missing)
    are skipped. All vectors go through the head in a few batched calls.
    """
    rec = adb.upsert_head(conn, head)
    todo = None if tracklet_ids is None else sorted({int(t) for t in tracklet_ids})
    done = set() if force else adb.annotated_tracklet_ids(conn, rec.id)
    col = vectors.get("body_samples")
    if len(col) and col.dim != head.in_dim:
        log.warning(
            "body_samples are %s-D but the head expects %d-D: skipped", col.dim, head.in_dim
        )
        return 0
    have = {int(k) for k in col.keys}
    by_tracklet: dict[int, list[tuple[int, float]]] = {}
    for r in _sample_rows(conn, todo):
        if r["tracklet_id"] not in done and r["id"] in have:
            by_tracklet.setdefault(r["tracklet_id"], []).append((r["id"], float(r["quality"])))
    if not by_tracklet:
        return 0
    sids = [sid for rows in by_tracklet.values() for sid, _ in rows]
    probs = np.concatenate(
        [
            predict(head, col.vectors(sids[k : k + _PREDICT_BATCH]))
            for k in range(0, len(sids), _PREDICT_BATCH)
        ]
    )
    out, pos = [], 0
    for tid, rows in by_tracklet.items():
        p = probs[pos : pos + len(rows)]
        pos += len(rows)
        w = np.maximum(np.array([q for _, q in rows], np.float64), 1e-6)
        out.append((tid, (p * (w / w.sum())[:, None]).sum(0).astype(np.float32), len(rows)))
    return adb.upsert_tracklet_attributes(conn, rec.id, out)


def annotate_camera(
    conn: sqlite3.Connection, vectors: VectorStore, head: HeadParams, run_id: int, camera_id: str
) -> int:
    """Annotate the tracklets of one camera of one run (the pipeline hook); never raises."""
    try:
        ids = [t.id for t in db.get_tracklets(conn, run_id=run_id, camera_id=camera_id)]
        return annotate(conn, vectors, head, [i for i in ids if i is not None])
    except Exception as exc:  # annotation must never fail a run
        log.warning("attribute annotation of camera %s failed: %s", camera_id, exc)
        return 0


# --------------------------------------------------------------------------- environments


@dataclass
class BackfillResult:
    """Outcome of annotating one environment."""

    env: str
    state: str  # annotated | skipped | empty
    annotated: int = 0  # tracklets written now
    total: int = 0  # tracklets annotated by this head after the call
    n_tracklets: int = 0
    seconds: float = 0.0
    reason: str = ""


def env_body_settings(data_dir: str | Path) -> tuple[str, str, int | None] | None:
    """``(model, neck_feat, dim)`` of the environment's latest run, None if it has no run."""
    ov = portable.env_model_overrides(data_dir).get("body_reid")
    return None if not ov else (ov["model"], ov["neck_feat"], int(ov["dim"]))


def backfill_env(data_dir: str | Path, head: HeadParams, force: bool = False) -> BackfillResult:
    """Annotate every tracklet of one environment (what ``reid attributes backfill`` runs).

    Environments processed with another body model (lite preset) are skipped with the reason.
    """
    root = Path(data_dir)
    t0 = time.perf_counter()
    res = BackfillResult(root.name, "skipped")
    settings = env_body_settings(root)
    if settings is not None:
        res.reason = incompatibility(head, *settings)
        if res.reason:
            return res
    conn = db.connect(files.db_path(root))
    try:
        res.n_tracklets = int(conn.execute("SELECT COUNT(*) FROM tracklets").fetchone()[0])
        if res.n_tracklets == 0:
            res.state, res.reason = "empty", "no tracklets"
            return res
        res.annotated = annotate(conn, VectorStore(files.vectors_dir(root)), head, None, force)
        res.total = adb.count_annotated(conn, adb.upsert_head(conn, head).id)
        res.state, res.reason = "annotated", ""
    finally:
        conn.close()
        res.seconds = time.perf_counter() - t0
    return res


def backfill_workspace(
    workspace: str | Path, head: HeadParams, force: bool = False
) -> list[BackfillResult]:
    """:func:`backfill_env` on every environment directly under ``workspace``."""
    return [backfill_env(e.path, head, force) for e in portable.list_envs(Path(workspace))]


@dataclass
class EnvStatus:
    """What the Find tab needs to know about an environment (read-only, cheap)."""

    state: str  # ready | not_annotated | incompatible | no_head | no_tracklets
    n_tracklets: int = 0
    n_annotated: int = 0
    reason: str = ""


def env_status(
    conn: sqlite3.Connection, data_dir: str | Path, head: HeadParams | None
) -> EnvStatus:
    """Whether an environment can be searched by attributes.

    ``head`` is the configured head (:func:`load_configured_head`; None if there is none). An
    environment that already carries annotations is ``ready`` even without a head file, because
    the labels and thresholds are stored in the database.
    """
    n = int(conn.execute("SELECT COUNT(*) FROM tracklets").fetchone()[0])
    rec = adb.latest_annotated_head(conn)
    n_ann = adb.count_annotated(conn, rec.id) if rec else 0
    if n == 0:
        return EnvStatus("no_tracklets")
    if n_ann:
        return EnvStatus("ready", n, n_ann)
    settings = env_body_settings(data_dir)
    if head is None:
        return EnvStatus("no_head", n, 0, "no attribute head installed (reid attributes use-head)")
    why = incompatibility(head, *settings) if settings else ""
    if why:
        return EnvStatus(
            "incompatible", n, 0, f"attributes unavailable for this environment: {why}"
        )
    return EnvStatus("not_annotated", n, 0)
