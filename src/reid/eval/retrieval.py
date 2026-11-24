"""Held-out query retrieval: Rank-1 / Rank-5 / mAP of body search over the identity space.

v1 evaluated retrieval on the crops the system itself had saved (circular, review W4). Here the
queries are GT boxes cropped from the video:

1. For every GT person, sample ``per_id`` boxes spread over cameras and time (large enough,
   not cut by the frame edge).
2. A GT box is *held out* if no stored body sample of any tracklet lies within ``guard`` frames
   of it on the same camera at IoU >= 0.3 (those are the views the gallery was built from).
   If a person has too few held-out boxes the remainder is filled with non-held-out ones and
   the report states the share of held-out queries.
3. Each crop is embedded and ranked against the gids (``body_proto``, else ``body_tracklet``,
   best prototype per gid). A gid's identity is its *majority GT id* (most matched boxes in the
   combined accumulator). A query is correct at rank k if one of the top-k gids has the query's
   GT id as majority; AP uses every gid whose majority is that GT id as relevant.
4. ``search_body`` (thresholded, with the ambiguity check) is run on the same embeddings: the
   share answered (MATCH/AMBIGUOUS) and the precision of those answers.

Queries of GT people that no gid represents (not the majority of any gid) cannot be retrieved;
they are reported as ``unretrievable`` and excluded from Rank-1/mAP (standard ReID practice), so
read the numbers together with the fragmentation report.
Limitation: a held-out frame still belongs to a tracklet that contributed other samples, so this
measures within-identity generalization to unseen views, not a fully disjoint gallery.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pandas as pd

from reid import search
from reid.config import Config
from reid.eval.mot import iou_matrix
from reid.storage.vectors import VectorStore

log = logging.getLogger(__name__)


def load_sampled_points(conn: sqlite3.Connection, run_id: int) -> pd.DataFrame:
    """Track points whose frame produced a stored body sample (``sampled = 1``)."""
    return pd.read_sql_query(
        "SELECT t.camera_id AS camera, p.frame AS frame, p.x1, p.y1, p.x2, p.y2 "
        "FROM track_points p JOIN tracklets t ON t.id = p.tracklet_id "
        "WHERE t.run_id = ? AND p.sampled = 1",
        conn,
        params=(run_id,),
    )


def _is_used(cands: pd.DataFrame, sampled: pd.DataFrame, guard: int) -> np.ndarray:
    """True for candidate boxes that overlap a stored sample of their camera within ``guard``."""
    used = np.zeros(len(cands), bool)
    for cam, idx in cands.groupby("camera").indices.items():
        s = sampled[sampled["camera"] == cam]
        if s.empty:
            continue
        c = cands.iloc[idx]
        near = np.abs(c["frame"].to_numpy()[:, None] - s["frame"].to_numpy()[None, :]) <= guard
        ious = iou_matrix(
            c[["x1", "y1", "x2", "y2"]].to_numpy(float), s[["x1", "y1", "x2", "y2"]].to_numpy(float)
        )
        used[idx] = (near & (ious >= 0.3)).any(axis=1)
    return used


def _spread(rows: pd.DataFrame, k: int) -> pd.DataFrame:
    """``k`` rows evenly spaced in time, taken round-robin over cameras."""
    cams = rows.groupby("camera").size().sort_values(ascending=False).index.tolist()
    quota = {c: k // len(cams) + (i < k % len(cams)) for i, c in enumerate(cams)} if cams else {}
    picks = []
    for cam, q in quota.items():
        r = rows[rows["camera"] == cam].sort_values("frame")
        picks.append(
            r.iloc[np.unique(np.linspace(0, len(r) - 1, min(q, len(r))).round().astype(int))]
        )
    return pd.concat(picks) if picks else rows.iloc[:0]


def sample_queries(
    gt: pd.DataFrame,
    sizes: dict[str, tuple[int, int]],
    sampled: pd.DataFrame,
    *,
    per_id: int = 5,
    min_height: float = 80.0,
    guard: int = 10,
    edge: int = 2,
) -> pd.DataFrame:
    """Choose query boxes from GT (see module docstring); adds a ``held_out`` column."""
    h = gt["y2"] - gt["y1"]
    w = gt["camera"].map(lambda c: sizes.get(c, (10**9, 10**9))[0])
    hh = gt["camera"].map(lambda c: sizes.get(c, (10**9, 10**9))[1])
    ok = (
        (h >= min_height)
        & (gt["x1"] > edge)
        & (gt["y1"] > edge)
        & (gt["x2"] < w - edge)
        & (gt["y2"] < hh - edge)
    )
    out = []
    for gid, g in gt[ok].groupby("gid"):
        if len(g) > 600:  # thin out before the IoU check
            g = g.iloc[np.linspace(0, len(g) - 1, 600).astype(int)]
        used = _is_used(g, sampled, guard)
        free = _spread(g[~used], per_id).assign(held_out=True)
        if len(free) < per_id:
            rest = _spread(g[used], per_id - len(free)).assign(held_out=False)
            free = pd.concat([free, rest])
        out.append(free.assign(gid=gid))
    return pd.concat(out, ignore_index=True) if out else gt.iloc[:0].assign(held_out=True)


def crop_queries(queries: pd.DataFrame, videos: dict[str, str]) -> list[np.ndarray | None]:
    """Tight crops of the query boxes, read sequentially per camera video (None if unreadable)."""
    crops: list[np.ndarray | None] = [None] * len(queries)
    pos = {i: p for p, i in enumerate(queries.index)}
    for cam, g in queries.groupby("camera"):
        cap = cv2.VideoCapture(str(videos[cam]))
        cur, frame = -1, None
        for idx, row in g.sort_values("frame").iterrows():
            f = int(row["frame"])
            if f != cur:
                while cur < f and cap.grab():
                    cur += 1
                ok, frame = cap.retrieve() if cur == f else (False, None)
                if not ok:
                    frame = None
            if frame is None:
                continue
            x1, y1 = max(0, int(row["x1"])), max(0, int(row["y1"]))
            c = frame[y1 : int(np.ceil(row["y2"])), x1 : int(np.ceil(row["x2"]))]
            crops[pos[idx]] = c.copy() if c.size else None
        cap.release()
    return crops


def _ap(relevant: np.ndarray) -> float:
    """Average precision of a ranked boolean relevance vector."""
    if not relevant.any():
        return 0.0
    prec = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
    return float(prec[relevant].mean())


def run_retrieval(
    conn: sqlite3.Connection,
    data_dir: Path,
    cfg: Config,
    run_id: int,
    gt: pd.DataFrame,
    majority: dict[int, int],
    videos: dict[str, str],
    sizes: dict[str, tuple[int, int]],
    *,
    per_id: int = 5,
    min_height: float = 80.0,
    guard: int = 10,
    models: Any = None,
) -> dict[str, Any]:
    """Run the held-out retrieval protocol and return the metrics dict (see module docstring)."""
    from reid.storage import files

    sampled = load_sampled_points(conn, run_id)
    q = sample_queries(gt, sizes, sampled, per_id=per_id, min_height=min_height, guard=guard)
    crops = crop_queries(q, videos)
    keep = [i for i, c in enumerate(crops) if c is not None]
    q, crops = q.iloc[keep].reset_index(drop=True), [crops[i] for i in keep]
    gallery_gts = set(majority.values())
    if models is None:
        from reid.models.body_reid import BodyReID

        models = SimpleNamespace(body=BodyReID(cfg), face=None)
    emb = models.body.embed(crops) if crops else np.zeros((0, 1), np.float32)
    vectors = VectorStore(files.vectors_dir(data_dir))
    col = vectors.get("body_proto")
    source = "body_proto"
    if len(col) == 0:
        col, source = vectors.get("body_tracklet"), "body_tracklet"
    res: dict[str, Any] = {
        "queries": len(q),
        "gt_ids_queried": int(q["gid"].nunique()) if len(q) else 0,
        "held_out_fraction": float(q["held_out"].mean()) if len(q) else None,
        "gallery_collection": source,
        "per_id": per_id,
        "min_query_height": min_height,
        "guard_frames": guard,
    }
    r1, r5, aps, answered, correct_answered, unretrievable = [], [], [], 0, 0, 0
    status = {"MATCH": 0, "AMBIGUOUS": 0, "UNKNOWN": 0}
    for i in range(len(q)):
        gid_true = int(q.loc[i, "gid"])
        hit = search.search_body(emb[i], conn, vectors, cfg)
        status[hit.status.value] = status.get(hit.status.value, 0) + 1
        if hit.status.value in ("MATCH", "AMBIGUOUS") and hit.gid is not None:
            answered += 1
            correct_answered += majority.get(hit.gid) == gid_true
        if gid_true not in gallery_gts:
            unretrievable += 1
            continue
        ranked = [g for g, _k, _s in col.best_per_label(emb[i])] if len(col) else []
        rel = np.array([majority.get(g) == gid_true for g in ranked], bool)
        r1.append(bool(rel[:1].any()))
        r5.append(bool(rel[:5].any()))
        aps.append(_ap(rel))
    n = len(r1)
    res.update(
        evaluated=n,
        unretrievable=unretrievable,
        rank1=float(np.mean(r1)) if n else None,
        rank5=float(np.mean(r5)) if n else None,
        mAP=float(np.mean(aps)) if n else None,
        search_status=status,
        search_answered=answered,
        search_answered_precision=(correct_answered / answered) if answered else None,
    )
    return res
