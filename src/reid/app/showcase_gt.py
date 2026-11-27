"""Match system identities to ground-truth ids, so a video can blur GT boxes of other people.

Showcase mode blurs the detector's boxes of everybody who is not allowed. People the detector
missed still show, so when the environment has a ``gt.csv`` (``camera,frame,gid,x1,y1,x2,y2``,
in the same pixels as the stored track points) its boxes are blurred too, except those of GT
ids that belong to allowed identities. A GT id is kept only when most of its matched boxes
(IoU >= 0.5 with a stored track point) went to an allowed gid; one without any match is blurred.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

IOU = 0.5
FRAME_STRIDE = 4  # votes from every 4th frame are plenty and keep a 16-camera scene quick


def gt_keep_ids(
    conn: sqlite3.Connection, gt_path: str | Path, allowed: frozenset[int]
) -> list[int]:
    """GT ids whose boxes belong to ``allowed`` gids (majority vote), sorted."""
    import pandas as pd

    from reid.eval.mot import iou_matrix

    gt = pd.read_csv(gt_path, dtype={"camera": str})
    gt = gt[gt["frame"] % FRAME_STRIDE == 0]
    pts = pd.DataFrame(
        conn.execute(
            "SELECT t.camera_id AS camera, p.frame, t.gid, p.x1, p.y1, p.x2, p.y2 "
            "FROM track_points p JOIN tracklets t ON t.id = p.tracklet_id"
        ).fetchall(),
        columns=["camera", "frame", "gid", "x1", "y1", "x2", "y2"],
    )
    pts = pts[pts["frame"] % FRAME_STRIDE == 0]
    hyp = {k: g for k, g in pts.groupby(["camera", "frame"], sort=False)}
    votes: dict[int, Counter] = defaultdict(Counter)
    for key, g in gt.groupby(["camera", "frame"], sort=False):
        h = hyp.get((str(key[0]), int(key[1])))
        if h is None:
            continue
        iou = iou_matrix(
            g[["x1", "y1", "x2", "y2"]].to_numpy(float), h[["x1", "y1", "x2", "y2"]].to_numpy(float)
        )
        best = iou.argmax(axis=1)
        hgid = h["gid"].to_numpy()
        for row, gt_id in enumerate(g["gid"].to_numpy()):
            if iou[row, best[row]] >= IOU:
                v = hgid[best[row]]
                votes[int(gt_id)][None if v is None or np.isnan(v) else int(v)] += 1
    keep = []
    for gt_id, c in votes.items():
        good = sum(n for gid, n in c.items() if gid is not None and gid in allowed)
        if good * 2 > sum(c.values()):
            keep.append(gt_id)
    log.info("showcase: %d of %d ground-truth ids are allowed", len(keep), gt["gid"].nunique())
    return sorted(keep)
