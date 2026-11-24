"""MOT-style metrics on stored track points, via ``motmetrics``.

Protocols (all use IoU >= ``iou`` for a box to match, one match per GT box):

* **Per camera** (``per_camera``): one accumulator per camera. Hypothesis id = the system gid of
  the box's tracklet; an unidentified tracklet (gid NULL) gets a unique negative id, so it never
  helps an identity. ``per_camera_tracklet`` uses the tracklet id instead of the gid: that
  isolates the tracker (detection + tracking) from the identity association.
* **CityFlow-style combined IDF1** (``mtmc``): ONE accumulator over all cameras. Every
  (camera, frame) pair is a distinct frame key (``camera_index * 10**7 + frame``, cameras in
  sorted order, so a person walking from one camera to the next is a continuation); GT ids are
  global and hypothesis ids are the global gids. IDF1 then needs a single gid for a GT person
  across *all* cameras: a person split into one gid per camera loses recall and precision, two
  people merged into one gid does too. This follows the idea of the CityFlow / AI City
  cross-camera protocol (global identity consistency) but is IDF1 on 2D boxes, not the official
  3D-location HOTA, so numbers are not comparable with the leaderboard.
* **Fragmentation** (:func:`fragmentation`): from the matched (GT id, gid) box counts of the
  combined accumulator.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import motmetrics as mm
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FRAME_SPAN = 10**7  # frame-key stride between cameras in the combined accumulator
_METRICS = [
    "idf1", "idp", "idr", "mota", "num_switches", "precision", "recall", "num_false_positives",
    "num_misses", "num_matches", "num_objects", "num_detections", "num_unique_objects",
    "mostly_tracked", "mostly_lost", "num_fragmentations",
]  # fmt: skip
_RENAME = {"num_switches": "id_switches", "num_false_positives": "fp", "num_misses": "fn"}


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of ``(N, 4)`` and ``(M, 4)`` xyxy boxes -> ``(N, M)``."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _groups(df: pd.DataFrame, idcol: str) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    out = {}
    for key, g in df.groupby(["camera", "frame"], sort=False):
        out[(str(key[0]), int(key[1]))] = (
            g[idcol].to_numpy(),
            g[["x1", "y1", "x2", "y2"]].to_numpy(float),
        )
    return out


def build_accumulator(
    gt: pd.DataFrame,
    hyp: pd.DataFrame,
    iou: float,
    cameras: list[str],
    hyp_id: str = "hid",
    combined: bool = False,
) -> mm.MOTAccumulator:
    """Fill a ``MOTAccumulator`` from GT (``gid`` column) and hypotheses (``hyp_id`` column).

    ``combined=False`` keeps the raw frame index as frame key (use it for a single camera);
    ``combined=True`` uses ``camera_index * FRAME_SPAN + frame`` over ``cameras`` in the order
    given.
    """
    g, h = _groups(gt, "gid"), _groups(hyp, hyp_id)
    acc = mm.MOTAccumulator(auto_id=False)
    empty = (np.zeros(0, int), np.zeros((0, 4)))
    for ci, cam in enumerate(cameras):
        frames = sorted({f for c, f in g if c == cam} | {f for c, f in h if c == cam})
        for f in frames:
            gi, gb = g.get((cam, f), empty)
            hi, hb = h.get((cam, f), empty)
            dist = 1.0 - iou_matrix(gb, hb)
            dist[dist > 1.0 - iou] = np.nan
            acc.update(gi, hi, dist, frameid=ci * FRAME_SPAN + f if combined else f)
    return acc


def _clean(v: Any) -> Any:
    v = float(v)
    return None if math.isnan(v) or math.isinf(v) else v


def summarize(acc: mm.MOTAccumulator) -> dict[str, float | None]:
    """The metrics of :data:`_METRICS` for one accumulator (NaN -> None)."""
    if len(acc.mot_events) == 0:
        return {_RENAME.get(m, m): None for m in _METRICS}
    row = mm.metrics.create().compute(acc, metrics=_METRICS, name="x").iloc[0]
    return {_RENAME.get(m, m): _clean(row[m]) for m in _METRICS}


def per_camera(
    gt: pd.DataFrame, hyp: pd.DataFrame, iou: float, cameras: list[str], hyp_id: str
) -> dict[str, dict]:
    """:func:`summarize` for each camera separately (one accumulator per camera)."""
    out = {}
    for cam in cameras:
        g, h = gt[gt["camera"] == cam], hyp[hyp["camera"] == cam]
        out[cam] = summarize(build_accumulator(g, h, iou, [cam], hyp_id))
    return out


def pair_counts(acc: mm.MOTAccumulator) -> pd.DataFrame:
    """Matched box counts per (GT id, hypothesis id) as columns ``oid, hid, n``."""
    ev = acc.mot_events
    m = ev[ev["Type"].isin(["MATCH", "SWITCH"])]
    if m.empty:
        return pd.DataFrame({"oid": [], "hid": [], "n": []}, dtype=int)
    c = m.groupby(["OId", "HId"]).size().rename("n").reset_index()
    return c.rename(columns={"OId": "oid", "HId": "hid"}).astype({"oid": int, "hid": int})


def fragmentation(
    pairs: pd.DataFrame, n_gt_ids: int, n_gids: int, min_boxes: int = 10, min_frac: float = 0.05
) -> dict[str, Any]:
    """Identity fragmentation and purity from :func:`pair_counts`.

    Only positive hypothesis ids (real gids) are linked; boxes matched to unidentified tracklets
    (negative ids) are counted in ``matched_boxes_unidentified``. A (GT id, gid) link counts only
    if it has >= ``min_boxes`` matched boxes and >= ``min_frac`` of the GT id's (resp. the
    gid's) matched boxes, so a few stray boxes do not create "fragments".

    * ``gt_fragmented``: GT ids linked to more than one gid; ``gt_one_gid``: exactly one.
    * ``gids_per_gt_mean``: mean number of linked gids per GT id (1.0 is ideal).
    * ``gid_impure``: gids linked to more than one GT id (a merge of people).
    * ``gid_purity``: share of a gid's matched boxes belonging to its majority GT id.
    * ``gt_completeness``: share of a GT id's matched boxes carried by its majority gid.
    """
    pos = pairs[pairs["hid"] > 0]
    total_pos = int(pos["n"].sum())
    res: dict[str, Any] = {
        "n_gt_ids": n_gt_ids,
        "n_system_gids": n_gids,
        "matched_boxes_identified": total_pos,
        "matched_boxes_unidentified": int(pairs[pairs["hid"] < 0]["n"].sum()),
        "min_boxes": min_boxes,
        "min_frac": min_frac,
    }
    if pos.empty:
        res.update(gt_with_match=0, gt_fragmented=0, gt_one_gid=0, gids_per_gt_mean=None,
                   gid_impure=0, gid_purity=None, gt_completeness=None, per_gt={})  # fmt: skip
        return res
    by_gt = pos.groupby("oid")["n"].transform("sum")
    by_gid = pos.groupby("hid")["n"].transform("sum")
    link = pos[(pos["n"] >= min_boxes) & (pos["n"] >= min_frac * by_gt)]
    link_g = pos[(pos["n"] >= min_boxes) & (pos["n"] >= min_frac * by_gid)]
    gids_per_gt = link.groupby("oid")["hid"].nunique()
    gts_per_gid = link_g.groupby("hid")["oid"].nunique()
    res.update(
        gt_with_match=int(gids_per_gt.size),
        gt_fragmented=int((gids_per_gt > 1).sum()),
        gt_one_gid=int((gids_per_gt == 1).sum()),
        gids_per_gt_mean=_clean(gids_per_gt.mean()),
        gid_impure=int((gts_per_gid > 1).sum()),
        gid_purity=_clean(pos.groupby("hid")["n"].max().sum() / total_pos),
        gt_completeness=_clean(pos.groupby("oid")["n"].max().sum() / total_pos),
        per_gt={
            str(o): {str(h): int(n) for h, n in zip(g["hid"], g["n"], strict=True)}
            for o, g in link.groupby("oid")
        },
    )
    return res


def majority_gt(pairs: pd.DataFrame) -> dict[int, int]:
    """For every positive gid, the GT id with most matched boxes (gid -> GT id)."""
    pos = pairs[pairs["hid"] > 0]
    if pos.empty:
        return {}
    best = pos.sort_values(["hid", "n", "oid"], ascending=[True, False, True]).drop_duplicates(
        "hid"
    )
    return {int(h): int(o) for h, o in zip(best["hid"], best["oid"], strict=True)}


def tracklet_purity(pairs: pd.DataFrame, hyp: pd.DataFrame, min_boxes: int = 5) -> dict[str, Any]:
    """ID switches *inside* tracklets, from :func:`pair_counts` of a tracklet-id accumulator.

    A tracklet is "mixed" if it is matched to two or more GT people with >= ``min_boxes``
    matched boxes each: the tracker swapped people (typically when they cross) and everything
    downstream sees one person. Box-level metrics barely notice a short swap, but the identity
    layer then assigns a blended appearance to one gid, which is far more visible to a user.

    * ``mixed``: number of mixed tracklets (0 is ideal), out of ``tracklets`` matched ones;
    * ``purity``: share of matched boxes that belong to their tracklet's majority GT person;
    * ``examples``: the mixed tracklets (id, camera, gid, matched boxes per GT person).
    """
    if pairs.empty:
        return {"tracklets": 0, "mixed": 0, "purity": None, "min_boxes": min_boxes, "examples": []}
    total = int(pairs["n"].sum())
    strong = pairs[pairs["n"] >= min_boxes]
    n_gt = strong.groupby("hid")["oid"].nunique()
    mixed = sorted(int(t) for t in n_gt[n_gt > 1].index)
    info = hyp.drop_duplicates("tid").set_index("tid")
    examples = []
    for t in mixed:
        g = pairs[pairs["hid"] == t].sort_values("n", ascending=False)
        gid = info.at[t, "gid"] if t in info.index else None
        examples.append(
            {
                "tracklet": t,
                "camera": str(info.at[t, "camera"]) if t in info.index else None,
                "gid": None if gid is None or pd.isna(gid) else int(gid),
                "gt_boxes": {str(o): int(n) for o, n in zip(g["oid"], g["n"], strict=True)},
            }
        )
    return {
        "tracklets": int(pairs["hid"].nunique()),
        "mixed": len(mixed),
        "purity": _clean(pairs.groupby("hid")["n"].max().sum() / total),
        "min_boxes": min_boxes,
        "examples": examples,
    }
