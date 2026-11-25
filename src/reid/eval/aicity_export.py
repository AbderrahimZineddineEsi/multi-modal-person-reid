"""Export a processed environment as an AI City 2024 Track 1 prediction file.

The official evaluator (NVIDIA, HOTA on floor positions) wants one row per identity per frame::

    <camera> <id> <frame> <x> <y> <w> <h> <xworld> <yworld>      (space separated, no header)

``x y w h`` are source-video pixels; ``xworld yworld`` is where the person stands on the floor.
This module builds that file from the database (deviations from the paper pipeline are ours):

* only tracklets with a gid are used; ``Id`` is the gid (global, so one identity across cameras);
* boxes of a tracklet are linearly interpolated between two consecutive stored points when the
  gap is at most ``interpolate_gap`` frames (default ``2 * frame_stride`` of the run, so a
  stride of 2 does not halve the detections); a tracklet is never extended past its ends;
* the foot point (bottom centre of the box, processed pixels) goes through the camera's stored
  image-to-floor homography (processed resolution, see :mod:`reid.geometry`);
* per (gid, frame) the world position is the median over the cameras that see the gid; the row
  keeps the camera whose box is largest (the evaluator only uses ``CameraId`` to know the scene)
  and that camera's box scaled to source pixels.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reid.eval.hypothesis import latest_run, run_config
from reid.geometry import apply_homography
from reid.storage import db, files

log = logging.getLogger(__name__)

_BOX = ["x1", "y1", "x2", "y2"]


def interpolate_points(points: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """Add the missing frames between consecutive points of a tracklet.

    ``points`` has ``tracklet_id, frame, x1, y1, x2, y2`` (any extra columns are carried from the
    earlier point). Between two consecutive points ``gap`` frames apart, ``gap - 1`` boxes are
    interpolated when ``2 <= gap <= max_gap``. Added rows have ``interpolated = True``.
    """
    df = points.sort_values(["tracklet_id", "frame"]).reset_index(drop=True)
    df["interpolated"] = False
    if max_gap < 2 or len(df) < 2:
        return df
    same = df["tracklet_id"].to_numpy()[1:] == df["tracklet_id"].to_numpy()[:-1]
    gap = np.diff(df["frame"].to_numpy())
    src = np.flatnonzero(same & (gap >= 2) & (gap <= max_gap))
    if len(src) == 0:
        return df
    n_new = gap[src] - 1
    rep = np.repeat(src, n_new)
    k = np.arange(n_new.sum()) - np.repeat(np.cumsum(n_new) - n_new, n_new) + 1  # 1 .. gap-1
    t = (k / gap[rep])[:, None]
    new = df.iloc[rep].copy().reset_index(drop=True)
    a, b = df.loc[rep, _BOX].to_numpy(), df.loc[rep + 1, _BOX].to_numpy()
    new[_BOX] = a + t * (b - a)
    new["frame"] = df.loc[rep, "frame"].to_numpy() + k
    new["interpolated"] = True
    return pd.concat([df, new], ignore_index=True).sort_values(["tracklet_id", "frame"])


def _load_points(conn: Any, run_id: int) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT t.id AS tracklet_id, t.camera_id AS camera, t.gid AS gid, p.frame AS frame, "
        "p.x1 AS x1, p.y1 AS y1, p.x2 AS x2, p.y2 AS y2 "
        "FROM tracklets t JOIN track_points p ON p.tracklet_id = t.id "
        "WHERE t.run_id = ? AND t.gid IS NOT NULL",
        conn,
        params=(run_id,),
    )


def _world(df: pd.DataFrame, homographies: Mapping[str, Any]) -> pd.DataFrame:
    """Add ``wx, wy``: foot points of every row mapped to the floor by their camera's homography."""
    df = df.copy()
    df["wx"] = np.nan
    df["wy"] = np.nan
    for cam, idx in df.groupby("camera").groups.items():
        h = homographies.get(cam)
        if h is None:
            raise ValueError(f"camera {cam!r} has no homography: cannot compute world positions")
        sub = df.loc[idx]
        foot = np.stack([((sub["x1"] + sub["x2"]) / 2).to_numpy(), sub["y2"].to_numpy()], axis=1)
        df.loc[idx, ["wx", "wy"]] = apply_homography(h, foot)
    return df


def fuse_rows(
    df: pd.DataFrame,
    camera_numbers: Mapping[str, int],
    src_sizes: Mapping[str, tuple[int, int]],
    proc_sizes: Mapping[str, tuple[int, int]],
) -> pd.DataFrame:
    """One output row per (gid, frame) from per-camera rows holding ``wx, wy`` and a box.

    World position = median over the cameras; camera and box come from the row with the largest
    box (in source pixels). Boxes are scaled to source pixels, clipped to the frame and rounded.
    """
    df = df.copy()
    sx = df["camera"].map({c: src_sizes[c][0] / proc_sizes[c][0] for c in proc_sizes})
    sy = df["camera"].map({c: src_sizes[c][1] / proc_sizes[c][1] for c in proc_sizes})
    W = df["camera"].map({c: src_sizes[c][0] for c in proc_sizes})
    H = df["camera"].map({c: src_sizes[c][1] for c in proc_sizes})
    x1, x2 = (df["x1"] * sx).clip(0, W), (df["x2"] * sx).clip(0, W)
    y1, y2 = (df["y1"] * sy).clip(0, H), (df["y2"] * sy).clip(0, H)
    df["bx"], df["by"] = x1.round(), y1.round()
    df["bw"], df["bh"] = (x2 - x1).round().clip(lower=1), (y2 - y1).round().clip(lower=1)
    df["area"] = (x2 - x1) * (y2 - y1)
    keys = ["gid", "frame"]
    med = df.groupby(keys)[["wx", "wy"]].median()
    best = df.loc[df.groupby(keys)["area"].idxmax()].set_index(keys)
    out = best[["camera", "bx", "by", "bw", "bh"]].join(med).reset_index()
    out["camera"] = out["camera"].map(dict(camera_numbers))
    return out.sort_values(["frame", "gid"]).reset_index(drop=True)


def export_predictions(
    data_dir: str | Path,
    out_path: str | Path,
    *,
    camera_numbers: Mapping[str, int],
    src_sizes: Mapping[str, tuple[int, int]],
    interpolate_gap: int | None = None,
    run_id: int | None = None,
) -> dict[str, int]:
    """Write the official prediction file for ``run_id`` (default: the latest run).

    ``camera_numbers`` maps camera id -> dataset camera number (the first column) and
    ``src_sizes`` camera id -> ``(width, height)`` of the source video. ``interpolate_gap``
    defaults to ``2 * runtime.frame_stride`` of the run. Raises ``ValueError`` for a camera
    without a homography or a camera missing from ``camera_numbers`` / ``src_sizes``.

    Returns counts: ``rows``, ``gids``, ``frames``, ``points`` (stored boxes used),
    ``interpolated_rows`` (boxes added before fusion) and ``skipped_tracklets`` (no gid).
    """
    data_dir = Path(data_dir)
    conn = db.connect(files.db_path(data_dir))
    try:
        row = latest_run(conn, run_id)
        rid = int(row["id"])
        gap = (
            int(interpolate_gap)
            if interpolate_gap is not None
            else 2 * (run_config(row).runtime.frame_stride)
        )
        cams = {c.id: c for c in db.list_cameras(conn)}
        pts = _load_points(conn, rid)
        skipped = int(
            conn.execute(
                "SELECT COUNT(*) FROM tracklets WHERE run_id=? AND gid IS NULL", (rid,)
            ).fetchone()[0]
        )
    finally:
        conn.close()
    used = set(pts["camera"].unique())
    missing = sorted(c for c in used if c not in camera_numbers or c not in src_sizes)
    if missing:
        raise ValueError(f"cameras {missing} need an entry in camera_numbers and src_sizes")
    homographies = {c: cams[c].homography for c in used if c in cams and cams[c].homography}
    n_points = len(pts)
    full = interpolate_points(pts, gap)
    n_interp = int(full["interpolated"].sum())
    full = _world(full, homographies) if len(full) else full.assign(wx=[], wy=[])
    proc = {c: (cams[c].width, cams[c].height) for c in used}
    fused = fuse_rows(full, camera_numbers, src_sizes, proc) if len(full) else full
    cols = ["camera", "gid", "frame", "bx", "by", "bw", "bh", "wx", "wy"]
    out = pd.DataFrame(columns=cols) if fused.empty else fused[cols].copy()
    for c in cols[:7]:
        out[c] = out[c].astype(int)
    out[["wx", "wy"]] = out[["wx", "wy"]].astype(float).round(3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep=" ", header=False, index=False, float_format="%.3f")
    stats = {
        "rows": len(out),
        "gids": int(out["gid"].nunique()),
        "frames": int(out["frame"].nunique()),
        "points": n_points,
        "interpolated_rows": n_interp,
        "skipped_tracklets": skipped,
    }
    log.info("exported %s: %s", out_path, stats)
    return stats
