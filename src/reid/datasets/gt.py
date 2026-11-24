"""Uniform internal ground-truth format shared by the dataset helpers and ``reid.eval``.

A GT file is a CSV with the header ``camera,frame,gid,x1,y1,x2,y2``: one row per person box,
``camera`` is the pipeline camera id (the ``id`` of the camera YAML entry), ``frame`` the frame
index in the (possibly trimmed) camera video, ``gid`` a person id that is **global**: the same
number on every camera for the same person. Box coordinates are pixels of that camera video.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

GT_COLUMNS = ["camera", "frame", "gid", "x1", "y1", "x2", "y2"]


def write_gt(df: pd.DataFrame, path: str | Path) -> Path:
    """Write ``df`` (columns :data:`GT_COLUMNS`) as the uniform GT CSV; returns the path."""
    missing = [c for c in GT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"GT frame lacks columns {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[GT_COLUMNS].sort_values(["camera", "frame", "gid"], kind="stable")
    out.to_csv(path, index=False, float_format="%.1f")
    return path


def read_gt(path: str | Path) -> pd.DataFrame:
    """Read a uniform GT CSV (see module docstring)."""
    df = pd.read_csv(path, dtype={"camera": str})
    missing = [c for c in GT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: not a uniform GT file, missing columns {missing}")
    df["frame"] = df["frame"].astype(int)
    df["gid"] = df["gid"].astype(int)
    return df[GT_COLUMNS]
