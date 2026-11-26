"""Privacy blur for rendered videos: blur people other than the highlighted one.

Two sources of boxes are blurred: the stored track points of every other person (what the
system detected) and, optionally, a ground-truth CSV (``camera,frame,gid,x1,y1,x2,y2`` in
processed pixels), so that people the detector missed are hidden as well. The highlighted
person's own boxes are never blurred; their ground-truth ids are listed in ``keep``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


class PrivacyIndex:
    """Ground-truth boxes as one sorted float32 array plus per-camera frame offsets.

    Memory is ~16 bytes per box (no per-frame Python objects); ``get`` is two ``searchsorted``.
    """

    def __init__(self, cameras: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        # camera -> (sorted frame numbers (int32), boxes (n, 4) float32 in the same order)
        self._cams = cameras

    def __len__(self) -> int:
        return sum(len(f) for f, _ in self._cams.values())

    def get(self, key: tuple[str, int]) -> np.ndarray | None:
        """Boxes of ``(camera, frame)``, or None when there are none."""
        entry = self._cams.get(key[0])
        if entry is None:
            return None
        frames, boxes = entry
        lo = int(np.searchsorted(frames, key[1], "left"))
        hi = int(np.searchsorted(frames, key[1], "right"))
        return boxes[lo:hi] if hi > lo else None


def load_privacy_boxes(
    path: str | Path,
    keep: list[int] | tuple[int, ...] = (),
) -> PrivacyIndex:
    """Index the ground-truth boxes of ``path`` by ``(camera, frame)``, without the ``keep`` ids."""
    import pandas as pd

    cols = {"camera": "category", "frame": "int32", "gid": "int32"}
    cols.update({c: "float32" for c in ("x1", "y1", "x2", "y2")})
    df = pd.read_csv(path, usecols=list(cols), dtype=cols)
    mask = ~df["gid"].isin(list(keep)) if keep else np.ones(len(df), bool)
    df = df[mask]
    cams: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cam, g in df.groupby("camera", observed=True):
        g = g.sort_values("frame", kind="stable")
        cams[str(cam)] = (
            g["frame"].to_numpy(np.int32),
            g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32),
        )
    idx = PrivacyIndex(cams)
    log.info("privacy: %d boxes in %d cameras from %s", len(idx), len(cams), path)
    return idx


def blur_boxes(
    img: np.ndarray, boxes: list[tuple[float, float, float, float]], strength: float
) -> None:
    """Gaussian-blur each box of ``img`` in place (padded by 15%, kernel ~ ``strength`` x width)."""
    H, W = img.shape[:2]
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            continue
        px, py = bw * 0.15, bh * 0.15
        a, b = max(int(x1 - px), 0), max(int(y1 - py), 0)
        c, d = min(int(np.ceil(x2 + px)), W), min(int(np.ceil(y2 + py)), H)
        if c - a < 2 or d - b < 2:
            continue
        k = max(3, int(bw * strength)) | 1
        img[b:d, a:c] = cv2.GaussianBlur(img[b:d, a:c], (k, k), 0)
