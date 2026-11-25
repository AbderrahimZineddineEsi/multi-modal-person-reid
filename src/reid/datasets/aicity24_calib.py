"""Calibration and official-GT helpers of the AI City 2024 fetcher.

``calibration_2025_format.json`` lists ``sensors``; each camera sensor has an ``id``
(``Camera_0699``) and a 3x3 ``homography`` that maps world floor coordinates ``(x, y, 1)`` to
full-resolution image pixels. The pipeline wants the opposite direction, so
:func:`floor_homographies` returns ``inv(homography)`` normalised to ``[2][2] == 1``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Collection, Mapping
from pathlib import Path

import numpy as np

from reid.geometry import invert_homography

log = logging.getLogger(__name__)

_SENSOR_RE = re.compile(r"(\d+)$")


def load_calibration(path: str | Path) -> dict[int, np.ndarray]:
    """Camera number -> 3x3 world-to-image homography, for every camera sensor in the file.

    Sensors without a usable ``homography`` are skipped (the calibration may also list sensors
    that have no video).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[int, np.ndarray] = {}
    for s in data.get("sensors", []):
        m = _SENSOR_RE.search(str(s.get("id", "")))
        if s.get("type", "camera") != "camera" or not m or "homography" not in s:
            continue
        h = np.asarray(s["homography"], dtype=np.float64)
        if h.shape == (3, 3) and np.isfinite(h).all() and abs(np.linalg.det(h)) > 1e-12:
            out[int(m.group(1))] = h
    return out


def floor_homographies(
    calibration: Mapping[int, np.ndarray], cameras: Collection[int]
) -> dict[int, list[list[float]]]:
    """Image-to-floor homographies (source pixels) of the chosen camera numbers.

    A camera missing from the calibration is logged and left out.
    """
    out: dict[int, list[list[float]]] = {}
    for c in cameras:
        if c in calibration:
            out[c] = invert_homography(calibration[c]).tolist()
        else:
            log.warning("camera %d has no usable homography in the calibration file", c)
    return out


def filter_official_gt(
    src: str | Path, dst: str | Path, cameras: Collection[int], max_frame: int | None = None
) -> int:
    """Copy the rows of ``src`` (official ``ground_truth.txt``) whose camera is in ``cameras``.

    The format is kept byte for byte (only whole lines are dropped), so the result can be fed to
    the official evaluator. ``max_frame`` additionally drops frames ``>= max_frame`` (a trimmed
    video). Returns the number of rows written. Streams the file: it can be 90 MB.
    """
    keep = {str(int(c)) for c in cameras}
    n = 0
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.split(" ", 3)
            if len(parts) < 3 or parts[0] not in keep:
                continue
            if max_frame is not None and int(parts[2]) >= max_frame:
                continue
            fout.write(line if line.endswith("\n") else line + "\n")
            n += 1
    return n


def write_scene_extras(
    calib_src: str | Path | None,
    gt_src: str | Path,
    out: str | Path,
    cameras: Collection[int],
    max_frame: int | None,
) -> tuple[dict[int, list[list[float]]], int]:
    """Write ``calibration.json`` and ``gt_official.txt`` into ``out`` for the chosen cameras.

    Returns ``(image-to-floor homographies by camera number, official GT rows written)``. Without
    a calibration file the homography dict is empty.
    """
    out = Path(out)
    homs: dict[int, list[list[float]]] = {}
    if calib_src is not None:
        shutil.copyfile(calib_src, out / "calibration.json")
        homs = floor_homographies(load_calibration(calib_src), cameras)
    n = filter_official_gt(gt_src, out / "gt_official.txt", cameras, max_frame)
    return homs, n
