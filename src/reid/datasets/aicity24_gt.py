"""Official AI City 2024 ground truth: parse it, choose cameras, convert to the uniform format."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from reid.datasets.gt import GT_COLUMNS


def load_raw_gt(path: str | Path) -> pd.DataFrame:
    """Parse an official ``ground_truth.txt`` into columns ``cam,gid,f,x,y,w,h``."""
    return pd.read_csv(
        path,
        sep=" ",
        header=None,
        usecols=range(7),
        names=["cam", "gid", "f", "x", "y", "w", "h"],
        dtype={"cam": int, "gid": int, "f": int, "x": float, "y": float, "w": float, "h": float},
    )


def select_cameras(
    raw: pd.DataFrame,
    available: Sequence[int],
    n: int,
    max_frame: int | None = None,
    *,
    min_h: float = 60.0,
    max_people: int = 14,
) -> tuple[list[int], dict[str, int]]:
    """Choose ``n`` cameras whose views overlap and that see a moderate number of people.

    Considers boxes taller than ``min_h`` source pixels in frames ``< max_frame``. Among the
    camera combinations that see at most ``max_people`` distinct people (if none does, the ones
    with the fewest), maximizes the number of people visible in two or more of the chosen
    cameras (what cross-camera association needs), then prefers fewer people. Every chosen
    camera must see at least two people. Returns ``(cameras, stats)``.
    """
    d = raw[raw["h"] >= min_h]
    if max_frame is not None:
        d = d[d["f"] < max_frame]
    avail = {int(c) for c in available}
    seen = {int(c): set(g["gid"].unique()) for c, g in d.groupby("cam") if int(c) in avail}
    cams = sorted(c for c, s in seen.items() if len(s) >= 2)
    n = min(n, len(cams))
    if n == 0:
        raise ValueError("no camera sees at least two people in the requested window")
    best: tuple | None = None
    combos = itertools.combinations(cams, n)
    for combo in itertools.islice(combos, 200_000):
        everyone = set().union(*(seen[c] for c in combo))
        multi = sum(sum(g in seen[c] for c in combo) >= 2 for g in everyone)
        key = (len(everyone) <= max_people, multi if len(everyone) <= max_people else 0)
        key = (*key, -len(everyone), tuple(-c for c in combo))
        if best is None or key > best[0]:
            best = (key, combo, len(everyone), multi)
    assert best is not None
    _, combo, total, multi = best
    return list(combo), {"people": total, "people_in_2plus_cameras": multi}


def convert_gt(
    raw: pd.DataFrame,
    cameras: dict[int, str],
    *,
    max_frame: int | None,
    scale: dict[int, tuple[float, float]],
    size: dict[int, tuple[int, int]],
    min_height: float = 0.0,
) -> pd.DataFrame:
    """Official GT -> uniform GT (:data:`reid.datasets.gt.GT_COLUMNS`).

    Keeps the cameras in ``cameras`` (number -> pipeline id) and frames ``< max_frame``, scales
    boxes per camera by ``scale[num] = (sx, sy)``, clips them to ``size[num] = (w, h)`` and drops
    boxes that end up shorter than ``min_height`` pixels or empty.
    """
    d = raw[raw["cam"].isin(cameras)]
    if max_frame is not None:
        d = d[d["f"] < max_frame]
    parts = []
    for num, cid in cameras.items():
        c = d[d["cam"] == num]
        sx, sy = scale[num]
        w, h = size[num]
        x1, y1 = (c["x"] * sx).clip(0, w), (c["y"] * sy).clip(0, h)
        x2, y2 = ((c["x"] + c["w"]) * sx).clip(0, w), ((c["y"] + c["h"]) * sy).clip(0, h)
        part = pd.DataFrame(
            {
                "camera": cid,
                "frame": c["f"],
                "gid": c["gid"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
        parts.append(part[((y2 - y1) > max(min_height, 0.0)) & ((x2 - x1) > 0)])
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=GT_COLUMNS)
    return out[GT_COLUMNS]
