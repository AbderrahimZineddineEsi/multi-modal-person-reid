"""Pure analytics over the database (PLAN section 3.5). No model, no vectors.

Replaces v1's Mongo ``sightings`` documents (whose timestamps were meaningless, B1): all times
here are ``camera.start_ts + frame / fps`` computed at extraction time, so journeys and
transitions are consistent across cameras.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from reid.storage import db


@dataclass(slots=True)
class JourneyStep:
    """One tracklet of an identity's journey."""

    tracklet_id: int
    camera_id: str
    camera_name: str
    start_ts: float
    end_ts: float
    duration_s: float
    gap_s: float | None  # seconds since the previous step ended (None for the first)
    assign_source: str
    best_crop_path: str | None
    best_face_path: str | None


def journey(conn: sqlite3.Connection, gid: int) -> list[JourneyStep]:
    """Ordered tracklets of ``gid`` across cameras with durations, gaps and thumbnails."""
    names = {c.id: c.name for c in db.list_cameras(conn)}
    steps: list[JourneyStep] = []
    prev_end: float | None = None
    for t in sorted(db.get_tracklets(conn, gid=gid), key=lambda x: (x.start_ts, x.id or 0)):
        steps.append(
            JourneyStep(
                t.id or 0,
                t.camera_id,
                names.get(t.camera_id, t.camera_id),
                t.start_ts,
                t.end_ts,
                t.end_ts - t.start_ts,
                None if prev_end is None else t.start_ts - prev_end,
                t.assign_source,
                t.best_crop_path,
                t.best_face_path,
            )
        )
        prev_end = max(prev_end or t.end_ts, t.end_ts)
    return steps


def transitions(
    conn: sqlite3.Connection, run_id: int | None = None
) -> tuple[list[str], np.ndarray]:
    """Camera -> camera count matrix of consecutive tracklets of the same identity.

    Returns ``(camera_ids, matrix)``; ``matrix[i, j]`` counts moves from ``camera_ids[i]`` to
    ``camera_ids[j]`` (the diagonal counts re-appearances in the same camera).
    """
    cams = [c.id for c in db.list_cameras(conn)]
    ci = {c: i for i, c in enumerate(cams)}
    mat = np.zeros((len(cams), len(cams)), dtype=int)
    by_gid: dict[int, list[str]] = {}
    for t in db.get_tracklets(conn, run_id=run_id):  # ordered by start_ts
        if t.gid is not None:
            by_gid.setdefault(t.gid, []).append(t.camera_id)
    for seq in by_gid.values():
        for a, b in zip(seq, seq[1:], strict=False):
            mat[ci[a], ci[b]] += 1
    return cams, mat


def dwell_stats(
    conn: sqlite3.Connection, camera_id: str | None = None, gid: int | None = None
) -> dict[str, dict[str, float]]:
    """Per-camera dwell statistics (seconds) of identified tracklets.

    Keys are camera ids; values hold ``n``, ``total_s``, ``mean_s``, ``median_s``, ``p90_s``,
    ``max_s``. Optionally restricted to one camera and/or identity.
    """
    d: dict[str, list[float]] = {}
    for t in db.get_tracklets(conn, camera_id=camera_id, gid=gid):
        if t.gid is not None:
            d.setdefault(t.camera_id, []).append(t.end_ts - t.start_ts)
    return {
        cam: {
            "n": float(len(v)),
            "total_s": float(np.sum(v)),
            "mean_s": float(np.mean(v)),
            "median_s": float(np.median(v)),
            "p90_s": float(np.percentile(v, 90)),
            "max_s": float(np.max(v)),
        }
        for cam, v in sorted(d.items())
    }


def occupancy(
    conn: sqlite3.Connection, camera_id: str, bin_s: float, run_id: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """People in view over time: ``(bin_start_ts, count)``.

    A bin counts the distinct identities (each unidentified tracklet counts as one person)
    whose tracklets overlap the bin.
    """
    if bin_s <= 0:
        raise ValueError("bin_s must be > 0")
    tl = db.get_tracklets(conn, camera_id=camera_id, run_id=run_id)
    if not tl:
        return np.zeros(0), np.zeros(0, dtype=int)
    t0 = min(t.start_ts for t in tl)
    t1 = max(t.end_ts for t in tl)
    n_bins = int(np.floor((t1 - t0) / bin_s)) + 1
    people: list[set[tuple[str, int]]] = [set() for _ in range(n_bins)]
    for t in tl:
        key = ("g", t.gid) if t.gid is not None else ("t", t.id or 0)
        lo = int(np.floor((t.start_ts - t0) / bin_s))
        hi = int(np.floor((t.end_ts - t0) / bin_s))
        for b in range(lo, min(hi, n_bins - 1) + 1):
            people[b].add(key)  # type: ignore[arg-type]
    return t0 + bin_s * np.arange(n_bins), np.array([len(p) for p in people], dtype=int)


def run_summary(conn: sqlite3.Connection, run_id: int) -> dict[str, object]:
    """Headline numbers of one run.

    Keys: ``n_tracklets``, ``n_people`` (distinct identities), ``identified_pct`` (tracklets
    with an identity), ``face_confirmed_pct`` (identities with at least one face-bearing
    tracklet), ``face_assigned_pct`` (identified tracklets decided by face),
    ``tracklets_per_identity`` (gid -> count; fragmentation), ``mean_tracklets_per_identity``
    and ``unidentified_tracklets`` (ids). Per-stage processing fps is stored by the pipeline,
    not derived here.
    """
    tl = db.get_tracklets(conn, run_id=run_id)
    per: dict[int, int] = {}
    face_gids: set[int] = set()
    face_src = 0
    for t in tl:
        if t.gid is None:
            continue
        per[t.gid] = per.get(t.gid, 0) + 1
        if t.face_quality is not None:
            face_gids.add(t.gid)
        face_src += t.assign_source == "face"
    n_id = sum(per.values())

    def pct(a: float, b: float) -> float:
        return 100.0 * a / b if b else 0.0

    return {
        "run_id": run_id,
        "n_tracklets": len(tl),
        "n_people": len(per),
        "identified_pct": pct(n_id, len(tl)),
        "face_confirmed_pct": pct(len(face_gids), len(per)),
        "face_assigned_pct": pct(face_src, n_id),
        "tracklets_per_identity": dict(sorted(per.items())),
        "mean_tracklets_per_identity": (n_id / len(per)) if per else 0.0,
        "unidentified_tracklets": [t.id for t in tl if t.gid is None],
    }
