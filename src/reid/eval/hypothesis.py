"""Load the system's boxes (``track_points``) as an evaluation hypothesis table."""

from __future__ import annotations

import json
import sqlite3

import pandas as pd

from reid.config import Config, RuntimeCfg


def latest_run(conn: sqlite3.Connection, run_id: int | None) -> sqlite3.Row:
    """The run with ``run_id`` or the most recent one (ValueError if there is none)."""
    from reid.storage import db

    row = db.get_run(conn, run_id) if run_id is not None else None
    if row is None:
        runs = db.list_runs(conn)
        if run_id is not None or not runs:
            raise ValueError(f"run {run_id!r} not found" if run_id else "no run in the database")
        row = runs[-1]
    return row


def run_config(row: sqlite3.Row) -> Config:
    """The :class:`Config` stored with a run."""
    return Config.model_validate(json.loads(row["config_json"]))


def load_hypothesis(conn: sqlite3.Connection, run_id: int) -> pd.DataFrame:
    """One row per stored track point of the run.

    Columns: ``camera, frame, tracklet_id, gid, x1, y1, x2, y2``, plus ``hid`` (the hypothesis id
    for identity evaluation: the gid, or ``-tracklet_id`` for an unidentified tracklet, so every
    unidentified tracklet is its own never-matching identity) and ``tid`` (the tracklet id, for
    tracker-only evaluation).
    """
    df = pd.read_sql_query(
        "SELECT t.camera_id AS camera, p.frame AS frame, t.id AS tracklet_id, t.gid AS gid, "
        "p.x1 AS x1, p.y1 AS y1, p.x2 AS x2, p.y2 AS y2 "
        "FROM tracklets t JOIN track_points p ON p.tracklet_id = t.id WHERE t.run_id = ? "
        "ORDER BY t.camera_id, p.frame",
        conn,
        params=(run_id,),
    )
    df["hid"] = df["gid"].where(df["gid"].notna(), -df["tracklet_id"]).astype(int)
    df["tid"] = df["tracklet_id"].astype(int)
    return df


def processed_frame_mask(
    frames: pd.Series, rt: RuntimeCfg, fps: float, n_frames: int | None = None
) -> pd.Series:
    """Which frame indices were actually processed under the run's ``runtime`` settings
    (``start_s``, ``end_s``, ``frame_stride``, ``max_frames``), so GT is only compared on frames
    the pipeline looked at."""
    start = int(round(rt.start_s * fps))
    end = int(round(rt.end_s * fps)) if rt.end_s is not None else None
    if rt.max_frames is not None:
        end = start + rt.max_frames if end is None else min(end, start + rt.max_frames)
    if n_frames is not None:
        end = n_frames if end is None else min(end, n_frames)
    ok = (frames >= start) & ((frames - start) % rt.frame_stride == 0)
    return ok if end is None else ok & (frames < end)


def infer_preset(cfg: Config) -> str | None:
    """Name of the shipped preset whose model choices (detector, tracker, body ReID, face model)
    equal ``cfg``'s, or None. Only a label for the report: the config itself is stored too."""
    import yaml

    from reid.config import expand_dotted, preset_path

    keys = (
        ("detection", "model"),
        ("tracking", "tracker"),
        ("body_reid", "model"),
        ("face", "model"),
    )
    full = cfg.model_dump()
    for name in ("lite", "balanced", "accurate", "max"):
        try:
            spec = expand_dotted(
                yaml.safe_load(preset_path(name).read_text(encoding="utf-8")) or {}
            )
        except FileNotFoundError:
            continue
        if all(spec.get(a, {}).get(b) in (None, full[a][b]) for a, b in keys):
            return name
    return None
