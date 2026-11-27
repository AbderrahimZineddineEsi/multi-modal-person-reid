"""Destructive maintenance on one data_dir: delete an identity or a run, wipe everything.

Plain functions without UI. Each keeps the SQLite rows, the vector collections and the crop
files consistent, and returns a small dict of what was removed. Callers confirm first.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from reid.config import Config
from reid.storage import db, files
from reid.storage.vectors import NO_LABEL, VectorStore

ENV_MARKER = "env.json"  # written by the app for named environments; kept by a wipe
_CHUNK = 500  # stay below SQLite's variable limit


def _chunks(ids: list[int]) -> Iterable[list[int]]:
    for k in range(0, len(ids), _CHUNK):
        yield ids[k : k + _CHUNK]


def _delete_tracklets(
    conn: sqlite3.Connection, vectors: VectorStore, data_dir: Path, tids: list[int]
) -> int:
    """Remove tracklets with their points, samples, history, vectors and crop files."""
    if not tids:
        return 0
    sample_ids: list[int] = []
    paths: list[Path] = []
    for chunk in _chunks(tids):
        marks = ",".join("?" * len(chunk))
        sample_ids += [
            r[0]
            for r in conn.execute(f"SELECT id FROM samples WHERE tracklet_id IN ({marks})", chunk)
        ]
        for r in conn.execute(
            f"SELECT best_crop_path, best_face_path FROM tracklets WHERE id IN ({marks})", chunk
        ):
            paths += [data_dir / p for p in (r[0], r[1]) if p]
    for tid in tids:
        paths += [files.crop_path(data_dir, tid), files.face_path(data_dir, tid)]
    with db.transaction(conn):
        for chunk in _chunks(tids):
            marks = ",".join("?" * len(chunk))
            for table, col in (
                ("tracklet_attributes", "tracklet_id"),
                ("tracklet_gid_history", "tracklet_id"),
                ("samples", "tracklet_id"),
                ("track_points", "tracklet_id"),
                ("tracklets", "id"),
            ):
                conn.execute(f"DELETE FROM {table} WHERE {col} IN ({marks})", chunk)
    vectors.get("body_samples").remove(sample_ids)
    vectors.get("body_tracklet").remove(tids)
    vectors.get("face_tracklet").remove(tids)
    for p in paths:
        p.unlink(missing_ok=True)
    return len(tids)


def _drop_identity_rows(conn: sqlite3.Connection, gids: Iterable[int]) -> None:
    for gid in gids:
        conn.execute("DELETE FROM identity_cannot_links WHERE gid_a=? OR gid_b=?", (gid, gid))
        db.delete_identity(conn, gid)
    conn.commit()


def delete_identity(
    conn: sqlite3.Connection,
    vectors: VectorStore,
    cfg: Config,
    gid: int,
    *,
    keep_tracklets: bool = False,
) -> dict[str, int]:
    """Delete identity ``gid``.

    By default its tracklets are deleted too (points, samples, vectors, crops). With
    ``keep_tracklets`` they stay as unidentified sightings: their vectors lose the label and
    leave the gallery. The identity's prototypes are always removed.
    """
    if db.get_identity(conn, gid) is None:
        raise KeyError(f"unknown identity {gid}")
    tracklets = db.get_tracklets(conn, gid=gid)
    tids = [int(t.id) for t in tracklets]  # type: ignore[arg-type]
    if keep_tracklets:
        db.set_gids(conn, [(t, None, None, "manual") for t in tids], reason="identity deleted")
        sample_ids = [r["id"] for t in tids for r in db.get_samples(conn, tracklet_id=t)]
        if sample_ids:
            db.set_in_gallery(conn, sample_ids, False)
            col = vectors.get("body_samples")
            col.relabel({k: NO_LABEL for k in sample_ids if k in col})
        for name in ("body_tracklet", "face_tracklet"):
            col = vectors.get(name)
            col.relabel({t: NO_LABEL for t in tids if t in col})
        removed = 0
    else:
        removed = _delete_tracklets(conn, vectors, cfg.data_dir.resolve(), tids)
    vectors.get("body_proto").remove_label(gid)
    _drop_identity_rows(conn, [gid])
    db.refresh_identity_stats(conn)
    vectors.save()
    return {"tracklets_deleted": removed, "tracklets_unassigned": len(tids) - removed}


def delete_run(
    conn: sqlite3.Connection, vectors: VectorStore, cfg: Config, run_id: int
) -> dict[str, int]:
    """Delete a run with its tracklets, points, samples, vectors and crops.

    Identities left without tracklets are dropped; the others get fresh statistics and
    prototypes. Cameras stay (they belong to the environment, not to a run).
    """
    from reid.identity.prototypes import rebuild_prototypes

    if db.get_run(conn, run_id) is None:
        raise KeyError(f"unknown run {run_id}")
    data_dir = cfg.data_dir.resolve()
    rows = conn.execute("SELECT id, gid FROM tracklets WHERE run_id=?", (run_id,)).fetchall()
    tids = [int(r["id"]) for r in rows]
    affected = sorted({int(r["gid"]) for r in rows if r["gid"] is not None})
    _delete_tracklets(conn, vectors, data_dir, tids)
    conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
    conn.commit()
    (data_dir / "cache" / f"run_{run_id}.json").unlink(missing_ok=True)
    db.refresh_identity_stats(conn)
    empty = [gid for gid in affected if _is_empty(conn, gid)]
    _drop_identity_rows(conn, empty)
    rebuild_prototypes(conn, vectors, cfg, [g for g in affected if g not in empty])
    for gid in empty:
        vectors.get("body_proto").remove_label(gid)
    vectors.save()
    return {"tracklets_deleted": len(tids), "identities_dropped": len(empty)}


def _is_empty(conn: sqlite3.Connection, gid: int) -> bool:
    return not conn.execute("SELECT 1 FROM tracklets WHERE gid=? LIMIT 1", (gid,)).fetchone()


def wipe_data_dir(data_dir: str | Path) -> None:
    """Delete everything inside ``data_dir`` but keep the directory and its ``env.json``."""
    root = Path(data_dir).expanduser().resolve()
    marker = root / ENV_MARKER
    keep = marker.read_text() if marker.exists() else None
    if root.exists() and any(root.iterdir()):
        (root / files.MARKER).touch()  # the caller has decided this directory is a data_dir
    files.reset_data_dir(root, confirm=True)
    if keep is not None:
        marker.write_text(keep)


def read_marker(data_dir: str | Path) -> dict:
    """Contents of ``env.json`` (empty dict if missing or broken)."""
    try:
        return json.loads((Path(data_dir) / ENV_MARKER).read_text())
    except (OSError, ValueError):
        return {}
