"""SQLite schema and typed repository functions (no ORM).

The connection runs in autocommit mode (``isolation_level=None``); use :func:`transaction`
to group writes. WAL + ``synchronous=NORMAL`` keeps bulk inserts of ~1M track points fast.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from reid.types import CameraInfo, Identity, Tracklet, TrackPoint

SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    created_at  REAL NOT NULL,
    config_json TEXT NOT NULL,
    status      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cameras (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    video_path      TEXT NOT NULL,
    fps             REAL NOT NULL,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    start_ts        REAL NOT NULL DEFAULT 0,
    is_face_cam     INTEGER NOT NULL DEFAULT 0,
    homography_json TEXT,
    background_path TEXT
);

CREATE TABLE IF NOT EXISTS tracklets (
    id              INTEGER PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES runs(id),
    camera_id       TEXT NOT NULL REFERENCES cameras(id),
    local_track_id  INTEGER NOT NULL,
    start_frame     INTEGER NOT NULL,
    end_frame       INTEGER NOT NULL,
    start_ts        REAL NOT NULL,
    end_ts          REAL NOT NULL,
    n_points        INTEGER NOT NULL,
    n_samples       INTEGER NOT NULL,
    mean_quality    REAL NOT NULL,
    gid             INTEGER,
    assign_score    REAL,
    assign_source   TEXT NOT NULL DEFAULT '',  -- 'body'|'face'|'consolidate'|'manual'
    best_crop_path  TEXT,
    best_face_path  TEXT,
    face_quality    REAL
);
CREATE INDEX IF NOT EXISTS idx_tracklets_run_cam ON tracklets(run_id, camera_id);
CREATE INDEX IF NOT EXISTS idx_tracklets_cam_frames ON tracklets(camera_id, start_frame, end_frame);
CREATE INDEX IF NOT EXISTS idx_tracklets_gid ON tracklets(gid);

-- EVERY tracked frame. WITHOUT ROWID: the (tracklet_id, frame) primary key is the clustered index.
CREATE TABLE IF NOT EXISTS track_points (
    tracklet_id INTEGER NOT NULL REFERENCES tracklets(id),
    frame       INTEGER NOT NULL,
    ts          REAL NOT NULL,
    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
    conf        REAL NOT NULL,
    sampled     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (tracklet_id, frame)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS identities (
    gid         INTEGER PRIMARY KEY,
    created_at  REAL NOT NULL,
    label       TEXT,
    first_ts    REAL,
    last_ts     REAL,
    n_tracklets INTEGER NOT NULL DEFAULT 0,
    has_face    INTEGER NOT NULL DEFAULT 0
);

-- Every gid change of a tracklet is recorded, so consolidation is reversible.
CREATE TABLE IF NOT EXISTS tracklet_gid_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tracklet_id INTEGER NOT NULL REFERENCES tracklets(id),
    old_gid     INTEGER,
    new_gid     INTEGER,
    old_source  TEXT NOT NULL DEFAULT '',
    new_source  TEXT NOT NULL DEFAULT '',
    reason      TEXT NOT NULL DEFAULT '',
    changed_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gid_history_tracklet ON tracklet_gid_history(tracklet_id);

-- v2 (P3): body samples of tracklets. ``id`` is the key in the ``body_samples`` vector
-- collection; ``in_gallery`` marks membership of the identity's self-adaptive gallery.
CREATE TABLE IF NOT EXISTS samples (
    id          INTEGER PRIMARY KEY,
    tracklet_id INTEGER NOT NULL REFERENCES tracklets(id),
    frame       INTEGER NOT NULL,
    quality     REAL NOT NULL,
    in_gallery  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_samples_tracklet ON samples(tracklet_id);

-- v2 (P3): persistent cannot-links between identities (e.g. after a manual split).
CREATE TABLE IF NOT EXISTS identity_cannot_links (
    gid_a  INTEGER NOT NULL,
    gid_b  INTEGER NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (gid_a, gid_b)
);

-- v3: pedestrian attribute heads (the weights themselves live in a head.npz) and the
-- attribute probabilities of each tracklet (see reid.attributes.annotate / reid.find).
CREATE TABLE IF NOT EXISTS attribute_heads (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    sha256          TEXT NOT NULL UNIQUE,
    body_model      TEXT NOT NULL,
    neck_feat       TEXT NOT NULL,
    labels_json     TEXT NOT NULL,
    groups_json     TEXT NOT NULL,
    thresholds_json TEXT NOT NULL,
    trained_json    TEXT NOT NULL,
    created_at      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tracklet_attributes (
    tracklet_id INTEGER NOT NULL REFERENCES tracklets(id),
    head_id     INTEGER NOT NULL REFERENCES attribute_heads(id),
    probs       BLOB NOT NULL,       -- float32[n_labels]: quality-weighted mean over the samples
    n_samples   INTEGER NOT NULL,
    PRIMARY KEY (tracklet_id, head_id)
) WITHOUT ROWID;
"""

POINT_DTYPE = np.dtype(
    [
        ("tracklet_id", "i8"),
        ("frame", "i8"),
        ("ts", "f8"),
        ("x1", "f4"),
        ("y1", "f4"),
        ("x2", "f4"),
        ("y2", "f4"),
        ("conf", "f4"),
        ("sampled", "i1"),
    ]
)

_TRACKLET_COLS = (
    "id, run_id, camera_id, local_track_id, start_frame, end_frame, start_ts, end_ts, n_points, "
    "n_samples, mean_quality, gid, assign_score, assign_source, best_crop_path, best_face_path, "
    "face_quality"
)
_IDENTITY_FIELDS = {"label", "first_ts", "last_ts", "n_tracklets", "has_face", "created_at"}


# --------------------------------------------------------------------------- connection


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the database at ``path`` and ensure the schema exists."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables and record/check the schema version."""
    conn.executescript(SCHEMA)  # CREATE IF NOT EXISTS: also adds tables new in later versions
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
    elif row["version"] in (1, 2):  # v1 -> v2 -> v3: new tables only (created above), additive
        conn.execute("UPDATE schema_version SET version=?", (SCHEMA_VERSION,))
    elif row["version"] != SCHEMA_VERSION:
        raise RuntimeError(
            f"database schema v{row['version']} != expected v{SCHEMA_VERSION}; migration needed"
        )


def schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT version FROM schema_version").fetchone()["version"])


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` ... ``COMMIT`` (``ROLLBACK`` on error). Not re-entrant."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Row counts of the main tables."""
    return {
        t: int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])  # noqa: S608
        for t in ("runs", "cameras", "tracklets", "track_points", "identities")
    }


# --------------------------------------------------------------------------- runs / cameras


def create_run(conn: sqlite3.Connection, config_json: str, status: str = "running") -> int:
    """Insert a run row and return its id."""
    cur = conn.execute(
        "INSERT INTO runs(created_at, config_json, status) VALUES (?, ?, ?)",
        (time.time(), config_json, status),
    )
    assert cur.lastrowid is not None
    return cur.lastrowid


def set_run_status(conn: sqlite3.Connection, run_id: int, status: str) -> None:
    conn.execute("UPDATE runs SET status=? WHERE id=?", (status, run_id))


def get_run(conn: sqlite3.Connection, run_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()


def list_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM runs ORDER BY id").fetchall()


def upsert_camera(conn: sqlite3.Connection, cam: CameraInfo) -> None:
    """Insert or replace a camera record."""
    conn.execute(
        """INSERT INTO cameras(id, name, video_path, fps, width, height, start_ts, is_face_cam,
                               homography_json, background_path)
           VALUES (?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET name=excluded.name, video_path=excluded.video_path,
             fps=excluded.fps, width=excluded.width, height=excluded.height,
             start_ts=excluded.start_ts, is_face_cam=excluded.is_face_cam,
             homography_json=excluded.homography_json, background_path=excluded.background_path""",
        (
            cam.id,
            cam.name,
            cam.video_path,
            cam.fps,
            cam.width,
            cam.height,
            cam.start_ts,
            int(cam.is_face_cam),
            json.dumps(cam.homography) if cam.homography else None,
            cam.background_path,
        ),
    )


def _camera(row: sqlite3.Row) -> CameraInfo:
    return CameraInfo(
        id=row["id"],
        name=row["name"],
        video_path=row["video_path"],
        fps=row["fps"],
        width=row["width"],
        height=row["height"],
        start_ts=row["start_ts"],
        is_face_cam=bool(row["is_face_cam"]),
        homography=json.loads(row["homography_json"]) if row["homography_json"] else None,
        background_path=row["background_path"],
    )


def get_camera(conn: sqlite3.Connection, camera_id: str) -> CameraInfo | None:
    row = conn.execute("SELECT * FROM cameras WHERE id=?", (camera_id,)).fetchone()
    return _camera(row) if row else None


def list_cameras(conn: sqlite3.Connection) -> list[CameraInfo]:
    return [_camera(r) for r in conn.execute("SELECT * FROM cameras ORDER BY id")]


# --------------------------------------------------------------------------- tracklets


def _point_rows(tracklet_id: int, points: Sequence[TrackPoint]) -> Iterator[tuple]:
    for p in points:
        yield (tracklet_id, p.frame, p.ts, p.x1, p.y1, p.x2, p.y2, p.conf, int(p.sampled))


def insert_tracklets(
    conn: sqlite3.Connection, run_id: int, tracklets: Iterable[Tracklet]
) -> list[int]:
    """Insert tracklets and all their track points in one transaction.

    Sets ``tracklet.id`` / ``run_id`` / ``n_points`` and returns the new ids in order.
    Points use ``executemany`` so ~1M points insert in a few seconds.
    """
    ids: list[int] = []
    with transaction(conn):
        for t in tracklets:
            cur = conn.execute(
                """INSERT INTO tracklets(run_id, camera_id, local_track_id, start_frame, end_frame,
                       start_ts, end_ts, n_points, n_samples, mean_quality, gid, assign_score,
                       assign_source, best_crop_path, best_face_path, face_quality)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    t.camera_id,
                    t.local_track_id,
                    t.start_frame,
                    t.end_frame,
                    t.start_ts,
                    t.end_ts,
                    len(t.points) or t.n_points,
                    t.n_samples,
                    t.mean_quality,
                    t.gid,
                    t.assign_score,
                    t.assign_source,
                    t.best_crop_path,
                    t.best_face_path,
                    t.face_quality,
                ),
            )
            assert cur.lastrowid is not None
            t.id, t.run_id = cur.lastrowid, run_id
            if t.points:
                t.n_points = len(t.points)
                conn.executemany(
                    "INSERT INTO track_points VALUES (?,?,?,?,?,?,?,?,?)",
                    _point_rows(t.id, t.points),
                )
            ids.append(t.id)
    return ids


def _tracklet(row: sqlite3.Row) -> Tracklet:
    return Tracklet(
        id=row["id"],
        run_id=row["run_id"],
        camera_id=row["camera_id"],
        local_track_id=row["local_track_id"],
        start_frame=row["start_frame"],
        end_frame=row["end_frame"],
        start_ts=row["start_ts"],
        end_ts=row["end_ts"],
        n_points=row["n_points"],
        n_samples=row["n_samples"],
        mean_quality=row["mean_quality"],
        gid=row["gid"],
        assign_score=row["assign_score"],
        assign_source=row["assign_source"],
        best_crop_path=row["best_crop_path"],
        best_face_path=row["best_face_path"],
        face_quality=row["face_quality"],
    )


def get_tracklet(conn: sqlite3.Connection, tracklet_id: int) -> Tracklet | None:
    row = conn.execute(f"SELECT {_TRACKLET_COLS} FROM tracklets WHERE id=?", (tracklet_id,))
    r = row.fetchone()
    return _tracklet(r) if r else None


def get_tracklets(
    conn: sqlite3.Connection,
    *,
    run_id: int | None = None,
    camera_id: str | None = None,
    gid: int | None = None,
    unassigned: bool = False,
    ids: Sequence[int] | None = None,
) -> list[Tracklet]:
    """Tracklets filtered by run / camera / gid (``unassigned=True`` selects ``gid IS NULL``).

    Ordered by ``start_ts, id``. Points are not loaded; see :func:`get_tracklet_points`.
    """
    where: list[str] = []
    args: list[object] = []
    if run_id is not None:
        where.append("run_id=?")
        args.append(run_id)
    if camera_id is not None:
        where.append("camera_id=?")
        args.append(camera_id)
    if gid is not None:
        where.append("gid=?")
        args.append(gid)
    if unassigned:
        where.append("gid IS NULL")
    if ids is not None:
        where.append(f"id IN ({','.join('?' * len(ids))})")
        args.extend(ids)
    sql = f"SELECT {_TRACKLET_COLS} FROM tracklets"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY start_ts, id"
    return [_tracklet(r) for r in conn.execute(sql, args)]


def _points_array(rows: list[tuple]) -> np.ndarray:
    return np.array(rows, dtype=POINT_DTYPE) if rows else np.zeros(0, dtype=POINT_DTYPE)


def get_tracklet_points(conn: sqlite3.Connection, tracklet_id: int) -> np.ndarray:
    """All points of one tracklet as a structured array (``POINT_DTYPE``), by frame."""
    rows = conn.execute(
        "SELECT tracklet_id, frame, ts, x1, y1, x2, y2, conf, sampled FROM track_points "
        "WHERE tracklet_id=? ORDER BY frame",
        (tracklet_id,),
    ).fetchall()
    return _points_array([tuple(r) for r in rows])


def get_points(
    conn: sqlite3.Connection,
    camera_id: str,
    start_frame: int,
    end_frame: int,
    run_id: int | None = None,
) -> np.ndarray:
    """Points of a camera with ``start_frame <= frame <= end_frame``, ordered by frame.

    Returns a structured array (``POINT_DTYPE``); join with tracklets via ``tracklet_id``.
    """
    sql = (
        "SELECT p.tracklet_id, p.frame, p.ts, p.x1, p.y1, p.x2, p.y2, p.conf, p.sampled "
        "FROM tracklets t JOIN track_points p ON p.tracklet_id = t.id "
        "WHERE t.camera_id=? AND t.end_frame>=? AND t.start_frame<=? "
        "AND p.frame BETWEEN ? AND ?"
    )
    args: list[object] = [camera_id, start_frame, end_frame, start_frame, end_frame]
    if run_id is not None:
        sql += " AND t.run_id=?"
        args.append(run_id)
    sql += " ORDER BY p.frame, p.tracklet_id"
    return _points_array([tuple(r) for r in conn.execute(sql, args)])


# --------------------------------------------------------------------------- gids


def set_gids(
    conn: sqlite3.Connection,
    assignments: Iterable[tuple[int, int | None, float | None, str]],
    reason: str = "",
) -> int:
    """Set ``(tracklet_id, gid, assign_score, assign_source)`` in bulk.

    Rows whose gid or source changes are logged in ``tracklet_gid_history``. Returns the
    number of tracklets whose gid or source changed (score-only updates are applied silently).
    """
    items = list(assignments)
    if not items:
        return 0
    now = time.time()
    changed = 0
    with transaction(conn):
        old: dict[int, tuple[int | None, str]] = {}
        ids = [i[0] for i in items]
        for k in range(0, len(ids), 500):  # stay below SQLite's variable limit
            chunk = ids[k : k + 500]
            marks = ",".join("?" * len(chunk))
            for r in conn.execute(
                f"SELECT id, gid, assign_source FROM tracklets WHERE id IN ({marks})", chunk
            ):
                old[r["id"]] = (r["gid"], r["assign_source"])
        hist: list[tuple] = []
        for tid, gid, _score, source in items:
            if tid not in old:
                raise KeyError(f"unknown tracklet {tid}")
            o_gid, o_src = old[tid]
            if o_gid != gid or o_src != source:
                hist.append((tid, o_gid, gid, o_src, source, reason, now))
                changed += 1
        conn.executemany(
            "UPDATE tracklets SET gid=?, assign_score=?, assign_source=? WHERE id=?",
            [(gid, score, source, tid) for tid, gid, score, source in items],
        )
        conn.executemany(
            "INSERT INTO tracklet_gid_history(tracklet_id, old_gid, new_gid, old_source, "
            "new_source, reason, changed_at) VALUES (?,?,?,?,?,?,?)",
            hist,
        )
    return changed


def get_gid_history(conn: sqlite3.Connection, tracklet_id: int) -> list[sqlite3.Row]:
    """History of gid changes of a tracklet, oldest first."""
    return conn.execute(
        "SELECT * FROM tracklet_gid_history WHERE tracklet_id=? ORDER BY id", (tracklet_id,)
    ).fetchall()


# --------------------------------------------------------------------------- identities


def _identity(row: sqlite3.Row) -> Identity:
    return Identity(
        gid=row["gid"],
        created_at=row["created_at"],
        label=row["label"],
        first_ts=row["first_ts"],
        last_ts=row["last_ts"],
        n_tracklets=row["n_tracklets"],
        has_face=bool(row["has_face"]),
    )


def next_gid(conn: sqlite3.Connection) -> int:
    """Smallest unused gid above every gid ever seen in identities or tracklets."""
    row = conn.execute(
        "SELECT MAX(COALESCE((SELECT MAX(gid) FROM identities), 0), "
        "COALESCE((SELECT MAX(gid) FROM tracklets), 0)) AS m"
    ).fetchone()
    return int(row["m"]) + 1


def create_identity(
    conn: sqlite3.Connection, gid: int | None = None, label: str | None = None
) -> int:
    """Create an identity (new unique gid if not given) and return the gid."""
    gid = next_gid(conn) if gid is None else gid
    conn.execute(
        "INSERT INTO identities(gid, created_at, label) VALUES (?,?,?)", (gid, time.time(), label)
    )
    return gid


def get_identity(conn: sqlite3.Connection, gid: int) -> Identity | None:
    row = conn.execute("SELECT * FROM identities WHERE gid=?", (gid,)).fetchone()
    return _identity(row) if row else None


def list_identities(conn: sqlite3.Connection) -> list[Identity]:
    return [_identity(r) for r in conn.execute("SELECT * FROM identities ORDER BY gid")]


def update_identity(conn: sqlite3.Connection, gid: int, **fields: object) -> None:
    """Update identity columns (label, first_ts, last_ts, n_tracklets, has_face, created_at)."""
    bad = set(fields) - _IDENTITY_FIELDS
    if bad:
        raise ValueError(f"unknown identity fields: {sorted(bad)}")
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    cur = conn.execute(f"UPDATE identities SET {sets} WHERE gid=?", (*fields.values(), gid))
    if cur.rowcount == 0:
        raise KeyError(f"unknown identity {gid}")


def delete_identity(conn: sqlite3.Connection, gid: int) -> None:
    """Delete an identity row (tracklets keep their gid column; reassign them first)."""
    conn.execute("DELETE FROM identities WHERE gid=?", (gid,))


def refresh_identity_stats(conn: sqlite3.Connection) -> None:
    """Recompute first/last ts, tracklet count and has_face from tracklets.

    Identities with no tracklets are left in place with zero counts; gids used by tracklets
    but missing from ``identities`` are created.
    """
    with transaction(conn):
        conn.execute(
            "INSERT OR IGNORE INTO identities(gid, created_at) "
            "SELECT DISTINCT gid, ? FROM tracklets WHERE gid IS NOT NULL",
            (time.time(),),
        )
        conn.execute(
            """UPDATE identities SET
                 first_ts    = (SELECT MIN(start_ts) FROM tracklets t WHERE t.gid=identities.gid),
                 last_ts     = (SELECT MAX(end_ts) FROM tracklets t WHERE t.gid=identities.gid),
                 n_tracklets = (SELECT COUNT(*) FROM tracklets t WHERE t.gid=identities.gid),
                 has_face    = EXISTS(SELECT 1 FROM tracklets t
                                      WHERE t.gid=identities.gid AND t.face_quality IS NOT NULL)"""
        )


# --------------------------------------------------------------------------- samples (v2)


def add_samples(
    conn: sqlite3.Connection, tracklet_id: int, rows: Sequence[tuple[int, float]]
) -> list[int]:
    """Insert ``(frame, quality)`` samples of a tracklet; returns their ids (vector keys)."""
    ids: list[int] = []
    with transaction(conn):
        for frame, quality in rows:
            cur = conn.execute(
                "INSERT INTO samples(tracklet_id, frame, quality) VALUES (?,?,?)",
                (tracklet_id, int(frame), float(quality)),
            )
            assert cur.lastrowid is not None
            ids.append(cur.lastrowid)
    return ids


def set_in_gallery(conn: sqlite3.Connection, sample_ids: Iterable[int], value: bool) -> None:
    """Flag samples as in (or out of) their identity's gallery."""
    ids = [(int(value), int(i)) for i in sample_ids]
    if ids:
        with transaction(conn):
            conn.executemany("UPDATE samples SET in_gallery=? WHERE id=?", ids)


def get_samples(
    conn: sqlite3.Connection,
    *,
    tracklet_id: int | None = None,
    gid: int | None = None,
    in_gallery: bool | None = None,
) -> list[sqlite3.Row]:
    """Samples (id, tracklet_id, frame, quality, in_gallery, gid) ordered by id.

    ``gid`` filters on the *tracklet's* current gid.
    """
    where: list[str] = []
    args: list[object] = []
    if tracklet_id is not None:
        where.append("s.tracklet_id=?")
        args.append(tracklet_id)
    if gid is not None:
        where.append("t.gid=?")
        args.append(gid)
    if in_gallery is not None:
        where.append("s.in_gallery=?")
        args.append(int(in_gallery))
    sql = (
        "SELECT s.id, s.tracklet_id, s.frame, s.quality, s.in_gallery, t.gid "
        "FROM samples s JOIN tracklets t ON t.id = s.tracklet_id"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    return conn.execute(sql + " ORDER BY s.id", args).fetchall()


def add_cannot_link(conn: sqlite3.Connection, gid_a: int, gid_b: int, reason: str = "") -> None:
    """Persist a cannot-link between two identities (stored with ``gid_a < gid_b``)."""
    a, b = sorted((int(gid_a), int(gid_b)))
    if a != b:
        conn.execute(
            "INSERT OR REPLACE INTO identity_cannot_links(gid_a, gid_b, reason) VALUES (?,?,?)",
            (a, b, reason),
        )


def get_cannot_links(conn: sqlite3.Connection) -> set[tuple[int, int]]:
    """All persisted identity cannot-links as ``(a, b)`` with ``a < b``."""
    return {
        (r["gid_a"], r["gid_b"])
        for r in conn.execute("SELECT gid_a, gid_b FROM identity_cannot_links")
    }


def remap_cannot_links(conn: sqlite3.Connection, old: int, new: int) -> None:
    """After merging identity ``old`` into ``new``, rewrite cannot-links that mention ``old``."""
    rows = conn.execute(
        "SELECT gid_a, gid_b, reason FROM identity_cannot_links WHERE gid_a=? OR gid_b=?",
        (old, old),
    ).fetchall()
    with transaction(conn):
        conn.execute("DELETE FROM identity_cannot_links WHERE gid_a=? OR gid_b=?", (old, old))
        for r in rows:
            other = r["gid_b"] if r["gid_a"] == old else r["gid_a"]
            if other != new:
                a, b = sorted((other, new))
                conn.execute(
                    "INSERT OR REPLACE INTO identity_cannot_links VALUES (?,?,?)",
                    (a, b, r["reason"]),
                )
