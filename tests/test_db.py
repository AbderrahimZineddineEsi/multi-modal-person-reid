import time

import numpy as np
import pytest

from reid.storage import db
from reid.types import CameraInfo, Tracklet, TrackPoint


def _cam(conn, cid="c1"):
    db.upsert_camera(
        conn, CameraInfo(id=cid, name=cid, video_path="v.mp4", fps=25.0, width=640, height=360)
    )


def _tracklet(cam, tid, start, n, gid=None, face=None):
    pts = [
        TrackPoint(
            frame=start + i,
            ts=(start + i) / 25.0,
            x1=i,
            y1=2.0,
            x2=i + 10.0,
            y2=50.0,
            conf=0.9,
            sampled=i % 5 == 0,
        )
        for i in range(n)
    ]
    return Tracklet(
        camera_id=cam, local_track_id=tid, start_frame=start, end_frame=start + n - 1,
        start_ts=start / 25.0, end_ts=(start + n - 1) / 25.0, n_samples=3, mean_quality=0.6,
        gid=gid, face_quality=face, points=pts,
    )  # fmt: skip


def test_schema_version_and_run_camera(conn):
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    rid = db.create_run(conn, '{"a": 1}')
    assert db.get_run(conn, rid)["status"] == "running"
    db.set_run_status(conn, rid, "done")
    assert db.get_run(conn, rid)["status"] == "done"
    h = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    cam = CameraInfo(
        id="c1", name="One", video_path="v.mp4", fps=29.97, width=1, height=2,
        start_ts=5.0, is_face_cam=True, homography=h,
    )  # fmt: skip
    db.upsert_camera(conn, cam)
    assert db.get_camera(conn, "c1") == cam
    cam.name = "Renamed"
    db.upsert_camera(conn, cam)
    assert db.get_camera(conn, "c1").name == "Renamed" and len(db.list_cameras(conn)) == 1


def test_100k_points_roundtrip(conn):
    _cam(conn)
    _cam(conn, "c2")
    rid = db.create_run(conn, "{}")
    tracklets = [_tracklet("c1", i, i * 1000, 1000) for i in range(100)]  # 100k points
    t0 = time.perf_counter()
    ids = db.insert_tracklets(conn, rid, tracklets)
    assert time.perf_counter() - t0 < 10
    db.insert_tracklets(conn, rid, [_tracklet("c2", 1, 0, 50)])
    assert ids == [t.id for t in tracklets] and len(set(ids)) == 100
    assert db.table_counts(conn)["track_points"] == 100_050

    pts = db.get_points(conn, "c1", 2_500, 2_999)
    assert len(pts) == 500 and (np.diff(pts["frame"]) == 1).all()
    assert (pts["tracklet_id"] == ids[2]).all()
    cross = db.get_points(conn, "c1", 995, 1004)  # spans two tracklets
    assert len(cross) == 10 and set(cross["tracklet_id"]) == {ids[0], ids[1]}
    assert len(db.get_points(conn, "c2", 0, 10_000)) == 50
    assert len(db.get_points(conn, "c1", 0, 10, run_id=rid + 1)) == 0
    one = db.get_tracklet_points(conn, ids[7])
    assert len(one) == 1000 and one["sampled"].sum() == 200
    assert one["x2"][3] == pytest.approx(13.0) and one["conf"][0] == pytest.approx(0.9)


def test_tracklet_queries_and_gids(conn):
    _cam(conn)
    _cam(conn, "c2")
    rid = db.create_run(conn, "{}")
    ts = [
        _tracklet("c1", 1, 0, 10),
        _tracklet("c1", 2, 100, 10, gid=7),
        _tracklet("c2", 1, 50, 10, face=0.8),
    ]
    a, b, c = db.insert_tracklets(conn, rid, ts)
    assert [t.id for t in db.get_tracklets(conn, camera_id="c1")] == [a, b]
    assert [t.id for t in db.get_tracklets(conn, gid=7)] == [b]
    assert [t.id for t in db.get_tracklets(conn, run_id=rid, unassigned=True)] == [a, c]
    assert db.get_tracklet(conn, c).face_quality == 0.8 and db.get_tracklet(conn, 999) is None

    assert db.set_gids(conn, [(a, 1, 0.9, "body"), (c, 1, 0.8, "face")], reason="online") == 2
    assert db.set_gids(conn, [(a, 1, 0.95, "body")]) == 0  # unchanged gid/source -> no history
    assert db.set_gids(conn, [(a, 2, 0.5, "consolidate"), (b, 2, 0.5, "consolidate")], "merge") == 2
    hist = db.get_gid_history(conn, a)
    assert [(h["old_gid"], h["new_gid"], h["reason"]) for h in hist] == [
        (None, 1, "online"),
        (1, 2, "merge"),
    ]
    assert hist[1]["old_source"] == "body" and hist[1]["new_source"] == "consolidate"
    assert [h["new_gid"] for h in db.get_gid_history(conn, b)] == [2]
    assert db.get_tracklet(conn, a).gid == 2
    with pytest.raises(KeyError):
        db.set_gids(conn, [(999, 1, None, "body")])
    assert db.get_tracklet(conn, a).gid == 2  # rolled back / untouched


def test_identities(conn):
    _cam(conn)
    rid = db.create_run(conn, "{}")
    ids = db.insert_tracklets(
        conn, rid, [_tracklet("c1", 1, 0, 10, face=0.7), _tracklet("c1", 2, 100, 10)]
    )
    db.set_gids(conn, [(ids[0], 3, 1.0, "face"), (ids[1], 3, 0.9, "body")])
    assert db.next_gid(conn) == 4
    g = db.create_identity(conn, label="alice")
    assert g == 4 and db.next_gid(conn) == 5
    db.refresh_identity_stats(conn)
    assert [i.gid for i in db.list_identities(conn)] == [3, 4]
    i3 = db.get_identity(conn, 3)
    assert i3.n_tracklets == 2 and i3.has_face and i3.first_ts == 0.0
    assert i3.last_ts == pytest.approx(109 / 25.0)
    assert db.get_identity(conn, 4).n_tracklets == 0
    db.update_identity(conn, 4, label="bob", has_face=1)
    assert db.get_identity(conn, 4).label == "bob" and db.get_identity(conn, 4).has_face
    with pytest.raises(ValueError):
        db.update_identity(conn, 4, bogus=1)
    with pytest.raises(KeyError):
        db.update_identity(conn, 99, label="x")


def test_transaction_rollback(conn):
    with pytest.raises(RuntimeError), db.transaction(conn):
        db.create_run(conn, "{}")
        raise RuntimeError
    assert db.table_counts(conn)["runs"] == 0


def test_persists_across_connections(tmp_path):
    p = tmp_path / "x" / "reid.db"
    c = db.connect(p)
    db.create_run(c, "{}")
    c.close()
    c = db.connect(p)
    assert len(db.list_runs(c)) == 1 and c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    c.close()
