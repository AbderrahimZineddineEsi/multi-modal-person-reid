"""Sync decisions, transfers (directory, rclone with a fake runner) and the URL downloader."""

from __future__ import annotations

import http.server
import json
import subprocess
import threading
from functools import partial

import numpy as np
import pytest
from test_portable import make_env

from reid import sync as sy
from reid.app import download
from reid.storage import db, files, portable
from reid.types import CameraInfo

# ------------------------------------------------------------------ pure decisions


@pytest.mark.parametrize(
    ("src", "dst", "base", "force", "action"),
    [
        (None, "a", "a", False, "missing"),
        ("a", None, None, False, "create"),
        ("a", "a", None, False, "unchanged"),
        ("b", "a", "a", False, "copy"),
        ("b", "a", None, False, "conflict"),
        ("a", "b", "a", False, "behind"),
        ("b", "c", "a", False, "conflict"),
        ("b", "c", "a", True, "overwrite"),
        ("a", "b", "a", True, "overwrite"),
    ],
)
def test_decide(src, dst, base, force, action):
    d = sy.decide(src, dst, base, force=force)
    assert d.action == action
    assert d.refused == (action in ("conflict", "behind"))


def test_synced_file_filter():
    ok = sy.is_synced_file
    assert ok("reid.db") and ok("vectors/a.npy") and ok("uploads/v.mp4")
    assert not ok("reid.db-wal") and not ok("reid.db-shm") and not ok(".sync_state.json")
    assert not ok("renders/m.mp4") and ok("renders/m.mp4", with_renders=True)
    assert not ok("uploads/v.mp4.part")


def test_remote_detection(tmp_path):
    assert sy.looks_like_remote("gdrive:reid_data/environments")
    assert not sy.looks_like_remote("C:\\data") and not sy.looks_like_remote(str(tmp_path))
    with pytest.raises(sy.SyncError, match="rclone"):
        sy.make_transfer("gdrive:x", which=lambda _n: None)
    assert isinstance(
        sy.make_transfer("gdrive:x", which=lambda _n: "/usr/bin/rclone"), sy.RcloneTransfer
    )
    assert isinstance(sy.make_transfer(str(tmp_path)), sy.DirTransfer)


# ------------------------------------------------------------------ directory round trips


def add_camera(env, cid):
    conn = db.connect(files.db_path(env))
    db.upsert_camera(conn, CameraInfo(id=cid, name=cid, video_path="uploads/x.mp4", fps=10.0))
    conn.close()


def run_sync(local, remote, direction, **kw):
    return sy.sync(sy.DirTransfer(local), sy.DirTransfer(remote), local, direction, **kw)


def test_push_pull_roundtrip_and_conflict(tmp_path):
    pc, drive = tmp_path / "pc", tmp_path / "drive"
    env = make_env(pc / "e1")
    (env / "vectors").mkdir(exist_ok=True)
    np.save(env / "vectors" / "body.npy", np.zeros((3, 4), np.float32))
    (env / "uploads").mkdir(exist_ok=True)
    (env / "uploads" / "x.mp4").write_bytes(b"video")

    rep = run_sync(pc, drive, "push", dry_run=True)
    assert rep.envs[0].decision.action == "create" and not (drive / "e1").exists()
    rep = run_sync(pc, drive, "push")
    assert rep.envs[0].done and (drive / "e1" / "uploads" / "x.mp4").read_bytes() == b"video"
    assert not list((drive / "e1").glob("reid.db-*")) and not (drive / "e1" / "renders").exists()
    assert not (drive / "e1" / sy.STATE_FILE).exists()
    assert run_sync(pc, drive, "push").envs[0].decision.action == "unchanged"
    assert run_sync(pc, drive, "pull").envs[0].decision.action == "unchanged"

    add_camera(drive / "e1", "cam_02")  # work on the "drive" side
    np.save(drive / "e1" / "vectors" / "body.npy", np.zeros((5, 4), np.float32))
    assert run_sync(pc, drive, "push").envs[0].decision.action == "behind"  # protected
    rep = run_sync(pc, drive, "pull")
    assert rep.envs[0].decision.action == "copy" and rep.envs[0].done
    assert {r.cameras for r in portable.list_envs(pc)} == {2}
    assert np.load(pc / "e1" / "vectors" / "body.npy").shape == (5, 4)

    add_camera(pc / "e1", "cam_03")  # both sides change
    add_camera(drive / "e1", "cam_04")
    for direction in ("push", "pull"):
        rep = run_sync(pc, drive, direction)
        assert rep.refused and not rep.envs[0].done
    assert {r.cameras for r in portable.list_envs(drive)} == {3}  # untouched
    rep = run_sync(pc, drive, "push", force=True)
    bak = rep.envs[0].backup
    assert bak and bak.startswith("e1.bak-") and (drive / bak / "reid.db").is_file()
    assert {r.cameras for r in portable.list_envs(drive) if r.name == "e1"} == {3}
    assert (drive / "e1" / "uploads" / "x.mp4").exists()
    assert json.loads((pc / "e1" / sy.STATE_FILE).read_text())["targets"]


def test_new_env_pull_and_delete(tmp_path):
    pc, drive = tmp_path / "pc", tmp_path / "drive"
    make_env(drive / "colab_run")
    make_env(pc / "old")
    rep = run_sync(pc, drive, "pull")
    assert [e.env for e in rep.envs] == ["colab_run"] and (pc / "colab_run" / "reid.db").is_file()
    assert (pc / "old").exists()  # nothing deleted without --delete
    rep = run_sync(pc, drive, "pull", delete=True)
    assert not (pc / "old").exists() and list(pc.glob("old.bak-*"))
    with pytest.raises(sy.SyncError, match="no such"):
        run_sync(pc, drive, "pull", envs=["nope"])


# ------------------------------------------------------------------ rclone with a fake runner


class FakeRclone:
    """Serves a directory as if it were ``remote:base`` (lsjson, copyto, deletefile, moveto)."""

    def __init__(self, root):
        self.root, self.calls = root, []

    def __call__(self, argv):
        self.calls.append(argv)
        cmd, args = argv[1], argv[2:]
        ok = subprocess.CompletedProcess(argv, 0, "", "")

        def loc(p):
            return self.root / p.split(":", 1)[1].lstrip("/")

        if cmd == "lsjson":
            base = loc(args[-1])
            rec = "-R" in args
            items = []
            for f in sorted(base.rglob("*") if rec else base.glob("*")):
                if f.is_file():
                    st = f.stat()
                    items.append(
                        {
                            "Path": f.relative_to(base).as_posix(),
                            "Name": f.name,
                            "Size": st.st_size,
                            "ModTime": "2026-01-01T00:00:00Z",
                        }
                    )
            if "--max-depth" in args:
                items = [i for i in items if i["Path"].count("/") <= 1]
            return subprocess.CompletedProcess(argv, 0, json.dumps(items), "")
        if cmd == "copyto":
            a, b = args
            src = loc(a) if ":" in a and not a.startswith("/") else __import__("pathlib").Path(a)
            dst = loc(b) if ":" in b and not b.startswith("/") else __import__("pathlib").Path(b)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            return ok
        if cmd == "deletefile":
            loc(args[0]).unlink()
            return ok
        if cmd == "moveto":
            loc(args[0]).rename(loc(args[1]))
            return ok
        raise AssertionError(argv)


def test_rclone_transfer_roundtrip(tmp_path):
    pc, store = tmp_path / "pc", tmp_path / "store"
    make_env(pc / "e1")
    fake = FakeRclone(store)
    remote = sy.RcloneTransfer("gdrive:reid/environments", "rclone", fake)
    # remote path reid/environments lives under store/reid/environments
    rep = sy.sync(sy.DirTransfer(pc), remote, pc, "push")
    assert rep.envs[0].done and (store / "reid" / "environments" / "e1" / "reid.db").is_file()
    assert remote.list_envs() == ["e1"]
    assert sy.sync(sy.DirTransfer(pc), remote, pc, "push").envs[0].decision.action == "unchanged"
    add_camera(store / "reid" / "environments" / "e1", "cam_09")
    rep = sy.sync(sy.DirTransfer(pc), remote, pc, "pull")
    assert rep.envs[0].decision.action == "copy"
    assert {r.cameras for r in portable.list_envs(pc)} == {2}
    assert all(c[0] == "rclone" for c in fake.calls)


def test_rclone_refuses_unmerged_wal(tmp_path):
    env = make_env(tmp_path / "store" / "e1")
    (env / "reid.db-wal").write_bytes(b"x" * 10)
    remote = sy.RcloneTransfer("gdrive:", "rclone", FakeRclone(tmp_path / "store"))
    remote.base = "gdrive:"
    with pytest.raises(sy.SyncError, match="WAL"):
        remote.snapshot_db("e1", tmp_path)


# ------------------------------------------------------------------ URL download


@pytest.fixture
def web(tmp_path):
    root = tmp_path / "www"
    root.mkdir()
    (root / "clip.mp4").write_bytes(b"0" * 5000)
    (root / "page.html").write_text("<html></html>")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(
        http.server.SimpleHTTPRequestHandler, directory=str(root)))  # fmt: skip
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_download(web, tmp_path):
    seen = []
    p = download.download_video(
        f"{web}/clip.mp4", tmp_path / "up", progress=lambda a, b: seen.append((a, b))
    )
    assert p.name == "clip.mp4" and p.stat().st_size == 5000 and seen[-1] == (5000, 5000)
    assert download.download_video(f"{web}/clip.mp4", tmp_path / "up").name == "clip_1.mp4"
    assert not list((tmp_path / "up").glob("*.part"))
    with pytest.raises(download.DownloadError, match="not a video"):
        download.download_video(f"{web}/page.html", tmp_path / "up")
    with pytest.raises(download.DownloadError, match="limit"):
        download.download_video(f"{web}/clip.mp4", tmp_path / "up", max_bytes=100)
    with pytest.raises(download.DownloadError, match="video page"):
        download.download_video("https://www.youtube.com/watch?v=x", tmp_path / "up")
    with pytest.raises(download.DownloadError, match="http"):
        download.download_video("file:///etc/passwd", tmp_path / "up")
    assert download.parse_urls(" a \n\nb\na") == ["a", "b"]


def test_uploads_resolve_after_move(tmp_path):
    from reid.viz.video_io import resolve_video

    env = make_env(tmp_path / "e", video=tmp_path / "e" / "uploads" / "v.mp4")
    (env / "uploads").mkdir(exist_ok=True)
    (env / "uploads" / "v.mp4").write_bytes(b"v")
    assert portable.relativize_videos(env) == 1
    conn = db.connect(files.db_path(env))
    assert db.list_cameras(conn)[0].video_path == "uploads/v.mp4"
    conn.close()
    moved = tmp_path / "moved"
    env.rename(moved)
    assert resolve_video("uploads/v.mp4", moved) == moved / "uploads" / "v.mp4"
    assert resolve_video("/content/drive/x/v.mp4", moved, "cam_01").name == "v.mp4"  # by name
