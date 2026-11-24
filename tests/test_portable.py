import json
import zipfile

import pytest

from reid import cli
from reid.config import load_config
from reid.storage import db, files, portable
from reid.types import CameraInfo
from reid.viz.video_io import resolve_video


def make_env(root, preset="balanced", video=None):
    files.ensure_dirs(root)
    (root / "env.json").write_text(json.dumps({"name": root.name}))
    conn = db.connect(files.db_path(root))
    cfg = load_config(preset=preset)
    db.create_run(conn, json.dumps(cfg.model_dump()))
    vp = str(video) if video else "/content/drive/x/cam_01.mp4"
    db.upsert_camera(conn, CameraInfo(id="cam_01", name="c1", video_path=vp, fps=10.0))
    conn.close()
    (root / "crops").mkdir(exist_ok=True)
    (root / "crops" / "a.jpg").write_bytes(b"jpg")
    (root / "renders" / "r.mp4").write_bytes(b"render")
    return root


def test_export_import_roundtrip(tmp_path):
    video = tmp_path / "src" / "cam_01.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video-bytes")
    env = make_env(tmp_path / "env1", video=video)
    res = portable.export_env(env, tmp_path / "a.zip", with_videos=True)
    assert res.videos == {"cam_01": "videos/cam_01.mp4"} and res.size_bytes > 0
    names = zipfile.ZipFile(tmp_path / "a.zip").namelist()
    assert "renders/r.mp4" not in names and "crops/a.jpg" in names and "reid.db" in names

    ws = tmp_path / "ws"
    r = portable.import_env(tmp_path / "a.zip", ws, name="pc")
    assert r.videos_relinked == 1 and not r.videos_missing and r.schema_version == db.SCHEMA_VERSION
    conn = db.connect(files.db_path(ws / "pc"))
    assert db.list_cameras(conn)[0].video_path == "videos/cam_01.mp4"
    conn.close()
    assert resolve_video("videos/cam_01.mp4", ws / "pc").read_bytes() == b"video-bytes"
    assert json.loads((ws / "pc" / "env.json").read_text())["name"] == "pc"
    with pytest.raises(FileExistsError):
        portable.import_env(tmp_path / "a.zip", ws, name="pc")
    portable.import_env(tmp_path / "a.zip", ws, name="pc", force=True)
    (row,) = portable.list_envs(ws)
    assert (row.name, row.cameras, row.runs, row.videos_present) == ("pc", 1, 1, 1)
    assert row.preset == "balanced"


def test_export_without_videos_and_missing_message(tmp_path):
    env = make_env(tmp_path / "e")
    res = portable.export_env(env, tmp_path / "n.zip", with_renders=True)
    assert "renders/r.mp4" in zipfile.ZipFile(tmp_path / "n.zip").namelist()
    assert res.videos == {}
    r = portable.import_env(tmp_path / "n.zip", tmp_path / "ws")
    assert r.name == "e" and r.videos_missing == ["cam_01"]
    with pytest.raises(FileNotFoundError, match="relink"):
        resolve_video("/content/drive/x/cam_01.mp4", r.path, "cam_01")


def test_zip_slip_rejected(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("reid.db", b"")
        zf.writestr("../evil.txt", b"x")
    with pytest.raises(ValueError, match="unsafe"):
        portable.import_env(bad, tmp_path / "ws", name="x")
    assert not (tmp_path / "evil.txt").exists() and not list((tmp_path / "ws").glob("*"))


def test_relink_and_videos_fallback(tmp_path):
    env = make_env(tmp_path / "e")
    vids = tmp_path / "vids" / "sub"
    vids.mkdir(parents=True)
    (vids / "cam_01.mp4").write_bytes(b"v")
    res = portable.relink_env(env, tmp_path / "vids")
    assert list(res.relinked) == ["cam_01"] and not res.missing
    assert portable.relink_env(env, tmp_path / "vids").already_ok == ["cam_01"]
    # data_dir/videos fallback by camera id
    env2 = make_env(tmp_path / "e2")
    (env2 / "videos").mkdir()
    (env2 / "videos" / "cam_01.avi").write_bytes(b"v")
    assert resolve_video("/gone/zzz.mp4", env2, "cam_01").name == "cam_01.avi"
    (env2 / "videos" / "cam_01.mkv").write_bytes(b"v")  # ambiguous: no guess
    with pytest.raises(FileNotFoundError):
        resolve_video("/gone/zzz.mp4", env2, "cam_01")


def test_model_overrides(tmp_path):
    env = make_env(tmp_path / "e", preset="balanced")
    ov = portable.env_model_overrides(env)
    assert ov["body_reid"]["model"] == "clipreid_msmt17" and ov["body_reid"]["dim"] == 1280
    lite = load_config(preset="lite")
    assert lite.body_reid.model != "clipreid_msmt17"
    cfg, note = portable.apply_env_models(lite, env)
    assert cfg.body_reid.model == "clipreid_msmt17" and cfg.body_reid.dim == 1280
    assert note and "body_reid.model" in note
    cfg2, note2 = portable.apply_env_models(cfg, env)
    assert note2 is None and cfg2 is cfg
    empty = tmp_path / "empty"
    files.ensure_dirs(empty)
    assert portable.env_model_overrides(empty) == {}
    assert portable.apply_env_models(lite, empty) == (lite, None)


def test_cli_env(tmp_path, capsys):
    env = make_env(tmp_path / "e")
    assert (
        cli.main(["env", "export", "--data-dir", str(env), "--out", str(tmp_path / "z.zip")]) == 0
    )
    assert (
        cli.main(["env", "import", str(tmp_path / "z.zip"), "--workspace", str(tmp_path / "w")])
        == 0
    )
    assert (
        cli.main(["env", "import", str(tmp_path / "z.zip"), "--workspace", str(tmp_path / "w")])
        == 1
    )
    assert cli.main(["env", "list", "--workspace", str(tmp_path / "w")]) == 0
    out = capsys.readouterr().out
    assert "e " in out and "videos 0/1" in out
    assert cli.main(["env", "relink", "--data-dir", str(env), "--videos", str(tmp_path)]) == 3


def test_processing_into_an_environment_keeps_its_body_model(tmp_path):
    """A lite run (OSNet 512-D) into an environment built with balanced (CLIP-ReID 1280-D) must
    keep CLIP-ReID, otherwise one store would hold vectors of two incompatible models."""
    from reid.config import load_config
    from reid.storage import db, files
    from reid.storage.portable import apply_env_models

    balanced = load_config(None, {"paths.data_dir": str(tmp_path)}, preset="balanced")
    conn = db.connect(files.db_path(tmp_path))
    db.create_run(conn, balanced.to_json(), status="done")
    conn.close()
    lite = load_config(None, {"paths.data_dir": str(tmp_path)}, preset="lite")
    cfg, note = apply_env_models(lite, tmp_path)
    assert cfg.body_reid.model == balanced.body_reid.model and cfg.body_reid.dim == 1280
    assert cfg.detection.model == lite.detection.model  # speed settings still follow the preset
    assert note
