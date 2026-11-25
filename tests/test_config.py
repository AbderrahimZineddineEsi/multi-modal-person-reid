from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from reid.config import Config, deep_merge, expand_dotted, load_cameras, load_config
from reid.config_yaml import render_default_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_ported_from_v1():
    cfg = load_config()
    assert cfg.association.min_cos == 0.78
    assert cfg.association.tau == 2.0
    assert cfg.association.face_match == 0.55
    assert cfg.face.det_conf == 0.7
    assert cfg.quality.kp_conf == 0.3
    assert cfg.quality.min_visible_ratio == 0.5
    assert cfg.association.proto_k == 3
    assert cfg.search.tau == 1.5


def test_default_yaml_matches_models():
    path = ROOT / "configs" / "default.yaml"
    assert path.read_text() == render_default_yaml(), "regenerate configs/default.yaml"
    assert load_config(path) == Config()


def test_default_yaml_comments_every_parameter():
    lines = (ROOT / "configs" / "default.yaml").read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("  ") and not line.lstrip().startswith("#") and ":" in line:
            j = i - 1
            while not lines[j].lstrip().startswith("#"):
                j -= 1
            assert "default:" in "\n".join(lines[max(j - 2, 0) : i]), line


def test_partial_file_and_dotted_overrides(tmp_path: Path):
    f = tmp_path / "c.yaml"
    f.write_text(
        yaml.safe_dump({"association": {"min_cos": 0.8, "tau": 3.0}, "paths.data_dir": "x"})
    )
    cfg = load_config(f, {"association.min_cos": 0.9, "search": {"top_k": 4}})
    assert cfg.association.min_cos == 0.9  # override beats file
    assert cfg.association.tau == 3.0  # file beats default (deep merge keeps siblings)
    assert cfg.association.face_match == 0.55
    assert cfg.search.top_k == 4
    assert cfg.paths.data_dir == "x"


def test_unknown_and_invalid_keys_rejected():
    with pytest.raises(ValidationError):
        load_config(None, {"association.min_cosine": 0.8})
    with pytest.raises(ValidationError):
        load_config(None, {"quality.kp_conf": 2.0})


def test_merge_helpers_do_not_mutate():
    base = {"a": {"b": 1, "c": 2}}
    out = deep_merge(base, {"a": {"b": 5}})
    assert base["a"]["b"] == 1 and out == {"a": {"b": 5, "c": 2}}
    assert expand_dotted({"a.b.c": 1, "a.b.d": 2}) == {"a": {"b": {"c": 1, "d": 2}}}


def test_load_cameras(tmp_path: Path):
    cs = load_cameras(ROOT / "configs" / "cameras" / "phones.example.yaml")
    assert [c.id for c in cs.cameras] == ["cam_entrance", "cam_aisle"]
    assert cs.get("cam_entrance").is_face_cam and cs.get("cam_aisle").start_ts == 0.0
    assert cs.min_transit("cam_aisle", "cam_entrance") == 3.0  # symmetric
    assert cs.min_transit("cam_aisle", "cam_aisle") is None
    assert cs.get("cam_aisle").name == "cam_aisle" or cs.get("cam_aisle").name == "Aisle"


def test_camera_validation(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("cameras: [{id: a, video_path: v.mp4}, {id: a, video_path: w.mp4}]\n")
    with pytest.raises(ValidationError):
        load_cameras(bad)
    bad.write_text("cameras: [{id: a, video_path: v.mp4, homography: [[1,0],[0,1]]}]\n")
    with pytest.raises(ValidationError):
        load_cameras(bad)
    bad.write_text(
        "cameras: [{id: a, video_path: v.mp4}]\ntopology: [{from: a, to: zz, min_transit_s: 1}]\n"
    )
    with pytest.raises(ValidationError):
        load_cameras(bad)


def test_camera_paths_relative_to_the_yaml_folder(tmp_path, monkeypatch):
    from reid.config import load_cameras

    (tmp_path / "set").mkdir()
    (tmp_path / "set" / "cam.mp4").write_bytes(b"")
    y = tmp_path / "set" / "cams.yaml"
    y.write_text("cameras:\n  - id: c1\n    name: c1\n    video_path: cam.mp4\n")
    monkeypatch.chdir(tmp_path)  # cam.mp4 does not exist relative to the working directory
    assert load_cameras(y).cameras[0].video_path == str(tmp_path / "set" / "cam.mp4")
