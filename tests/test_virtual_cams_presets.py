"""Virtual-camera geometry and preset merging."""

import cv2
import numpy as np
import pytest
import yaml

from reid.config import Config, load_cameras, load_config, preset_path
from reid.virtual_cams import make_virtual_cameras, plan_cameras


@pytest.mark.parametrize("n,overlap", [(2, 0.15), (3, 0.25), (4, 0.0), (2, 0.6)])
def test_plan_geometry(n, overlap):
    cams = plan_cameras(3840, 2160, n, overlap)
    assert len(cams) == n and cams[0].x == 0
    assert cams[-1].x + cams[-1].w == 3840  # full coverage
    assert all(c.w % 2 == 0 and c.h % 2 == 0 for c in cams)
    for a, b in zip(cams, cams[1:], strict=False):
        assert (a.x + a.w - b.x) / a.w == pytest.approx(overlap, abs=0.01)


def test_plan_options():
    cams = plan_cameras(1000, 500, 3, 0.2, mirror=[1], delays=[0, 2.5], out_width=200)
    assert [c.mirror for c in cams] == [False, True, False]
    assert [c.delay_s for c in cams] == [0.0, 2.5, 0.0]
    assert cams[0].out_w == 200 and cams[0].out_h % 2 == 0
    assert cams[0].out_h == pytest.approx(500 * 200 / cams[0].w, abs=2)
    with pytest.raises(ValueError):
        plan_cameras(100, 100, 0)
    with pytest.raises(ValueError):
        plan_cameras(100, 100, 2, overlap=1.0)


def test_make_virtual_cameras_writes_videos_and_yaml(tmp_path):
    src = tmp_path / "src.mp4"
    vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 96))
    for i in range(20):
        vw.write(np.full((96, 160, 3), i * 10, np.uint8))
    vw.release()
    yp = make_virtual_cameras(
        src,
        tmp_path / "out",
        n=2,
        overlap=0.25,
        mirror=[1],
        delays=[0, 3.0],
        out_width=64,
        end_s=1.0,
        face_cams=[0],
    )
    cs = load_cameras(yp)
    assert [c.id for c in cs.cameras] == ["cam_virt_0", "cam_virt_1"]
    assert cs.cameras[1].start_ts == 3.0 and cs.cameras[0].is_face_cam
    assert not cs.cameras[1].is_face_cam and cs.cameras[0].fps == pytest.approx(10.0)
    cap = cv2.VideoCapture(cs.cameras[0].video_path)
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
    assert yaml.safe_load(open(yp))["cameras"][0]["video_path"].endswith("cam_virt_0.mp4")


@pytest.mark.parametrize("name", ["lite", "balanced", "accurate", "max"])
def test_presets_are_valid_configs(name):
    cfg = load_config(preset=name)
    assert preset_path(name).exists()
    expected_dim = 512 if name == "lite" else 1280
    assert cfg.body_reid.dim == expected_dim


def test_balanced_preset_equals_defaults():
    assert load_config(preset="balanced") == Config()


def test_preset_then_user_then_set_precedence(tmp_path):
    user = tmp_path / "user.yaml"
    user.write_text("detection:\n  conf: 0.4\n  imgsz: 800\nface:\n  model: buffalo_l\n")
    cfg = load_config(user, {"detection.imgsz": 1024}, preset="lite")
    assert cfg.detection.model == "yolo26n-pose.pt"  # from preset
    assert cfg.detection.conf == 0.4  # user over default
    assert cfg.face.model == "buffalo_l"  # user over preset (lite says buffalo_sc)
    assert cfg.detection.imgsz == 1024  # --set over user over preset
    assert cfg.tracking.tracker == "bytetrack"
    with pytest.raises(FileNotFoundError):
        preset_path("nope")


def test_preset_model_slots():
    assert load_config(preset="lite").quality.pose_model is None
    acc = load_config(preset="accurate")
    assert acc.detection.model.startswith("rfdetr") and acc.quality.pose_model == "yolo26s-pose.pt"
    assert load_config(preset="max").tracking.tracker == "occluboost"
