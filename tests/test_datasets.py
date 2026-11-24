"""Dataset helpers: scene resolution (mocked HF listing), GT conversion, camera YAML."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reid.config import load_cameras
from reid.datasets import aicity24 as ac
from reid.datasets import read_gt, write_camera_yaml, write_gt

FILES = [
    "README.md",
    "MTMC_Tracking_2024/eval/main.py",
    "MTMC_Tracking_2024/val/scene_041/ground_truth.txt",
    "MTMC_Tracking_2024/val/scene_041/camera_0361/video.mp4",
    "MTMC_Tracking_2024/test/scene_071/ground_truth.txt",
    "MTMC_Tracking_2024/test/scene_071/camera_0649/video.mp4",  # corrupt, must be skipped
    "MTMC_Tracking_2024/test/scene_071/camera_0650/video.mp4",
    "MTMC_Tracking_2024/test/scene_071/camera_0650/calibration.json",
    "MTMC_Tracking_2024/test/scene_075/ground_truth.txt",
    "MTMC_Tracking_2024/test/scene_075/camera_0700/video.mp4",
    "MTMC_Tracking_2024/test/scene_075/camera_0701/video.mp4",
    "MTMC_Tracking_2024/test/scene_075/camera_0702/video.mp4",
    "MTMC_Tracking_2024/train/scene_001/camera_0001/video.mp4",  # no ground truth
]


def test_parse_scene_id():
    assert ac.parse_scene_id(75) == ac.parse_scene_id("075") == ac.parse_scene_id("scene_075")
    with pytest.raises(ValueError):
        ac.parse_scene_id("retail")


def test_resolve_scene_paths():
    lay = ac.resolve_scene(FILES, "scene_075")
    assert (lay.split, lay.scene) == ("test", "075")
    assert lay.gt == "MTMC_Tracking_2024/test/scene_075/ground_truth.txt"
    assert lay.cameras[700] == "MTMC_Tracking_2024/test/scene_075/camera_0700/video.mp4"
    assert sorted(lay.cameras) == [700, 701, 702]
    assert sorted(ac.resolve_scene(FILES, 71).cameras) == [650]  # 649 is corrupt
    assert ac.resolve_scene(FILES, 41).split == "val"


def test_resolve_scene_errors():
    with pytest.raises(ValueError, match="available: 041, 071, 075"):
        ac.resolve_scene(FILES, 999)
    with pytest.raises(ValueError, match="no camera videos"):
        ac.resolve_scene(["MTMC_Tracking_2024/val/scene_041/ground_truth.txt"], 41)


def _raw_gt(path: Path) -> Path:
    # <camera> <obj_id> <frame> <x> <y> <w> <h> <xworld> <yworld>
    lines = []
    for f in range(0, 100):
        lines.append(f"700 1 {f} 100 200 50 150 1.0 2.0")  # person 1 on 700 and 701
        lines.append(f"701 1 {f} 400 300 60 160 1.0 2.0")
        lines.append(f"700 2 {f} 800 100 20 64 3.0 4.0")  # short person 2, only on 700
        lines.append(f"701 2 {f} 700 120 22 64 3.0 4.0")
        lines.append(f"702 3 {f} 10 10 30 90 5.0 6.0")  # person 3 only on 702
    path.write_text("\n".join(lines) + "\n")
    return path


def test_load_and_convert_gt(tmp_path):
    raw = ac.load_raw_gt(_raw_gt(tmp_path / "ground_truth.txt"))
    assert list(raw.columns) == ["cam", "gid", "f", "x", "y", "w", "h"] and len(raw) == 500
    cams = {700: "cam_0700", 701: "cam_0701"}
    out = ac.convert_gt(
        raw, cams, max_frame=50, scale={700: (0.5, 0.5), 701: (0.5, 0.5)},
        size={700: (960, 540), 701: (960, 540)},
    )  # fmt: skip
    assert set(out["camera"]) == {"cam_0700", "cam_0701"} and out["frame"].max() == 49
    r = out[(out["camera"] == "cam_0700") & (out["gid"] == 1)].iloc[0]
    assert (r.x1, r.y1, r.x2, r.y2) == (50.0, 100.0, 75.0, 175.0)  # scaled by 0.5
    assert len(out) == 4 * 50  # camera 702 dropped, frames >= 50 dropped
    keep = ac.convert_gt(
        raw, cams, max_frame=50, scale={700: (0.5, 0.5), 701: (0.5, 0.5)},
        size={700: (960, 540), 701: (960, 540)}, min_height=40,
    )  # fmt: skip
    assert set(keep["gid"]) == {1}  # person 2 is 32 px tall after scaling
    clipped = ac.convert_gt(
        raw, {700: "a"}, max_frame=1, scale={700: (1.0, 1.0)}, size={700: (120, 300)}
    )
    assert clipped["x2"].max() == 120 and clipped["y2"].max() == 300


def test_gt_roundtrip(tmp_path):
    df = pd.DataFrame(
        {"camera": ["a", "b"], "frame": [3, 1], "gid": [7, 8],
         "x1": [1.0, 2.0], "y1": [3.0, 4.0], "x2": [5.0, 6.0], "y2": [7.0, 9.5]}
    )  # fmt: skip
    back = read_gt(write_gt(df, tmp_path / "sub" / "gt.csv"))
    assert back["camera"].tolist() == ["a", "b"] and back["y2"].tolist() == [7.0, 9.5]
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        read_gt(tmp_path / "bad.csv")


def test_select_cameras_prefers_overlap_and_few_people():
    rows = []
    for cam, people in {
        1: [1, 2, 3],
        2: [1, 2],
        3: [1, 4],
        4: [7, 8],
        5: [1, 2, 3, 4, 5, 6],
    }.items():
        rows += [(cam, p, f, 0, 0, 50, 200) for p in people for f in range(10)]
    raw = pd.DataFrame(rows, columns=["cam", "gid", "f", "x", "y", "w", "h"])
    cams, stats = ac.select_cameras(raw, [1, 2, 3, 4, 5], 3, max_people=5)
    assert cams == [1, 2, 3] and stats == {"people": 4, "people_in_2plus_cameras": 2}


def test_fetch_with_fake_download_and_trim(tmp_path, monkeypatch):
    calls = []

    def fake_download(repo_path, local_dir):
        calls.append(repo_path)
        p = Path(local_dir) / repo_path
        p.parent.mkdir(parents=True, exist_ok=True)
        if repo_path.endswith(".txt"):
            _raw_gt(p)
        else:
            p.write_bytes(b"x")
        return p

    monkeypatch.setattr(ac, "probe_video", lambda p: ac.VideoInfo(1920, 1080, 30.0, 1000))
    monkeypatch.setattr(
        ac, "trim_video", lambda src, dst, secs, h, **k: (Path(dst).write_bytes(b"y"),
        ac.VideoInfo(1280, 720, 30.0, 60))[1],
    )  # fmt: skip
    out = tmp_path / "ds"
    res = ac.fetch_aicity24(
        75, cameras=2, minutes=2 / 60, out=out, yaml_path=tmp_path / "cams.yaml",
        files=FILES, download=fake_download,
    )  # fmt: skip
    assert calls[0].endswith("ground_truth.txt") and len(calls) == 3  # GT + 2 cameras only
    assert res.cameras == ["cam_0700", "cam_0701"] and res.frames == 60
    assert not (out / "raw").exists() and (out / "meta.json").exists()
    gt = read_gt(res.gt_path)
    assert gt["frame"].max() < 60 and gt["camera"].nunique() == 2
    cs = load_cameras(res.camera_yaml)
    assert [c.id for c in cs.cameras] == ["cam_0700", "cam_0701"]
    assert all(c.start_ts == 0 and not c.is_face_cam for c in cs.cameras)
    with pytest.raises(ValueError, match="not in scene"):
        ac.fetch_aicity24(75, camera_ids=[1], out=out, files=FILES, download=fake_download)
    assert np.isclose(res.fps, 30.0)


def test_write_camera_yaml_from_video_list(tmp_path):
    p = write_camera_yaml(
        tmp_path / "c.yaml", ["a/entrance.mp4", {"video_path": "b.mp4", "id": "aisle"}],
        face_cams=["cam_01"],
    )  # fmt: skip
    cs = load_cameras(p)
    assert [c.id for c in cs.cameras] == ["cam_01", "aisle"]
    assert cs.cameras[0].is_face_cam and cs.cameras[0].name == "entrance"
    assert not cs.cameras[1].is_face_cam and cs.cameras[1].start_ts == 0
