"""AI City official-format export: homographies, interpolation, fusion, format (synthetic DB)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reid.config import load_config
from reid.datasets import aicity24_calib as calib
from reid.eval.aicity_export import export_predictions, interpolate_points
from reid.geometry import apply_homography, invert_homography, scale_homography
from reid.storage import db
from reid.types import CameraInfo, Tracklet, TrackPoint

# world (x, y) -> source pixels (100 px per metre, origin at (500, 300)); a perspective part
# makes the roundtrip test meaningful
H_W2I = np.array([[100.0, 0.0, 500.0], [0.0, 100.0, 300.0], [0.0, 0.0, 1.0]])
H_PERSP = np.array([[100.0, 5.0, 500.0], [2.0, 90.0, 300.0], [0.0004, 0.0002, 1.0]])
SRC = (1920, 1080)


def _floor_h(proc: tuple[int, int]) -> list[list[float]]:
    """Stored (processed-resolution) image-to-floor homography of the affine test camera."""
    return scale_homography(invert_homography(H_W2I), SRC, proc)


def _world(x1: float, x2: float, y2: float, proc: tuple[int, int]) -> tuple[float, float]:
    """Expected world position of a box foot, by hand (source px = processed px * scale)."""
    s = SRC[0] / proc[0]
    return ((((x1 + x2) / 2) * s - 500) / 100, (y2 * s - 300) / 100)


def _points(frames, box) -> list[TrackPoint]:
    return [TrackPoint(f, f / 30, *box(f), 0.9) for f in frames]


def _tracklet(cam: str, ltid: int, gid: int | None, frames, box) -> Tracklet:
    t = Tracklet(cam, ltid, frames[0], frames[-1], frames[0] / 30, frames[-1] / 30, gid=gid)
    t.points = _points(frames, box)
    return t


@pytest.fixture
def env(tmp_path: Path):
    """data_dir with 3 cameras (two downscaled to 1280x720, one full size, one no homography)."""
    cfg = load_config(None, {"paths.data_dir": str(tmp_path), "runtime.frame_stride": 2})
    conn = db.connect(tmp_path / "reid.db")
    for cid, proc, hom in (
        ("cam_0001", (1280, 720), _floor_h((1280, 720))),
        ("cam_0002", (1280, 720), _floor_h((1280, 720))),
        ("cam_0003", (1920, 1080), _floor_h((1920, 1080))),
        ("cam_0004", (1920, 1080), None),
    ):
        db.upsert_camera(conn, CameraInfo(cid, cid, f"{cid}.mp4", 30.0, *proc, homography=hom))
    run_id = db.create_run(conn, cfg.to_json())
    yield tmp_path, conn, run_id
    conn.close()


NUMBERS = {"cam_0001": 701, "cam_0002": 702, "cam_0003": 703, "cam_0004": 704}
SIZES = dict.fromkeys(NUMBERS, SRC)


def test_homography_roundtrip_and_scaling():
    pts = np.array([[100.0, 200.0], [1500.0, 900.0], [960.0, 540.0]])
    floor = apply_homography(invert_homography(H_PERSP), pts)
    assert np.allclose(apply_homography(H_PERSP, floor), pts)
    assert np.isclose(invert_homography(H_PERSP)[2, 2], 1.0)
    # a point in processed pixels gives the same floor position as its source pixel
    h_proc = scale_homography(invert_homography(H_PERSP), SRC, (1280, 720))
    proc_pts = pts * np.array([1280 / 1920, 720 / 1080])
    assert np.allclose(apply_homography(h_proc, proc_pts), floor)
    assert scale_homography(H_W2I, SRC, SRC) == H_W2I.tolist()  # no downscale: unchanged


def test_calibration_loading(tmp_path: Path):
    sensors = [
        {"type": "camera", "id": "Camera_0699", "homography": H_PERSP.tolist()},
        {"type": "camera", "id": "Camera_0700", "homography": [[0.0] * 3] * 3},  # singular
        {"type": "camera", "id": "Camera_0701"},  # none
    ]
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps({"calibrationType": "cartesian", "sensors": sensors}))
    cal = calib.load_calibration(p)
    assert sorted(cal) == [699]
    homs = calib.floor_homographies(cal, [699, 750])  # 750 is not in the file
    assert sorted(homs) == [699]
    assert np.isclose(homs[699][2][2], 1.0)
    assert np.allclose(apply_homography(homs[699], [[500.0, 300.0]]), apply_homography(
        np.linalg.inv(H_PERSP), [[500.0, 300.0]]))  # fmt: skip


def test_official_gt_filter_keeps_format(tmp_path: Path):
    src = tmp_path / "gt.txt"
    lines = [f"{c} 1 {f} 10 20 30 40 1.5 2.5\n" for c in (700, 701, 702) for f in (0, 5, 9)]
    src.write_text("".join(lines))
    n = calib.filter_official_gt(src, tmp_path / "out.txt", [700, 702], max_frame=6)
    assert n == 4
    assert (tmp_path / "out.txt").read_text() == "".join(
        ln for ln in lines if ln.split()[0] in ("700", "702") and int(ln.split()[2]) < 6
    )
    assert calib.filter_official_gt(src, tmp_path / "all.txt", [700, 701, 702], None) == 9


def test_interpolation_gaps_and_no_extrapolation():
    rows = []
    for f, x in ((10, 0.0), (12, 20.0), (16, 60.0), (30, 100.0)):  # gaps 2, 4, 14
        rows.append((1, f, x, 0.0, x + 10, 50.0))
    rows.append((2, 13, 5.0, 5.0, 15.0, 55.0))  # other tracklet, single point
    df = pd.DataFrame(rows, columns=["tracklet_id", "frame", "x1", "y1", "x2", "y2"])
    out = interpolate_points(df, max_gap=4)
    t1 = out[out["tracklet_id"] == 1].set_index("frame")
    assert sorted(t1.index) == [10, 11, 12, 13, 14, 15, 16, 30]  # gap 14 not filled
    assert t1.loc[11, "x1"] == pytest.approx(10.0) and t1.loc[11, "interpolated"]
    assert t1.loc[14, "x1"] == pytest.approx(40.0) and t1.loc[14, "x2"] == pytest.approx(50.0)
    assert not t1.loc[12, "interpolated"] and t1["interpolated"].sum() == 4
    assert out[out["tracklet_id"] == 2]["frame"].tolist() == [13]  # nothing beyond the ends
    assert t1.index.min() == 10 and t1.index.max() == 30
    assert interpolate_points(df, max_gap=1)["interpolated"].sum() == 0  # disabled


def test_export_file_format_fusion_and_skips(env):
    data_dir, conn, run_id = env
    # gid 7 seen by cam_0001 (small box), cam_0002 (large box), cam_0003 (medium) at frames 4 and 6
    boxes = {
        "cam_0001": lambda f: (400.0, 300.0, 460.0, 420.0),
        "cam_0002": lambda f: (450.0, 280.0, 530.0, 430.0),  # largest box in source px? 80x150
        "cam_0003": lambda f: (700.0, 400.0, 760.0, 600.0),
    }
    ts = [
        _tracklet("cam_0001", 1, 7, [4, 6], boxes["cam_0001"]),
        _tracklet("cam_0002", 2, 7, [4, 6], boxes["cam_0002"]),
        _tracklet("cam_0003", 3, 7, [4, 6], boxes["cam_0003"]),
        _tracklet("cam_0001", 4, None, [4, 6], boxes["cam_0001"]),  # unidentified: skipped
        _tracklet("cam_0001", 5, 9, [6], boxes["cam_0001"]),  # a second identity, one frame
    ]
    db.insert_tracklets(conn, run_id, ts)
    out = data_dir / "aicity" / "predictions.txt"
    stats = export_predictions(data_dir, out, camera_numbers=NUMBERS, src_sizes=SIZES)
    assert stats["gids"] == 2 and stats["frames"] == 3  # frames 4, 5 (interpolated), 6
    assert stats["skipped_tracklets"] == 1 and stats["points"] == 7
    assert stats["interpolated_rows"] == 3  # frame 5 on three cameras (stride 2: gap <= 4)
    df = pd.read_csv(out, sep=" ", header=None)
    assert df.shape == (stats["rows"], 9) and stats["rows"] == 4  # (7,4) (7,5) (7,6) (9,6)
    assert df[2].tolist() == sorted(df[2].tolist())  # sorted by frame
    assert (df.groupby([1, 2]).size() == 1).all()  # one row per (gid, frame)
    row = df[(df[1] == 7) & (df[2] == 4)].iloc[0]
    # world = median of three cameras, each from its own (scaled) foot point
    w = [
        _world(400, 460, 420, (1280, 720)),
        _world(450, 530, 430, (1280, 720)),
        _world(700, 760, 600, (1920, 1080)),
    ]
    assert row[7] == pytest.approx(np.median([a for a, _ in w]), abs=1e-3)
    assert row[8] == pytest.approx(np.median([b for _, b in w]), abs=1e-3)
    # the camera with the largest box in source pixels: cam_0002 (80*1.5 x 150*1.5) beats cam_0003
    assert row[0] == NUMBERS["cam_0002"]
    assert list(row[3:7]) == [675, 420, 120, 225]  # box scaled by 1.5 and rounded
    text = out.read_text().splitlines()[0].split(" ")
    assert len(text) == 9 and all(len(t.split(".")[1]) == 3 for t in text[7:])  # 3 decimals
    assert all("." not in t for t in text[:7])  # integers


def test_export_interpolation_follows_stride_and_ends(env):
    data_dir, conn, run_id = env
    box = lambda f: (400.0 + f, 300.0, 460.0 + f, 420.0)  # noqa: E731
    db.insert_tracklets(conn, run_id, [_tracklet("cam_0003", 1, 3, [10, 12, 14, 20], box)])
    out = data_dir / "p.txt"
    stats = export_predictions(data_dir, out, camera_numbers=NUMBERS, src_sizes=SIZES)
    df = pd.read_csv(out, sep=" ", header=None)
    # run stride is 2 -> default gap 4: frames 11, 13 added; the 14 -> 20 gap stays empty
    assert df[2].tolist() == [10, 11, 12, 13, 14, 20] and stats["interpolated_rows"] == 2
    assert df[3].tolist() == [410, 411, 412, 413, 414, 420]
    stats0 = export_predictions(
        data_dir, out, camera_numbers=NUMBERS, src_sizes=SIZES, interpolate_gap=0
    )
    assert stats0["rows"] == 4 and stats0["interpolated_rows"] == 0


def test_export_requires_homography_and_camera_numbers(env):
    data_dir, conn, run_id = env
    db.insert_tracklets(conn, run_id, [_tracklet("cam_0004", 1, 1, [0, 2], lambda f: (1, 1, 5, 9))])
    with pytest.raises(ValueError, match="no homography"):
        export_predictions(data_dir, data_dir / "p.txt", camera_numbers=NUMBERS, src_sizes=SIZES)
    with pytest.raises(ValueError, match="camera_numbers"):
        export_predictions(data_dir, data_dir / "p.txt", camera_numbers={}, src_sizes=SIZES)


def test_pipeline_converts_homography_to_processed_size():
    from reid.config import CameraSpec
    from reid.pipeline import _camera_info

    spec = CameraSpec(id="c", video_path="v.mp4", homography=invert_homography(H_W2I).tolist())
    info = _camera_info(spec, 30.0, 1280, 720, SRC)
    assert np.allclose(info.homography, _floor_h((1280, 720)))
    same = _camera_info(spec, 30.0, 1920, 1080, SRC)  # no downscale: unchanged
    assert same.homography == spec.homography
    assert (
        _camera_info(CameraSpec(id="c", video_path="v.mp4"), 30.0, 1280, 720, SRC).homography
        is None
    )
