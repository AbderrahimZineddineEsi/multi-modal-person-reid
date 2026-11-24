"""End-to-end extraction on a synthetic video with a fake detector and fake embedder."""

import cv2
import numpy as np
import pytest

pytest.importorskip("boxmot")

from reid.config import CameraSet, CameraSpec, Config  # noqa: E402
from reid.models.registry import Models  # noqa: E402
from reid.pipeline import run  # noqa: E402
from reid.storage import db, files  # noqa: E402

W, H, FPS, N = 320, 240, 10.0, 40


def make_video(path, fps=FPS):
    """Two rectangles on separate rows walking in opposite directions on a black background."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(N):
        fr = np.zeros((H, W, 3), np.uint8)
        fr[10:110, 20 + 4 * i : 70 + 4 * i] = 255
        fr[130:230, 250 - 4 * i : 300 - 4 * i] = (0, 200, 255)
        vw.write(fr)
    vw.release()


class FakeDetector:
    def load(self):
        return self

    def detect(self, frame):
        mask = (frame.max(axis=2) > 100).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(mask)
        boxes = [(x, y, x + w, y + h) for x, y, w, h, _ in st[1:]]
        kps = np.tile(np.array([0, 0, 0.9], np.float32), (len(boxes), 17, 1))
        return (
            np.array(boxes, np.float32).reshape(-1, 4),
            np.full(len(boxes), 0.9, np.float32),
            kps if boxes else np.zeros((0, 17, 3), np.float32),
        )


class FakeBody:
    dim = 8

    def load(self):
        return self

    def embed(self, crops):
        return np.tile(np.eye(8, dtype=np.float32)[0], (len(crops), 1))


class FakeEngine:
    def __init__(self):
        self.calls = []

    def on_tracklet(self, t):
        assert t.id is not None and t.run_id is not None
        self.calls.append(("tracklet", t.camera_id, t.id))
        return None

    def end_camera(self, camera_id):
        self.calls.append(("end_camera", camera_id))

    def end_run(self, run_id):
        self.calls.append(("end_run", run_id))
        return "report"


def make_cfg(tmp_path, **over):
    base = {
        "paths": {"data_dir": str(tmp_path / "data")},
        "quality": {"min_box_h": 50, "sample_every": 3, "min_det_conf": 0.5},
        "tracklets": {"max_samples_per_tracklet": 4},
        "tracking": {"tracker": "bytetrack"},
        "body_reid": {"dim": 8},
        "runtime": {"batch_size": 4},
    }
    for k, v in over.items():
        base.setdefault(k, {}).update(v)
    return Config.model_validate(base)


def fake_models(cfg):
    return Models(detector=FakeDetector(), pose=None, body=FakeBody(), face=None)


@pytest.fixture
def cams(tmp_path):
    make_video(tmp_path / "a.mp4")
    make_video(tmp_path / "b.mp4")
    return CameraSet(
        cameras=[
            CameraSpec(id="a", video_path=str(tmp_path / "a.mp4")),
            CameraSpec(id="b", video_path=str(tmp_path / "b.mp4"), start_ts=1000.0),
        ]
    )


def test_extraction_only(tmp_path, cams):
    cfg = make_cfg(tmp_path)
    calls = []
    res = run(cfg, cams, None, lambda c, f, t: calls.append((c, f, t)), models=fake_models(cfg))
    assert res.frames_processed == 2 * N and res.tracklets >= 4
    assert calls[0][0] == "a" and calls[-1][0] == "b" and calls[-1][1] == N
    conn = db.connect(files.db_path(cfg.data_dir))
    assert db.get_run(conn, res.run_id)["status"] == "done"
    assert {c.id for c in db.list_cameras(conn)} == {"a", "b"}
    assert db.get_camera(conn, "a").fps == pytest.approx(FPS)
    tls = db.get_tracklets(conn, run_id=res.run_id)
    assert all(t.gid is None for t in tls) and res.tracklets_unidentified == len(tls)
    for t in tls:
        pts = db.get_tracklet_points(conn, t.id)
        assert len(pts) == t.n_points == t.end_frame - t.start_frame + 1  # every frame persisted
        assert t.start_ts == pytest.approx(
            (1000.0 if t.camera_id == "b" else 0.0) + t.start_frame / FPS
        )
        assert int(pts["sampled"].sum()) == t.n_samples <= 4
        assert t.best_crop_path and (cfg.data_dir / t.best_crop_path).exists()
    conn.close()
    assert res.ms_per_frame["detect"] >= 0 and set(res.ms_per_frame) >= {"detect", "track", "reid"}


def test_engine_callbacks_order_and_stride_limits(tmp_path, cams):
    cfg = make_cfg(tmp_path, runtime={"frame_stride": 2, "start_s": 1.0, "end_s": 3.0})
    eng = FakeEngine()
    res = run(cfg, cams, eng, models=fake_models(cfg))
    kinds = [c[0] for c in eng.calls]
    assert kinds[-1] == "end_run" and eng.calls[-1][1] == res.run_id
    assert kinds.index("end_camera") > kinds.index("tracklet")
    ends = [c[1] for c in eng.calls if c[0] == "end_camera"]
    assert ends == ["a", "b"]
    assert all(c[1] == "a" for c in eng.calls[: kinds.index("end_camera")] if c[0] == "tracklet")
    assert res.frames_processed == 2 * 10  # frames 10..29 step 2 per camera
    assert res.identity_report == "report"
    conn = db.connect(files.db_path(cfg.data_dir))
    for t in db.get_tracklets(conn, run_id=res.run_id):
        assert 10 <= t.start_frame and t.end_frame <= 28
    conn.close()


def test_max_frames_and_missing_video(tmp_path, cams):
    cfg = make_cfg(tmp_path, runtime={"max_frames": 12})
    assert run(cfg, cams, None, models=fake_models(cfg)).frames_processed == 24
    bad = CameraSet(cameras=[CameraSpec(id="x", video_path=str(tmp_path / "nope.mp4"))])
    with pytest.raises(FileNotFoundError):
        run(cfg, bad, None, models=fake_models(cfg))
    conn = db.connect(files.db_path(cfg.data_dir))
    assert db.list_runs(conn)[-1]["status"] == "failed"
    conn.close()
