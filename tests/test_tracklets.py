"""TrackletBuilder logic with fake detections and fake models (CPU, no weights)."""

from contextlib import contextmanager

import numpy as np
import pytest

from reid.config import Config
from reid.tracking.tracklets import TrackletBuilder, is_truncated, visibility_ratio
from reid.types import CameraInfo

W, H, FPS = 640, 480, 10.0


class FakeBody:
    """Embedding = (mean pixel value, 1) so quality ordering can be traced back."""

    def embed(self, crops):
        e = np.array([[c.mean() + 1.0, 1.0] for c in crops], np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)


class FakeFace:
    """Returns the faces queued in ``results`` (box in region coordinates)."""

    def __init__(self):
        self.results = []

    def analyze(self, img):
        return list(self.results)


class FakeModels:
    def __init__(self, face=None):
        self.body, self.pose, self.face = FakeBody(), None, face

    @contextmanager
    def stage(self, name):
        yield


def full_kps(conf=0.9):
    kp = np.zeros((17, 3), np.float32)
    kp[:, 2] = conf
    return kp


def make_cfg(**over):
    base = {
        "quality": {
            "sample_every": 2,
            "min_box_h": 100,
            "min_det_conf": 0.5,
            "min_quality": 0.3,
            "edge_margin_px": 2,
        },
        "tracklets": {"track_lost_s": 1.0, "min_points": 3, "max_samples_per_tracklet": 3},
        "runtime": {"batch_size": 4},
    }
    for k, v in over.items():
        base.setdefault(k, {}).update(v)
    return Config.model_validate(base)


def make_builder(models=None, is_face=False, start_ts=100.0, **over):
    cam = CameraInfo("c1", "c1", "x.mp4", FPS, W, H, start_ts, is_face)
    return TrackletBuilder(make_cfg(**over), cam, models or FakeModels(), FPS)


def feed(b, frame_idx, rows, kp=None, shade=100):
    """rows: [(tid, box, conf)] -> process one frame; kps indexed by row order."""
    frame = np.full((H, W, 3), shade, np.uint8)
    tracks = [(tid, box, conf, i) for i, (tid, box, conf) in enumerate(rows)]
    kps = None if kp is None else np.stack([kp] * len(rows))
    return b.process(frame_idx, frame, tracks, kps)


BOX = (200.0, 100.0, 300.0, 300.0)  # 200 px tall, away from the edges


def test_helpers():
    assert visibility_ratio(None, 0.3) == 0.0
    kp = full_kps()
    kp[:8, 2] = 0.1
    assert visibility_ratio(kp, 0.3) == pytest.approx(9 / 17)
    assert is_truncated((0, 10, 50, 90), W, H, 2)
    assert is_truncated((10, 10, W - 1, 90), W, H, 2)
    assert not is_truncated((10, 10, 50, 90), W, H, 2)


def test_every_box_is_a_track_point_and_timestamps_use_fps():
    b = make_builder()
    for f in range(0, 20, 2):  # frame_stride 2 in the source video
        feed(b, f, [(1, BOX, 0.2)], kp=full_kps())  # conf below the sampling gate
    (t,) = b.flush()
    assert t.n_points == 10 == len(t.points) and t.n_samples == 0
    assert t.points[3].frame == 6 and t.points[3].ts == pytest.approx(100.0 + 6 / FPS)
    assert t.start_ts == 100.0 and t.end_frame == 18
    assert t.mean_quality == 0.0 and t.body_embedding is None


def test_gates_reject_small_truncated_lowconf_and_invisible():
    b = make_builder(quality={"sample_every": 1})
    cases = {
        "det_conf": ((200, 100, 300, 300), 0.4, full_kps()),
        "box_h": ((200, 100, 250, 150), 0.9, full_kps()),
        "truncated": ((0, 100, 100, 300), 0.9, full_kps()),
        "visibility": (BOX, 0.9, full_kps(0.1)),
        "quality": (BOX, 0.5, np.vstack([full_kps()[:9], full_kps(0.0)[:8]])),  # ratio .53*.5<.3
    }
    for i, (_, (box, conf, kp)) in enumerate(cases.items()):
        feed(b, i, [(i, box, conf)], kp=kp)
    assert dict(b.rejects) == dict.fromkeys(cases, 1)
    assert not b.pending
    assert all(len(tr.points) == 1 for tr in b.tracks.values())  # still tracked


def test_sampling_cadence_and_top_k_retention():
    b = make_builder()  # sample_every=2, K=3
    qual_by_frame = {}
    for f in range(12):
        conf = 0.6 + 0.03 * (f if f < 6 else 12 - f)  # rises then falls
        shade = 10 * f  # crop pixel value identifies the frame
        feed(b, f, [(1, BOX, conf)], kp=full_kps(), shade=shade)
        qual_by_frame[f] = conf * 1.0
    (t,) = b.flush()
    sampled = [p.frame for p in t.points if p.sampled]
    assert len(t.samples) == 3 == len(sampled) == t.n_samples
    assert sorted(sampled) == sorted(s.frame for s in t.samples)
    # all candidate frames were spaced by sample_every (every 2nd processed frame)
    cand = list(range(0, 12, 2))
    assert set(sampled) <= set(cand)
    # kept samples are the 3 best candidates
    best3 = sorted(cand, key=lambda f: -qual_by_frame[f])[:3]
    assert sorted(sampled) == sorted(best3)
    assert t.samples[0].quality >= t.samples[1].quality >= t.samples[2].quality
    # best crop belongs to the highest quality candidate
    assert t.best_crop is not None and int(t.best_crop.mean()) == 10 * best3[0]
    assert t.body_embedding is not None and np.linalg.norm(t.body_embedding) == pytest.approx(1)
    assert t.mean_quality == pytest.approx(np.mean([qual_by_frame[f] for f in best3]))


def test_finalize_on_loss_and_at_end_and_drop_flicker():
    b = make_builder()  # track_lost_s=1.0 -> 10 frames at 10 fps
    out = []
    for f in range(6):
        out += feed(b, f, [(1, BOX, 0.9), (2, (400.0, 100.0, 500.0, 300.0), 0.9)], kp=full_kps())
    assert out == []
    for f in range(6, 40):  # track 2 vanishes at frame 6, track 1 continues
        out += feed(b, f, [(1, BOX, 0.9)], kp=full_kps())
    assert [t.local_track_id for t in out] == [2]  # finalized once lost for >= 1 s
    assert out[0].end_frame == 5 and out[0].n_points == 6
    feed(b, 41, [(3, BOX, 0.9)], kp=full_kps())  # a 1-point flicker, never confirmed
    end = b.flush()
    assert sorted(t.local_track_id for t in end) == [1]  # at the end; flicker (<min_points) dropped
    assert end[0].n_points == 40 and not b.tracks


def test_reappearing_track_within_lost_window_stays_one_tracklet():
    b = make_builder()
    for f in list(range(0, 5)) + list(range(9, 14)):  # 4-frame gap = 0.4 s < track_lost_s
        feed(b, f, [(1, BOX, 0.9)], kp=full_kps())
    ts = b.flush()
    assert len(ts) == 1 and ts[0].n_points == 10


def face_tuple(box, score=0.95, frontal=True):
    x1, y1, x2, y2 = box
    cx, cy, w = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1
    nose_dx = 0.0 if frontal else 0.45 * w * 0.5
    kps = np.array(
        [
            [cx - 0.25 * w, cy - 0.1 * w],
            [cx + 0.25 * w, cy - 0.1 * w],
            [cx + nose_dx, cy + 0.11 * w],
            [cx - 0.2 * w, cy + 0.25 * w],
            [cx + 0.2 * w, cy + 0.25 * w],
        ],
        np.float32,
    )
    emb = np.zeros(512, np.float32)
    emb[int(x1) % 512] = 1.0
    return np.array(box, np.float32), kps, score, emb


def test_best_face_in_head_region_is_selected():
    face = FakeFace()
    b = make_builder(
        FakeModels(face),
        is_face=True,
        face={"max_faces": 2, "head_region": 0.4},
        quality={"sample_every": 1},
    )
    # body box (200,100)-(300,300): head region is y in [100, 180]. Region crop origin = (190, 90).
    inside_small = face_tuple((10, 10, 50, 50))  # frame coords (200,100)-(240,140): small
    inside_big = face_tuple((40, 10, 100, 70))  # (230,100)-(290,160): bigger -> best
    profile_big = face_tuple((40, 10, 100, 70), frontal=False)
    below_head = face_tuple((10, 150, 70, 210))  # centre y ~ 270, outside head region
    face.results = [inside_small, below_head, inside_big]
    feed(b, 0, [(1, BOX, 0.9)], kp=full_kps())
    face.results = [profile_big]
    feed(b, 2, [(1, BOX, 0.9)], kp=full_kps())
    face.results = [below_head]  # only an out-of-region face: ignored
    feed(b, 4, [(1, BOX, 0.9)], kp=full_kps())
    (t,) = b.flush()
    assert len(t.faces) == 2  # max_faces, and the below-head face never appears
    # the best face is the big frontal one, not the first one found (v1 B18)
    assert t.faces[0].box[0] == pytest.approx(230.0)
    assert t.faces[0].quality > t.faces[1].quality
    assert t.face_quality == t.faces[0].quality
    assert t.face_embedding is not None and np.linalg.norm(t.face_embedding) == pytest.approx(1)


def test_faces_only_on_face_cameras():
    face = FakeFace()
    face.results = [face_tuple((10, 10, 50, 50))]
    b = make_builder(FakeModels(face), is_face=False)
    for f in range(0, 6, 2):
        feed(b, f, [(1, BOX, 0.9)], kp=full_kps())
    (t,) = b.flush()
    assert t.faces == [] and t.n_samples == 2  # procs 1 and 3 are due (sample_every=2)
