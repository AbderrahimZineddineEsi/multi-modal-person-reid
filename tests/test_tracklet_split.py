"""Appearance-change splitting of tracklets, appearance-aware tracker options, tracklet purity."""

import numpy as np
import pandas as pd
import pytest

from reid.config import TrackingCfg
from reid.tracking.tracker import Tracker, appearance_active, tracker_options
from reid.tracking.tracklets import TrackletBuilder, choose_split_frame, split_indices
from reid.types import CameraInfo
from tests.test_tracklets import FPS, FakeModels, H, W, full_kps, make_cfg

RNG = np.random.default_rng(0)


def unit(v):
    v = np.asarray(v, np.float64)
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def person(base, n, noise=0.05):
    return unit(base + noise * RNG.normal(size=(n, len(base))))


A, B, C = np.eye(16)[0], np.eye(16)[1], np.eye(16)[2]


def test_split_indices_one_switch():
    e = np.vstack([person(A, 10), person(B, 6)])
    assert split_indices(e, 0.7, 3) == [10]


def test_split_indices_same_person_with_noise():
    assert split_indices(person(A, 16, noise=0.08), 0.7, 3) == []


def test_split_indices_two_switches():
    e = np.vstack([person(A, 6), person(B, 7), person(C, 5)])
    assert split_indices(e, 0.7, 3) == [6, 13]


def test_split_indices_too_few_samples():
    e = np.vstack([person(A, 3), person(B, 2)])  # 5 < 2 * 3
    assert split_indices(e, 0.7, 3) == []
    e = np.vstack([person(A, 5), person(B, 2)])  # B side below the minimum: cut at 5 is refused
    assert 5 not in split_indices(e, 0.7, 3)


def test_choose_split_frame_prefers_max_overlap_then_middle():
    frames = list(range(10, 20))
    ov = [0, 0, 0, 0.1, 0.6, 0.3, 0, 0, 0, 0]
    assert choose_split_frame(frames, ov, 11, 18) == 14  # the crossing frame
    assert choose_split_frame(frames, [0.0] * 10, 10, 18) == 15  # none: middle of (10, 18]
    assert choose_split_frame(frames, ov, 30, 40) == 40  # no points in range


# ---------------------------------------------------------------- builder level
RED, BLUE = (0, 0, 255), (255, 0, 0)


class ColourBody:
    """Embedding determined by the crop's colour (red person vs blue person)."""

    def embed(self, crops):
        out = []
        for c in crops:
            m = c.reshape(-1, 3).mean(0)
            out.append([m[2] + 1.0, m[0] + 1.0, 1.0])
        return unit(np.array(out))


class ColourModels(FakeModels):
    def __init__(self):
        super().__init__()
        self.body = ColourBody()


def run_swap(split=True, other_box=(240.0, 100.0, 340.0, 300.0), switch=20, n=40):
    """One track whose appearance changes red -> blue at frame ``switch``; a second (static)
    track overlaps it for frames 18-22 only, so that is where the cut should be placed."""
    cam = CameraInfo("c1", "c1", "x.mp4", FPS, W, H, 100.0, False)
    cfg = make_cfg(
        quality={"sample_every": 1},
        tracklets={"max_samples_per_tracklet": 4, "split_on_appearance": split},
    )
    b = TrackletBuilder(cfg, cam, ColourModels(), FPS)
    out = []
    box = (200.0, 100.0, 300.0, 300.0)
    for f in range(n):
        frame = np.zeros((H, W, 3), np.uint8)
        frame[:] = RED if f < switch else BLUE
        rows = [(1, box, 0.9, 0)]
        if 18 <= f <= 22:
            rows.append((2, other_box, 0.9, 1))
        out += b.process(f, frame, rows, np.stack([full_kps()] * len(rows)))
    out += b.flush()
    return b, [t for t in out if t.local_track_id == 1]


def test_builder_splits_into_pieces_at_the_crossing():
    b, pieces = run_swap()
    assert b.n_splits == 1 and len(pieces) == 2
    a, c = sorted(pieces, key=lambda t: t.start_frame)
    assert a.start_frame == 0 and c.end_frame == 39 and a.end_frame + 1 == c.start_frame
    assert 18 <= c.start_frame <= 22  # inside the overlap window, middle of equal maxima
    assert a.n_points + c.n_points == 40
    for t, col in ((a, RED), (c, BLUE)):
        assert t.n_samples == len(t.samples) == 4
        assert all(s.frame < c.start_frame for s in a.samples)
        assert t.best_crop is not None and t.best_crop.shape[-1] == 3
        assert tuple(int(v) for v in t.best_crop.reshape(-1, 3).mean(0).round()) == pytest.approx(
            col, abs=8
        )
        assert t.body_embedding is not None and np.linalg.norm(t.body_embedding) == pytest.approx(1)
        assert {p.frame for p in t.points if p.sampled} == {s.frame for s in t.samples}
    assert all(s.frame >= c.start_frame for s in c.samples)
    assert float(a.body_embedding @ c.body_embedding) < 0.7


def test_builder_split_disabled_keeps_one_blended_tracklet():
    b, pieces = run_swap(split=False)
    assert b.n_splits == 0 and len(pieces) == 1 and pieces[0].n_points == 40


def test_builder_partitions_faces_by_frame():
    b, _ = run_swap()  # reuse the setup, then check faces on a fresh builder
    from reid.types import FaceObs

    cam = CameraInfo("c1", "c1", "x.mp4", FPS, W, H, 100.0, False)
    cfg = make_cfg(quality={"sample_every": 1}, tracklets={"max_samples_per_tracklet": 4})
    b = TrackletBuilder(cfg, cam, ColourModels(), FPS)
    box = (200.0, 100.0, 300.0, 300.0)
    for f in range(30):
        frame = np.zeros((H, W, 3), np.uint8)
        frame[:] = RED if f < 15 else BLUE
        b.process(f, frame, [(1, box, 0.9, 0)], np.stack([full_kps()]))
    b.tracks[1].faces = [
        FaceObs(f, (0, 0, 1, 1), 0.9, 0.5, np.eye(4, dtype=np.float32)[0], None) for f in (3, 25)
    ]
    pieces = b.flush()
    assert len(pieces) == 2
    assert [len(t.faces) for t in sorted(pieces, key=lambda t: t.start_frame)] == [1, 1]
    assert all(t.face_embedding is not None for t in pieces)


# ---------------------------------------------------------------- tracker options
def test_tracker_options_appearance_only_for_supporting_trackers():
    on = TrackingCfg(tracker="botsort", appearance=True, appearance_thresh=0.3)
    assert appearance_active(on)
    o = tracker_options(on)
    assert o["use_embeddings"] is True and o["appearance_thresh"] == 0.3
    assert tracker_options(TrackingCfg(tracker="botsort"))["use_embeddings"] is False
    assert tracker_options(TrackingCfg(tracker="occluboost", appearance=True))["use_embeddings"]
    bt = TrackingCfg(tracker="bytetrack", appearance=True)
    assert not appearance_active(bt) and "use_embeddings" not in tracker_options(bt)


def test_tracker_update_forwards_embeddings(caplog):
    import torch
    from boxmot.structures import Boxes, Tracks

    seen = {}

    class Fake:
        def update(self, dets, frame):
            seen["dets"] = dets
            n = len(dets)
            return Tracks(
                geometry=Boxes(dets.geometry.values.clone()),
                track_ids=torch.arange(1, n + 1),
                scores=dets.scores.clone(),
                class_ids=torch.zeros(n, dtype=torch.int64),
                detection_indices=torch.arange(n),
                sample_id="x",
            )

    t = Tracker(TrackingCfg(tracker="botsort", appearance=True))
    t._tracker = Fake()
    boxes = np.array([[10, 10, 50, 90], [60, 10, 100, 90]], np.float32)
    embs = unit(np.random.default_rng(1).normal(size=(2, 8)))
    rows = t.update(boxes, np.array([0.9, 0.8]), np.zeros((100, 120, 3), np.uint8), embs)
    assert [r[0] for r in rows] == [1, 2] and [r[3] for r in rows] == [0, 1]
    got = seen["dets"].embeddings.numpy()
    assert np.allclose(got, embs)
    assert (
        t.update(np.zeros((0, 4)), np.zeros(0), np.zeros((100, 120, 3), np.uint8), np.zeros((0, 8)))
        == []
    )
    # motion-only: embeddings are ignored and boxmot's NumPy path is used
    t2 = Tracker(TrackingCfg(tracker="bytetrack", appearance=True))
    assert not t2.use_appearance


# ---------------------------------------------------------------- eval metric
def test_tracklet_purity():
    pytest.importorskip("motmetrics")
    from reid.eval.mot import tracklet_purity

    pairs = pd.DataFrame(
        {"oid": [1, 1, 2, 3, 4], "hid": [10, 11, 11, 12, 12], "n": [30, 20, 12, 40, 2]}
    )
    hyp = pd.DataFrame({"tid": [10, 11, 12], "camera": ["c1"] * 3, "gid": [1, 2, None]})
    r = tracklet_purity(pairs, hyp, min_boxes=5)
    assert r["tracklets"] == 3 and r["mixed"] == 1  # tracklet 11 has GT 1 (20) and GT 2 (12)
    assert [e["tracklet"] for e in r["examples"]] == [11]
    assert r["examples"][0]["gt_boxes"] == {"1": 20, "2": 12} and r["examples"][0]["gid"] == 2
    assert r["purity"] == pytest.approx((30 + 20 + 40) / (30 + 20 + 12 + 40 + 2))
    assert tracklet_purity(pairs.iloc[0:0], hyp)["mixed"] == 0


def test_history_is_bounded_and_keeps_best_sample():
    from reid.tracking.tracklets import _Track, thin_history
    from reid.types import Sample

    tr = _Track(1, 0, 0)
    for f in range(100):
        tr.history.append(Sample(f, 0.99 if f == 7 else 0.5, np.zeros(4, np.float32)))
        tr.jpegs.append(b"")
    thin_history(tr)
    frames = [s.frame for s in tr.history]
    assert len(frames) < 100 and len(tr.jpegs) == len(frames)
    assert 7 in frames  # the best-quality sample survives
    assert frames[-50:] == list(range(50, 100))  # the recent half stays dense
