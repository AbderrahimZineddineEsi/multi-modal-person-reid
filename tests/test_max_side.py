"""Processing-resolution cap (runtime.max_side), frame-size-aware drawing and GT scaling."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from reid.config import load_config
from reid.eval import scale_gt_to_cameras
from reid.pipeline import read_frames
from reid.types import CameraInfo
from reid.viz import draw
from reid.viz.video_io import FrameSource, VideoSink, fit_frame, processed_size


def test_processed_size():
    assert processed_size(3840, 2160, 1920) == (1920, 1080)
    assert processed_size(2160, 3840, 1920) == (1080, 1920)
    assert processed_size(1280, 720, 1920) == (1280, 720)
    assert processed_size(3840, 2160, 0) == (3840, 2160)
    w, h = processed_size(3001, 1999, 1000)
    assert max(w, h) == 1000 and w % 2 == 0 and h % 2 == 0


def test_fit_frame():
    big = np.full((216, 384, 3), 100, np.uint8)
    out = fit_frame(big, 192)
    assert out.shape[:2] == (108, 192)
    assert fit_frame(big, 0) is big and fit_frame(big, 1000) is big


def _video(path, w=384, h=216, n=6):
    with VideoSink(path, w, h, 10) as s:
        for i in range(n):
            s.write(np.full((h, w, 3), 20 * i, np.uint8))
    return path


def test_read_frames_and_framesource_resize(tmp_path):
    v = _video(tmp_path / "a.mp4")
    cap = cv2.VideoCapture(str(v))
    got = list(read_frames(cap, 0, None, 2, None, 192))
    cap.release()
    assert [i for i, _ in got] == [0, 2, 4] and got[0][1].shape[:2] == (108, 192)
    with FrameSource(v, (192, 108)) as src:
        assert (src.width, src.height) == (192, 108)
        assert src.get(1).shape[:2] == (108, 192)
    with FrameSource(v, (384, 216)) as src:  # equal size: untouched
        assert src.get(1).shape[:2] == (216, 384)
    with FrameSource(v) as src:
        assert src.get(1).shape[:2] == (216, 384)


def test_draw_scales_with_frame_height():
    assert draw.label_font_px(2160) > draw.label_font_px(720) > draw.label_font_px(240)
    assert draw.line_thickness(2160, 2) > draw.line_thickness(720, 2) == 2.0
    chips = []
    for H in (360, 1080):
        W = H * 16 // 9
        f = np.zeros((H, W, 3), np.uint8)
        d = draw.Det(1, 3, (W - 40.0, 2.0, W - 5.0, H / 2), np.zeros((0, 2)))
        draw.draw_dets(f, [d], draw.Style(trail_len=0))
        chips.append(draw.chip_sprite("ID 3", (200, 50, 50), draw.label_font_px(H)).h)
        assert f.any()  # label near the right/top edge is clamped inside, not dropped
    assert chips[1] > 2 * chips[0]


def test_scale_gt_to_camera_size(tmp_path):
    v = _video(tmp_path / "a.mp4", 384, 216)
    gt = pd.DataFrame(
        {"camera": ["c", "c", "o"], "frame": [0, 1, 0], "gid": [1, 1, 2],
         "x1": [100.0, 0, 10], "y1": [50.0, 0, 10], "x2": [200.0, 384, 20], "y2": [150.0, 216, 20]}
    )  # fmt: skip

    def mk(cid, w, h):
        return CameraInfo(id=cid, name=cid, video_path=str(v), fps=10, width=w, height=h)

    out = scale_gt_to_cameras(gt, {"c": mk("c", 192, 108), "o": mk("o", 384, 216)}, tmp_path)
    assert out.loc[0, ["x1", "y1", "x2", "y2"]].tolist() == [50.0, 25.0, 100.0, 75.0]
    assert out.loc[1, "x2"] == 192 and out.loc[1, "y2"] == 108
    assert out.loc[2, ["x1", "x2"]].tolist() == [10.0, 20.0]  # same size: unchanged


def test_config_default():
    assert load_config().runtime.max_side == 1920
