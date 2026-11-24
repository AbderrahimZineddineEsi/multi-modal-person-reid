"""Per-camera multi-object tracker wrapper around boxmot 26 (motion-only)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from reid.config import TrackingCfg

log = logging.getLogger(__name__)

TrackRow = tuple[int, tuple[float, float, float, float], float, int]  # id, box, conf, det_index


def tracker_options(cfg: TrackingCfg) -> dict[str, Any]:
    """Map our tracker config onto the option names of the selected boxmot tracker.

    Appearance is always disabled: identity across occlusions is handled at the tracklet level
    (docs/dev/PLAN.md section 3.2), so no ReID model is loaded by the tracker (fixes v1 W1).
    """
    if cfg.tracker == "bytetrack":
        return {
            "track_thresh": cfg.track_high_thresh,
            "min_conf": cfg.track_low_thresh,
            "match_thresh": cfg.match_thresh,
            "max_lost_frames": cfg.track_buffer,
        }
    if cfg.tracker == "botsort":
        return {
            "track_high_thresh": cfg.track_high_thresh,
            "track_low_thresh": cfg.track_low_thresh,
            "new_track_thresh": cfg.new_track_thresh,
            "match_thresh": cfg.match_thresh,
            "max_lost_frames": cfg.track_buffer,
            "use_embeddings": False,
            "use_cmc": cfg.use_cmc,
        }
    # occluboost ships tuned defaults of its own; only shared knobs are forwarded.
    return {
        "track_low_thresh": cfg.track_low_thresh,
        "new_track_thresh": cfg.new_track_thresh,
        "use_embeddings": False,
    }


class Tracker:
    """Swappable tracker: ``update(boxes, confs, frame) -> [(track_id, box, conf, det_index)]``.

    ``det_index`` is the row of the input detections that produced the track at this frame
    (-1 when the track was only propagated by the motion model), so keypoints computed per
    detection can be matched to tracks.
    """

    def __init__(self, cfg: TrackingCfg) -> None:
        self.cfg = cfg
        self._tracker: Any = None

    def _build(self) -> Any:
        from boxmot.trackers import create_tracker

        opts = tracker_options(self.cfg)
        log.info("tracker %s (motion-only) %s", self.cfg.tracker, opts)
        return create_tracker(self.cfg.tracker, **opts)

    def reset(self) -> None:
        """Forget all tracks (call between cameras)."""
        self._tracker = None

    def update(self, boxes: np.ndarray, confs: np.ndarray, frame: np.ndarray) -> list[TrackRow]:
        """Advance one frame. Must be called every processed frame, even without detections."""
        if self._tracker is None:
            self._tracker = self._build()
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        dets = np.concatenate(
            [
                boxes,
                np.asarray(confs, np.float32).reshape(-1, 1),
                np.zeros((len(boxes), 1), np.float32),
            ],
            axis=1,
        )
        out = np.asarray(self._tracker.update(dets, frame))
        rows: list[TrackRow] = []
        if out.size == 0:
            return rows
        for r in out.reshape(-1, out.shape[-1]):
            det_ind = int(r[7]) if out.shape[-1] > 7 else -1
            rows.append(
                (
                    int(r[4]),
                    (float(r[0]), float(r[1]), float(r[2]), float(r[3])),
                    float(r[5]),
                    det_ind,
                )
            )
        return rows
