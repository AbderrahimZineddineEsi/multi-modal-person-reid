"""Per-camera multi-object tracker wrapper around boxmot 26 (motion-only)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from reid.config import TrackingCfg

log = logging.getLogger(__name__)

TrackRow = tuple[int, tuple[float, float, float, float], float, int]  # id, box, conf, det_index


def appearance_active(cfg: TrackingCfg) -> bool:
    """True if the tracker will use appearance embeddings (``cfg.appearance`` and supported)."""
    return cfg.appearance and cfg.tracker != "bytetrack"


def tracker_options(cfg: TrackingCfg) -> dict[str, Any]:
    """Map our tracker config onto the option names of the selected boxmot tracker.

    Appearance is off by default: identity across occlusions is handled at the tracklet level
    (docs/dev/PLAN.md section 3.2), so no ReID model is loaded by the tracker (fixes v1 W1).
    With ``cfg.appearance`` the pipeline supplies the embeddings itself (``Tracker.update(...,
    embs=...)``); boxmot only generates embeddings when none are supplied, so it never builds
    its own ReID model.
    """
    use_emb = appearance_active(cfg)
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
            "use_embeddings": use_emb,
            "use_cmc": cfg.use_cmc,
            **({"appearance_thresh": cfg.appearance_thresh} if use_emb else {}),
        }
    # occluboost ships tuned defaults of its own; only shared knobs are forwarded.
    return {
        "track_low_thresh": cfg.track_low_thresh,
        "new_track_thresh": cfg.new_track_thresh,
        "use_embeddings": use_emb,
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
        self.use_appearance = appearance_active(cfg)
        if cfg.appearance and not self.use_appearance:
            log.warning("tracking.appearance ignored: %s has no appearance matching", cfg.tracker)

    def _build(self) -> Any:
        from boxmot.trackers import create_tracker

        opts = tracker_options(self.cfg)
        log.info(
            "tracker %s (%s) %s",
            self.cfg.tracker,
            "appearance" if self.use_appearance else "motion-only",
            opts,
        )
        return create_tracker(self.cfg.tracker, **opts)

    def reset(self) -> None:
        """Forget all tracks (call between cameras)."""
        self._tracker = None

    def update(
        self,
        boxes: np.ndarray,
        confs: np.ndarray,
        frame: np.ndarray,
        embs: np.ndarray | None = None,
    ) -> list[TrackRow]:
        """Advance one frame. Must be called every processed frame, even without detections.

        ``embs`` (one row per detection) are used for appearance matching when appearance is
        active; otherwise they are ignored.
        """
        if self._tracker is None:
            self._tracker = self._build()
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        if self.use_appearance and embs is not None:
            return self._update_with_embs(boxes, confs, frame, embs)
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

    def _update_with_embs(
        self, boxes: np.ndarray, confs: np.ndarray, frame: np.ndarray, embs: np.ndarray
    ) -> list[TrackRow]:
        """boxmot's NumPy path cannot carry embeddings, so use its canonical ``Detections``."""
        import torch
        from boxmot.structures import Boxes, Detections

        n = len(boxes)
        embs = np.ascontiguousarray(embs, np.float32).reshape(n, np.shape(embs)[-1])
        b = boxes.copy()
        b[:, 2:] = np.maximum(b[:, 2:], b[:, :2] + 1e-3)  # Boxes requires x2 > x1, y2 > y1
        dets = Detections(
            geometry=Boxes(torch.from_numpy(b)),
            scores=torch.from_numpy(np.clip(np.asarray(confs, np.float32).reshape(-1), 0.0, 1.0)),
            class_ids=torch.zeros(n, dtype=torch.int64),
            embeddings=torch.from_numpy(embs),
        )
        out = self._tracker.update(dets, frame)
        xyxy = out.geometry.values.numpy()
        ids, scores, det_ix = (
            out.track_ids.numpy(),
            out.scores.numpy(),
            out.detection_indices.numpy(),
        )
        return [
            (int(i), tuple(float(v) for v in bx), float(sc), int(d))  # type: ignore[misc]
            for i, bx, sc, d in zip(ids, xyxy, scores, det_ix, strict=True)
        ]
