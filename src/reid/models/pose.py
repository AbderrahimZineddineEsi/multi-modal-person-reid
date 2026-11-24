"""Pose estimator for the two-model presets: pose on padded crops, mapped to their own box."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from reid.config import Config
from reid.models.device import resolve_device, ultralytics_weights, use_half

log = logging.getLogger(__name__)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between ``a`` (N,4) and ``b`` (M,4) xyxy boxes."""
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match_pose_to_box(pose_boxes: np.ndarray, target: np.ndarray, min_iou: float = 0.1) -> int:
    """Index of the pose that belongs to ``target`` (both in the same coordinate frame), or -1.

    The owner is the pose whose box contains the target's centre (the centre of the crop) and
    overlaps the target box best. v1 took ``keypoints[0]`` of the crop, i.e. whichever person the
    model ranked first, which is often a neighbour (review W12).
    """
    pose_boxes = np.asarray(pose_boxes, np.float32).reshape(-1, 4)
    if len(pose_boxes) == 0:
        return -1
    t = np.asarray(target, np.float32).reshape(1, 4)
    cx, cy = (t[0, 0] + t[0, 2]) / 2, (t[0, 1] + t[0, 3]) / 2
    contains = (
        (pose_boxes[:, 0] <= cx)
        & (cx <= pose_boxes[:, 2])
        & (pose_boxes[:, 1] <= cy)
        & (cy <= pose_boxes[:, 3])
    )
    iou = box_iou(pose_boxes, t)[:, 0]
    iou = np.where(contains, iou, -1.0)
    best = int(np.argmax(iou))
    return best if iou[best] >= min_iou else -1


class PoseEstimator:
    """Lazy-loaded Ultralytics pose model run on padded person crops."""

    def __init__(self, cfg: Config) -> None:
        if not cfg.quality.pose_model:
            raise ValueError(
                "quality.pose_model is not set (single-model mode needs no PoseEstimator)"
            )
        self.name = cfg.quality.pose_model
        self.imgsz = cfg.quality.pose_imgsz
        self.pad = cfg.quality.pose_pad
        self.device = resolve_device(cfg.runtime.embed_device)
        self.half = use_half(self.device, cfg.runtime.fp16)
        # ultralytics >= 8.4 replaced the deprecated ``half`` argument by ``quantize``
        self._prec = {"quantize": "fp16"} if self.half else {}
        self.batch = cfg.runtime.batch_size
        self._wdir = cfg.paths.weights_dir
        self._model: Any = None

    def load(self) -> PoseEstimator:
        """Load the weights (idempotent)."""
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(ultralytics_weights(self.name, self._wdir))
            log.info("pose %s on %s", self.name, self.device)
        return self

    def _crop(self, frame: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Padded crop and the target box expressed in crop coordinates."""
        h, w = frame.shape[:2]
        bw, bh = box[2] - box[0], box[3] - box[1]
        x1 = int(max(0, np.floor(box[0] - self.pad * bw)))
        y1 = int(max(0, np.floor(box[1] - self.pad * bh)))
        x2 = int(min(w, np.ceil(box[2] + self.pad * bw)))
        y2 = int(min(h, np.ceil(box[3] + self.pad * bh)))
        local = np.array([box[0] - x1, box[1] - y1, box[2] - x1, box[3] - y1], np.float32)
        return frame[y1:y2, x1:x2], local, np.array([x1, y1], np.float32)  # type: ignore[return-value]

    def keypoints(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Keypoints (N,17,3) in frame pixels, one pose per input box (zeros if none matched)."""
        self.load()
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        out = np.zeros((len(boxes), 17, 3), np.float32)
        items = []
        for i, b in enumerate(boxes):
            crop, local, off = self._crop(frame, b)
            if crop.size:
                items.append((i, crop, local, off))
        for s in range(0, len(items), self.batch):
            chunk = items[s : s + self.batch]
            results = self._model.predict(
                [c for _, c, _, _ in chunk],
                imgsz=self.imgsz,
                device=self.device,
                **self._prec,
                conf=0.1,
                verbose=False,
            )
            for (i, _, local, off), res in zip(chunk, results, strict=True):
                if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
                    continue
                j = match_pose_to_box(res.boxes.xyxy.cpu().numpy(), local)
                if j >= 0:
                    kp = res.keypoints.data[j].cpu().numpy().astype(np.float32)
                    kp[:, :2] += off
                    out[i] = kp
        return out
