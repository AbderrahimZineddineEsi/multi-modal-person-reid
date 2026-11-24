"""Person detector: Ultralytics (YOLO26, YOLO26-pose, RT-DETR) or RF-DETR backends."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from reid.config import Config
from reid.models.device import resolve_device, ultralytics_weights, use_half
from reid.models.placement import Placeable

log = logging.getLogger(__name__)

#: Person class id in RF-DETR predictions (COCO, 1-indexed).
RFDETR_PERSON_ID = 1

Detections = tuple[np.ndarray, np.ndarray, "np.ndarray | None"]


def _empty(with_kps: bool) -> Detections:
    return (
        np.zeros((0, 4), np.float32),
        np.zeros((0,), np.float32),
        np.zeros((0, 17, 3), np.float32) if with_kps else None,
    )


class PersonDetector(Placeable):
    """Lazy-loaded person detector.

    ``detect(frame)`` takes a BGR frame and returns ``(boxes Nx4 xyxy in frame pixels,
    confs N, keypoints Nx17x3 or None)``. Keypoints (x, y, conf) are only returned when the
    configured model is a pose model (single-model mode); each box then carries its own pose
    from the same forward pass.
    """

    role = "detector"

    def __init__(self, cfg: Config) -> None:
        self.name = cfg.detection.model
        self.conf = cfg.detection.conf
        self.imgsz = cfg.detection.imgsz
        self.classes = list(cfg.detection.classes)
        self.device = resolve_device(cfg.runtime.detect_device)
        self._wdir = cfg.paths.weights_dir
        self._want_half = cfg.runtime.fp16
        self.half = use_half(self.device, self._want_half)
        self._on_device_change()
        self.backend = "rfdetr" if self.name.lower().startswith("rfdetr") else "ultralytics"
        self.is_pose = self.backend == "ultralytics" and "pose" in self.name.lower()
        self._model: Any = None

    def _on_device_change(self) -> None:
        # ultralytics >= 8.4 replaced the deprecated ``half`` argument by ``quantize``
        self._prec = {"quantize": "fp16"} if self.half else {}

    def _unload(self) -> None:
        self._model = None

    # ------------------------------------------------------------------ loading
    def load(self) -> PersonDetector:
        """Load the weights (idempotent)."""
        if self._model is not None:
            return self
        if self.backend == "rfdetr":
            self._load_rfdetr()
        else:
            self._load_ultralytics()
        log.info("detector %s (%s) on %s", self.name, self.backend, self.device)
        return self

    def _load_ultralytics(self) -> None:
        if "rtdetr" in self.name.lower():
            from ultralytics import RTDETR

            self._model = RTDETR(ultralytics_weights(self.name, self._wdir))
        else:
            from ultralytics import YOLO

            self._model = YOLO(ultralytics_weights(self.name, self._wdir))

    def _load_rfdetr(self) -> None:
        import rfdetr

        _logs_to_stderr("rf-detr")  # it logs to stdout, which corrupts `reid process` JSON

        sizes = {
            "small": "RFDETRSmall",
            "medium": "RFDETRMedium",
            "large": "RFDETRLarge",
            "nano": "RFDETRNano",
            "base": "RFDETRBase",
        }
        key = self.name.lower().replace("rfdetr", "").strip("-_ .pt") or "medium"
        if key not in sizes:
            raise ValueError(f"unknown RF-DETR size {key!r}; choose from {sorted(sizes)}")
        cls = getattr(rfdetr, sizes[key])
        self._model = cls(device=self.device)

    # ------------------------------------------------------------------ inference
    def detect(self, frame: np.ndarray) -> Detections:
        """Detect people in a BGR frame (falls back to CPU if the GPU runs out of memory)."""
        return self.guarded(lambda: self._detect(frame))

    def _detect(self, frame: np.ndarray) -> Detections:
        self.load()
        if self.backend == "rfdetr":
            return self._detect_rfdetr(frame)
        return self._detect_ultralytics(frame)

    def _detect_ultralytics(self, frame: np.ndarray) -> Detections:
        res = self._model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=self.classes,
            device=self.device,
            **self._prec,
            verbose=False,
        )[0]
        b = res.boxes
        if b is None or len(b) == 0:
            return _empty(self.is_pose)
        boxes = b.xyxy.cpu().numpy().astype(np.float32)
        confs = b.conf.cpu().numpy().astype(np.float32)
        kps = None
        if self.is_pose:
            if res.keypoints is None:
                return _empty(True)
            kps = res.keypoints.data.cpu().numpy().astype(np.float32)
        return boxes, confs, kps

    def _detect_rfdetr(self, frame: np.ndarray) -> Detections:
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        det = self._model.predict(rgb, threshold=self.conf)
        if det is None or len(det) == 0:
            return _empty(False)
        # RF-DETR predictions carry 1-indexed COCO ids (person = 1, verified on rfdetr 1.12.1)
        # while ``model.class_names`` is a 0-indexed list; select by the per-detection class
        # *name* when present (robust to either convention), else by id 1.
        names = getattr(det, "data", {}).get("class_name")
        if names is not None:
            keep = np.asarray(names) == "person"
        else:
            keep = np.asarray(det.class_id) == RFDETR_PERSON_ID
        return (
            np.asarray(det.xyxy, np.float32)[keep].reshape(-1, 4),
            np.asarray(det.confidence, np.float32)[keep],
            None,
        )


def _logs_to_stderr(name: str) -> None:
    """Point the stdout handlers of logger ``name`` at stderr (stdout carries our JSON)."""
    import logging
    import sys

    for h in logging.getLogger(name).handlers:
        if getattr(h, "_stream_name", None) == "stdout":  # rfdetr's redirect-aware handler
            h._stream_name = "stderr"  # type: ignore[attr-defined]
        elif isinstance(h, logging.StreamHandler) and h.stream is sys.stdout:
            h.setStream(sys.stderr)
