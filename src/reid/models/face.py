"""Face detection + embedding (InsightFace SCRFD + ArcFace on onnxruntime) and face quality."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from reid.config import Config
from reid.models.device import onnx_providers, resolve_device, weights_dir
from reid.models.placement import Placeable

log = logging.getLogger(__name__)

FaceResult = tuple[np.ndarray, np.ndarray, float, np.ndarray]  # box(4), kps5(5x2), score, emb512


def size_factor(box: np.ndarray, min_px: int, full_px: int) -> float:
    """0 below ``min_px`` (shorter box side), rising linearly to 1 at ``full_px``."""
    side = float(min(box[2] - box[0], box[3] - box[1]))
    if full_px <= min_px:
        return float(side >= full_px)
    return float(np.clip((side - min_px) / (full_px - min_px), 0.0, 1.0))


def frontalness(kps: np.ndarray) -> float:
    """Frontal-pose score in [0, 1] from the 5 InsightFace landmarks (eyes, nose, mouth corners).

    Yaw: horizontal offset of the nose from the eye midpoint relative to the inter-ocular
    distance (0 for a frontal face, about 0.5 or more in profile). Pitch: where the nose lies
    between the eye line and the mouth line (about 0.5-0.65 when facing the camera).
    """
    kps = np.asarray(kps, np.float32).reshape(5, 2)
    le, re, nose, lm, rm = kps
    eye_dist = float(np.linalg.norm(re - le))
    if eye_dist < 1e-3:
        return 0.0
    yaw_ratio = abs(float(nose[0] - (le[0] + re[0]) / 2)) / eye_dist
    yaw_f = float(np.clip(1.0 - yaw_ratio / 0.5, 0.0, 1.0))
    eye_y, mouth_y = (le[1] + re[1]) / 2, (lm[1] + rm[1]) / 2
    if mouth_y - eye_y < 1e-3:
        return 0.0
    t = float((nose[1] - eye_y) / (mouth_y - eye_y))
    pitch_f = float(np.clip(1.0 - abs(t - 0.6) / 0.5, 0.0, 1.0))
    return yaw_f * (0.5 + 0.5 * pitch_f)


def face_quality(
    det_score: float, box: np.ndarray, kps: np.ndarray, min_px: int, full_px: int
) -> float:
    """Face quality = detection score x size factor x frontalness (PLAN section 3.1 step 5)."""
    return float(det_score) * size_factor(box, min_px, full_px) * frontalness(kps)


class FaceModel(Placeable):
    """Lazy-loaded InsightFace pack (``buffalo_l``, ``buffalo_sc``, ``antelopev2``)."""

    role = "face"

    def __init__(self, cfg: Config) -> None:
        self.name = cfg.face.model
        self.det_size = cfg.face.det_size
        self.pad = cfg.search.face_pad
        self.device = resolve_device(cfg.runtime.embed_device)
        self.root = str(weights_dir(cfg.paths.weights_dir) / "insightface")
        self._app: Any = None
        self.provider = ""

    def _unload(self) -> None:
        self._app = None

    def load(self) -> FaceModel:
        """Download (first use) and load the model pack (idempotent)."""
        if self._app is None:
            providers, ctx_id = onnx_providers(self.device)
            try:
                self._build(providers, ctx_id)
            except Exception as exc:  # e.g. onnxruntime-gpu built without this GPU's arch (Pascal)
                if ctx_id < 0:
                    raise
                log.warning("face model failed on CUDA (%s); falling back to CPU", str(exc)[:120])
                self._build(["CPUExecutionProvider"], -1)
        return self

    def _build(self, providers: list, ctx_id: int) -> None:
        from insightface.app import FaceAnalysis
        from insightface.utils.storage import ensure_available

        flatten_model_pack(Path(ensure_available("models", self.name, root=self.root)))
        self._app = FaceAnalysis(
            name=self.name,
            root=self.root,
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        # det_thresh 0.5 is the library default used by v1; callers filter on face.det_conf.
        self._app.prepare(ctx_id=ctx_id, det_size=(self.det_size, self.det_size), det_thresh=0.5)
        self.provider = providers[0] if isinstance(providers[0], str) else providers[0][0]
        log.info("face %s on %s", self.name, self.provider)

    def analyze(self, image: np.ndarray) -> list[FaceResult]:
        """Detect faces in a BGR image; returns ``(box, kps5, det_score, embedding512)`` each.

        Embeddings are L2-normalized (ArcFace ``normed_embedding``).
        """
        return self.guarded(lambda: self._analyze(image))

    def _analyze(self, image: np.ndarray) -> list[FaceResult]:
        self.load()
        out: list[FaceResult] = []
        for f in self._app.get(image):
            if f.normed_embedding is None or f.kps is None:
                continue
            out.append(
                (
                    np.asarray(f.bbox, np.float32),
                    np.asarray(f.kps, np.float32),
                    float(f.det_score),
                    np.asarray(f.normed_embedding, np.float32),
                )
            )
        return out

    def embed_aligned(self, face_chip: np.ndarray) -> np.ndarray | None:
        """Embedding of a pre-cropped face (query images), or None if no face is found.

        The chip is centred on a mid-gray canvas padded by ``search.face_pad`` x max(h, w) on
        each side so the detector sees context (v1 review section 1.14); the highest-scoring
        detection is embedded. Note: for a square chip v1's padding makes a 3x canvas.
        """
        h, w = face_chip.shape[:2]
        if h == 0 or w == 0:
            return None
        p = int(round(self.pad * max(h, w)))
        canvas = np.full((h + 2 * p, w + 2 * p, 3), 128, np.uint8)
        canvas[p : p + h, p : p + w] = face_chip
        res = self.analyze(canvas)
        return max(res, key=lambda r: r[2])[3] if res else None


def flatten_model_pack(pack: Path) -> None:
    """Move ``<pack>/<name>/*.onnx`` up to ``<pack>`` (the antelopev2 zip nests its folder).

    InsightFace only looks for ``*.onnx`` directly in ``<pack>`` and otherwise fails with a bare
    ``assert "detection" in self.models``.
    """
    inner = pack / pack.name
    if not inner.is_dir() or any(pack.glob("*.onnx")):
        return
    for f in inner.iterdir():
        shutil.move(str(f), str(pack / f.name))
    inner.rmdir()
    log.info("flattened nested model pack %s", pack)
