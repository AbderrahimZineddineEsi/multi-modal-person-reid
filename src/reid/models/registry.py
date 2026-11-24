"""Model registry: ``build_models(cfg)`` returns a :class:`Models` holder (all lazy)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from reid.config import Config

if TYPE_CHECKING:  # keep ``import reid.models.registry`` free of torch (CPU-only test runs)
    from reid.models.body_reid import BodyReID
    from reid.models.detector import PersonDetector
    from reid.models.face import FaceModel
    from reid.models.pose import PoseEstimator

log = logging.getLogger(__name__)


@dataclass
class Models:
    """The four model slots. ``pose`` is None in single-model mode (the detector is a pose
    model) and ``face`` is None when ``face.enabled`` is false. Weights load on first use."""

    detector: PersonDetector
    pose: PoseEstimator | None
    body: BodyReID
    face: FaceModel | None
    vram_peak_mb: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Record the peak ``torch.cuda.max_memory_allocated`` reached inside the block.

        The figure includes weights of every model already resident on the GPU (it is the peak
        of the whole process while the stage ran). ONNX Runtime allocations (face models) are not
        visible to torch and are not counted.
        """
        try:
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]
        if torch is None or not torch.cuda.is_available():
            yield
            return
        torch.cuda.reset_peak_memory_stats()
        try:
            yield
        finally:
            mb = torch.cuda.max_memory_allocated() / 2**20
            self.vram_peak_mb[name] = max(self.vram_peak_mb.get(name, 0.0), mb)

    def vram_report(self) -> dict[str, float]:
        """Per-stage peak VRAM in MiB plus ``total_peak`` (empty without CUDA)."""
        out = dict(self.vram_peak_mb)
        if out:
            out["total_peak"] = max(out.values())
        return out

    def warm(self) -> None:
        """Load every configured model now (otherwise they load on first use)."""
        self.detector.load()
        if self.pose:
            self.pose.load()
        self.body.load()
        if self.face:
            self.face.load()


def build_models(cfg: Config) -> Models:
    """Create the model holder from config. No weights are downloaded or loaded yet."""
    from reid.models.body_reid import BodyReID
    from reid.models.detector import PersonDetector
    from reid.models.face import FaceModel
    from reid.models.pose import PoseEstimator

    pose = PoseEstimator(cfg) if cfg.quality.pose_model else None
    det = PersonDetector(cfg)
    if pose is None and not det.is_pose:
        log.warning(
            "detection.model %r has no keypoints and quality.pose_model is unset: the visibility "
            "gate is disabled (quality = detection confidence)",
            det.name,
        )
    return Models(
        detector=det,
        pose=pose,
        body=BodyReID(cfg),
        face=FaceModel(cfg) if cfg.face.enabled else None,
    )
