"""Model registry: ``build_models(cfg)`` returns a :class:`Models` holder (all lazy)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from reid.config import Config
from reid.models.planner import (
    DevicePlan,
    Hardware,
    apply_cpu_threads,
    detect_hardware,
    plan_devices,
)

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
    plan: DevicePlan | None = None  # where each model was placed (see models/planner.py)

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

    def slots(self) -> dict[str, Any]:
        """role -> wrapper for the configured models."""
        out = {"detector": self.detector, "body": self.body, "pose": self.pose, "face": self.face}
        return {r: m for r, m in out.items() if m is not None}

    def devices(self) -> dict[str, str]:
        """role -> current device (differs from the plan after a runtime OOM fallback)."""
        return {r: str(getattr(m, "device", "?")) for r, m in self.slots().items()}

    def warm(self) -> None:
        """Load every configured model now (otherwise they load on first use).

        A model whose weights do not fit on the GPU falls back to CPU instead of failing.
        """
        for m in self.slots().values():
            guard = getattr(m, "guarded", None)
            guard(m.load) if guard else m.load()

    def device_plan(self) -> dict[str, Any]:
        """The plan as a dict plus the final devices and any OOM fallbacks (for ``RunResult``)."""
        d = self.plan.to_dict() if self.plan else {}
        d["final_devices"] = self.devices()
        return d


def build_models(cfg: Config, hardware: Hardware | None = None) -> Models:
    """Create the model holder from config. No weights are downloaded or loaded yet.

    The device planner (``runtime.device_plan``) assigns each model GPU or CPU from the free VRAM
    and the plan is logged; ``hardware`` overrides the detected machine (tests).
    """
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
    models = Models(
        detector=det,
        pose=pose,
        body=BodyReID(cfg),
        face=FaceModel(cfg) if cfg.face.enabled else None,
    )
    hw = hardware or detect_hardware(_gpu_index(cfg))
    plan = plan_devices(cfg, hw)
    models.plan = plan
    for role, m in models.slots().items():
        dev = plan.device(role)
        if dev != m.device:
            m.set_device(dev)
        m.events = plan.events
    ov = plan.overrides
    if "detector" in ov:
        det.imgsz = ov["detector"]["imgsz"]
    if pose is not None and "pose" in ov:
        pose.imgsz = ov["pose"]["imgsz"]
    if models.face is not None and "face" in ov:
        models.face.det_size = ov["face"]["det_size"]
    threads = apply_cpu_threads(cfg, hw, plan)
    log.info("hardware: %s", hw.describe())
    log.info("device plan (%s): %s", plan.mode, plan.summary())
    for a in plan.adjustments:
        log.info("CPU input size: %s", a)
    if threads:
        log.info("CPU models use %d torch threads", threads)
    return models


def _gpu_index(cfg: Config) -> int:
    for spec in (cfg.runtime.detect_device, cfg.runtime.embed_device):
        _, _, idx = str(spec).partition("cuda:")
        if idx.isdigit():
            return int(idx)
    return 0
