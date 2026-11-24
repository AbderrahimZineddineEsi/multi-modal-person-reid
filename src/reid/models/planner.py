"""Device planner: put as many models on the GPU as fit, the rest on CPU.

``plan_devices(cfg, hw)`` is a pure function of the config and a :class:`Hardware` description,
so it is tested with fake machines. Models are placed greedily in order of speed gain per MB:
detector (runs on every frame), body ReID, pose (crops), face (ONNX Runtime, GPU only where its
wheels ship kernels). A model that does not fit goes to CPU and the search continues with the
smaller ones. Explicit ``runtime.detect_device`` / ``embed_device`` values other than ``auto``
are honoured and their cost is deducted first.

``choose_preset`` implements ``--preset auto``.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from reid.config import Config

log = logging.getLogger(__name__)

#: Fraction of the available VRAM kept free (allocator fragmentation, other processes).
SAFETY_MARGIN = 0.12
#: CUDA context of a fresh process (kernels, cuBLAS/cuDNN handles load lazily and are part of the
#: per-model costs below, which were each measured in a fresh process).
CONTEXT_MB = 100.0
PRIORITY = ("detector", "body", "pose", "face")

#: Whole-process GPU memory (MB) per model, ``(base, per_batch_item)``. Measured with
#: ``python -m reid.models.vram_probe`` on a GTX 1050 (torch 2.x, fp32 as Pascal has no fast
#: fp16), model alone in a fresh process, one batch run, NVML per-process figure minus the bare
#: CUDA context. Face packs are *estimates* (onnxruntime GPU cannot run on that card).
COSTS: dict[str, tuple[float, float]] = {
    # detectors (one 1080p frame, imgsz 640; rfdetr-large measured at 704)
    "yolo26n-pose": (180, 0),
    "yolo26s-pose": (300, 0),
    "yolo26m-pose": (490, 0),
    "rfdetr-small": (200, 0),
    "rfdetr-medium": (220, 0),
    "rfdetr-large": (250, 0),
    # body ReID: base + per_item * batch (batch 1 / 8 / 16 measured)
    "clipreid_msmt17": (385, 9),
    "osnet_x1_0_msmt17": (33, 13),
    # face packs (estimates)
    "buffalo_sc": (150, 0),
    "buffalo_l": (350, 0),
    "antelopev2": (450, 0),
}
_DEFAULT_COST = {"detector": 500.0, "pose": 300.0, "body": 600.0, "face": 400.0}


@dataclass
class Hardware:
    """What the planner needs to know about the machine."""

    cuda: bool = False
    gpu_name: str = ""
    total_mb: float = 0.0
    free_mb: float = 0.0
    capability: tuple[int, int] = (0, 0)
    cpu_count: int = 1
    physical_cores: int = 1
    ort_cuda: bool = False  # onnxruntime has the CUDA provider
    context_exists: bool = False  # this process already holds a CUDA context (free excludes it)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        cpu = f"{self.physical_cores} cores ({self.cpu_count} threads)"
        if not self.cuda:
            return f"no CUDA GPU; CPU {cpu}"
        cc = f"sm_{self.capability[0]}{self.capability[1]}"
        return (
            f"{self.gpu_name} {self.total_mb / 1024:.1f} GB ({self.free_mb / 1024:.2f} GB free, "
            f"{cc}); CPU {cpu}"
        )


def detect_hardware(device_index: int = 0) -> Hardware:
    """Query CUDA (torch + NVML) and the CPU. Never raises."""
    n = os.cpu_count() or 1
    hw = Hardware(cpu_count=n, physical_cores=n)
    try:
        import psutil

        hw.physical_cores = psutil.cpu_count(logical=False) or n
    except Exception:
        hw.physical_cores = max(1, n // 2)
    try:
        import torch

        if not torch.cuda.is_available():
            return hw
        hw.cuda = True
        hw.gpu_name = torch.cuda.get_device_name(device_index)
        hw.capability = tuple(torch.cuda.get_device_capability(device_index))  # type: ignore[assignment]
        hw.context_exists = torch.cuda.is_initialized()
        hw.total_mb = torch.cuda.get_device_properties(device_index).total_memory / 2**20
        hw.free_mb = hw.total_mb
    except Exception as exc:
        log.debug("torch CUDA query failed: %s", exc)
        return hw
    try:  # NVML does not create a CUDA context, torch.cuda.mem_get_info does
        import pynvml

        pynvml.nvmlInit()
        info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(device_index))
        hw.total_mb, hw.free_mb = info.total / 2**20, info.free / 2**20
        pynvml.nvmlShutdown()
    except Exception:
        try:
            import torch

            free, total = torch.cuda.mem_get_info(device_index)
            hw.total_mb, hw.free_mb = total / 2**20, free / 2**20
        except Exception as exc:
            log.debug("could not read free VRAM: %s", exc)
    try:
        import onnxruntime as ort

        hw.ort_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        pass
    return hw


@dataclass
class Placement:
    role: str
    model: str
    device: str
    est_mb: float
    reason: str


@dataclass
class DevicePlan:
    """Result of :func:`plan_devices`; ``to_dict()`` goes into ``RunResult.device_plan``."""

    mode: str
    hardware: Hardware
    budget_mb: float | None  # VRAM the plan was allowed to use (None without CUDA)
    placements: dict[str, Placement] = field(default_factory=dict)
    adjustments: list[str] = field(default_factory=list)  # CPU input-size reductions
    #: input-size changes to apply, e.g. ``{"detector": {"imgsz": 640}}``
    overrides: dict[str, dict[str, int]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)  # runtime OOM fallbacks

    def device(self, role: str) -> str:
        return self.placements[role].device

    @property
    def gpu_est_mb(self) -> float:
        return sum(p.est_mb for p in self.placements.values() if p.device.startswith("cuda"))

    @property
    def any_cpu(self) -> bool:
        return any(not p.device.startswith("cuda") for p in self.placements.values())

    def summary_lines(self) -> list[str]:
        out = []
        for role in PRIORITY:
            p = self.placements.get(role)
            if p is None:
                continue
            est = f" (est {p.est_mb / 1024:.2f} GB)" if p.device.startswith("cuda") else ""
            out.append(f"{role} {p.model} -> {p.device}{est}" + ("" if est else f" ({p.reason})"))
        return out

    def summary(self) -> str:
        return ", ".join(self.summary_lines())

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "hardware": self.hardware.to_dict(),
            "budget_mb": self.budget_mb,
            "placements": {r: asdict(p) for r, p in self.placements.items()},
            "adjustments": list(self.adjustments),
            "events": list(self.events),
            "summary": self.summary(),
        }


def model_cost_mb(role: str, name: str, batch: int) -> float:
    """Estimated GPU memory (MB) of a model for the given batch size."""
    key = re.sub(r"\.(pt|onnx)$", "", name)
    base, per = COSTS.get(key, (_DEFAULT_COST[role], 0.0))
    return base + per * batch


def _models(cfg: Config) -> dict[str, tuple[str, float]]:
    """role -> (model name, estimated MB) for the configured pipeline."""
    b = cfg.runtime.batch_size
    out = {
        "detector": (cfg.detection.model, model_cost_mb("detector", cfg.detection.model, 1)),
        "body": (cfg.body_reid.model, model_cost_mb("body", cfg.body_reid.model, b)),
    }
    if cfg.quality.pose_model:
        out["pose"] = (cfg.quality.pose_model, model_cost_mb("pose", cfg.quality.pose_model, b))
    if cfg.face.enabled:
        out["face"] = (cfg.face.model, model_cost_mb("face", cfg.face.model, 1))
    return out


def _gpu_name(spec: str) -> str:
    """``cuda`` or ``cuda:N`` for a device spec."""
    return spec if spec.startswith("cuda:") else "cuda"


def plan_devices(cfg: Config, hw: Hardware) -> DevicePlan:
    """Decide the device of every model (see the module docstring)."""
    rt = cfg.runtime
    mode = rt.device_plan
    models = _models(cfg)
    usable: float | None = None
    budget: float | None = None
    if hw.cuda:
        budget = rt.vram_budget_gb * 1024 if rt.vram_budget_gb is not None else hw.free_mb
        usable = budget * (1 - SAFETY_MARGIN) - (0 if hw.context_exists else CONTEXT_MB)
    plan = DevicePlan(mode=mode, hardware=hw, budget_mb=budget)
    face_gpu_ok = hw.capability[0] >= 7 and hw.ort_cuda
    specs = {
        "detector": rt.detect_device,
        **dict.fromkeys(("body", "pose", "face"), rt.embed_device),
    }

    def face_blocker() -> str:
        if not hw.ort_cuda:
            return "onnxruntime has no CUDA provider"
        return f"onnxruntime GPU unsupported on sm_{hw.capability[0]}{hw.capability[1]}"

    # pass 1: explicit devices and hard limits; pass 2: greedy by priority
    pending = []
    for role in PRIORITY:
        if role not in models:
            continue
        name, est = models[role]
        spec = str(specs[role]).lower()
        if mode == "cpu" or spec == "cpu":
            why = "device_plan=cpu" if mode == "cpu" else "pinned to cpu by runtime config"
            plan.placements[role] = Placement(role, name, "cpu", est, why)
        elif not hw.cuda:
            plan.placements[role] = Placement(role, name, "cpu", est, "no CUDA GPU")
        elif role == "face" and not face_gpu_ok:
            plan.placements[role] = Placement(role, name, "cpu", est, face_blocker())
        elif spec != "auto" and spec.startswith("cuda"):
            plan.placements[role] = Placement(role, name, _gpu_name(spec), est, "pinned by config")
            assert usable is not None
            usable -= est
        else:
            pending.append(role)
    for role in pending:
        name, est = models[role]
        assert usable is not None
        if mode == "gpu" or est <= usable:
            plan.placements[role] = Placement(role, name, "cuda", est, "fits")
            usable -= est
        else:
            why = f"no room: needs {est:.0f} MB, {max(usable, 0):.0f} MB left"
            plan.placements[role] = Placement(role, name, "cpu", est, why)
    plan.placements = {r: plan.placements[r] for r in PRIORITY if r in plan.placements}
    if rt.cpu_downsize:
        _cpu_downsize(cfg, plan)
    return plan


def _cpu_downsize(cfg: Config, plan: DevicePlan) -> None:
    """Smaller inputs for models that run on CPU (logged; never touches the GPU path)."""
    P = plan.placements
    det, pose, face = P.get("detector"), P.get("pose"), P.get("face")
    if det and det.device == "cpu" and "rfdetr" not in cfg.detection.model:
        if cfg.detection.imgsz > 640:
            plan.overrides["detector"] = {"imgsz": 640}
            plan.adjustments.append(f"detector imgsz {cfg.detection.imgsz} -> 640 (CPU)")
    if pose and pose.device == "cpu" and cfg.quality.pose_imgsz > 320:
        plan.overrides["pose"] = {"imgsz": 320}
        plan.adjustments.append(f"pose imgsz {cfg.quality.pose_imgsz} -> 320 (CPU)")
    if face and face.device == "cpu" and cfg.face.det_size > 480:
        plan.overrides["face"] = {"det_size": 480}
        plan.adjustments.append(f"face det_size {cfg.face.det_size} -> 480 (CPU)")


def apply_cpu_threads(cfg: Config, hw: Hardware, plan: DevicePlan) -> int | None:
    """Use all physical cores for torch when something runs on CPU; returns the thread count."""
    if not plan.any_cpu:
        return None
    n = cfg.runtime.cpu_threads or hw.physical_cores
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        return None
    return n


def choose_preset(
    hw: Hardware, budget_gb: float | None = None, device_plan: str = "auto"
) -> tuple[str, str]:
    """Preset for ``--preset auto`` and the reason.

    No CUDA (or ``device_plan=cpu``) or < 2 GB VRAM -> ``lite``; 2-6 GB -> ``balanced``;
    >= 6 GB -> ``accurate``. ``max`` is never picked automatically: it is slower for a small
    accuracy gain, so it stays an explicit choice.
    """
    if device_plan == "cpu" or not hw.cuda:
        return "lite", "CPU only" if not hw.cuda else "device_plan=cpu"
    gb = (budget_gb * 1024 if budget_gb is not None else hw.free_mb) / 1024
    src = "VRAM budget" if budget_gb is not None else "free VRAM"
    if gb < 2:
        return "lite", f"{gb:.1f} GB {src} (< 2 GB)"
    if gb < 6:
        return "balanced", f"{gb:.1f} GB {src} (2-6 GB)"
    return "accurate", f"{gb:.1f} GB {src} (>= 6 GB)"
