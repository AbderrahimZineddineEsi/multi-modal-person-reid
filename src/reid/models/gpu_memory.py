"""Peak GPU memory of the whole process, as ``nvidia-smi`` sees it.

``torch.cuda.max_memory_allocated`` only counts tensors PyTorch allocated. What a GPU must
really have free to run the pipeline is larger:

* the CUDA context of the process (driver, kernels, cuBLAS/cuDNN handles: ~300-500 MB);
* memory PyTorch keeps reserved in its cache (``memory_reserved`` >= ``memory_allocated``);
* ONNX Runtime allocations (the InsightFace face models), invisible to PyTorch.

:class:`GpuMemoryMonitor` samples NVML in a background thread and keeps the peak. It prefers the
per-process figure (``nvmlDeviceGetComputeRunningProcesses``); some containers (Colab) hide
per-process usage, so it falls back to the device-wide used memory minus what was already used
when monitoring started.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

_MB = 2**20


@dataclass
class GpuMemoryPeak:
    """Peak figures in MiB (None when unknown)."""

    process_mb: float | None = None  # this process (NVML per-process); the real requirement
    device_delta_mb: float | None = None  # device used minus baseline (fallback; other users count)
    device_total_mb: float | None = None
    torch_reserved_mb: float | None = None  # PyTorch allocator cache peak
    torch_allocated_mb: float | None = None  # PyTorch tensors peak
    gpu_name: str | None = None
    source: str = "none"  # "process", "device" or "none"

    @property
    def required_mb(self) -> float | None:
        """Best estimate of the GPU memory this process needs."""
        return self.process_mb if self.process_mb is not None else self.device_delta_mb

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["required_mb"] = self.required_mb
        return d


class GpuMemoryMonitor:
    """Context manager sampling GPU memory every ``interval_s`` seconds; see :attr:`peak`."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.2) -> None:
        self.index, self.interval = device_index, interval_s
        self.peak = GpuMemoryPeak()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml: Any = None
        self._handle: Any = None
        self._baseline = 0.0

    # ---------------------------------------------------------------- lifecycle
    def __enter__(self) -> GpuMemoryMonitor:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
            name = pynvml.nvmlDeviceGetName(self._handle)
            self.peak.gpu_name = name.decode() if isinstance(name, bytes) else str(name)
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._baseline = info.used / _MB
            self.peak.device_total_mb = info.total / _MB
        except Exception as exc:  # no NVIDIA driver, no pynvml, no GPU
            log.debug("GPU memory monitor disabled: %s", exc)
            self._nvml = None
            return self
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="gpu-mem", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._nvml is not None:
            self._sample()
            self._torch_peaks()
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    # ---------------------------------------------------------------- sampling
    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self) -> None:
        p, nv = self.peak, self._nvml
        try:
            used = nv.nvmlDeviceGetMemoryInfo(self._handle).used / _MB
            p.device_delta_mb = max(p.device_delta_mb or 0.0, used - self._baseline)
            mine = self._process_used_mb()
            if mine is not None:
                p.process_mb = max(p.process_mb or 0.0, mine)
            p.source = "process" if p.process_mb is not None else "device"
        except Exception as exc:  # a failed sample must never break processing
            log.debug("GPU memory sample failed: %s", exc)

    def _process_used_mb(self) -> float | None:
        nv, pid = self._nvml, os.getpid()
        for fn in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
            try:
                procs = getattr(nv, fn)(self._handle)
            except Exception:
                continue
            for pr in procs:
                used = getattr(pr, "usedGpuMemory", None)
                if pr.pid == pid and used is not None:  # None: hidden by the driver/container
                    return used / _MB
        return None

    def _torch_peaks(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                self.peak.torch_reserved_mb = torch.cuda.max_memory_reserved() / _MB
                self.peak.torch_allocated_mb = torch.cuda.max_memory_allocated() / _MB
        except Exception:
            pass


def format_peak(p: GpuMemoryPeak | dict[str, Any] | None) -> str:
    """One human line, e.g. ``1.42 GB needed on Tesla T4 (process peak; PyTorch 0.93 GB)``."""
    d = p.to_dict() if isinstance(p, GpuMemoryPeak) else (p or {})
    req = d.get("required_mb")
    if not req:
        return "GPU memory: not measured (no NVIDIA GPU / NVML)"
    how = "process peak" if d.get("source") == "process" else "device increase during the run"
    extra = ""
    if d.get("torch_reserved_mb"):
        extra = f"; PyTorch reserved {d['torch_reserved_mb'] / 1024:.2f} GB"
    gpu = f" on {d['gpu_name']}" if d.get("gpu_name") else ""
    return f"GPU memory: {req / 1024:.2f} GB needed{gpu} ({how}{extra})"
