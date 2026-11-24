"""Device placement of one model and the runtime out-of-memory fallback.

:class:`Placeable` is mixed into the four model wrappers. It lets the device planner assign a
device after construction (:meth:`set_device`) and keeps processing alive when the GPU runs out
of memory: :meth:`guarded` retries the failed call with half the batch (once) and then on CPU.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from reid.models.device import use_half

log = logging.getLogger(__name__)

T = TypeVar("T")

_OOM_MARKERS = (
    "out of memory",
    "cudaerrormemoryallocation",
    "cublas_status_alloc_failed",
    "failed to allocate memory",  # onnxruntime BFC arena
    "cudnn_status_alloc_failed",
)


def is_oom(exc: BaseException) -> bool:
    """True if ``exc`` is a CUDA / ONNX Runtime GPU out-of-memory error."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    msg = str(exc).lower()
    return isinstance(exc, (RuntimeError, MemoryError, OSError)) and any(
        m in msg for m in _OOM_MARKERS
    )


def free_cuda_cache() -> None:
    """Drop Python garbage and PyTorch's cached GPU blocks (no-op without CUDA)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # never let cleanup break the fallback
        pass


class Placeable:
    """Mixin: ``device``, ``half``, ``batch`` (optional) plus OOM-safe execution.

    Subclasses set ``role``, ``device`` and ``_want_half`` in ``__init__`` and implement
    :meth:`_unload` (drop the loaded weights; they reload lazily on the new device) and, when
    they cache precision-dependent arguments, :meth:`_on_device_change`.
    """

    role = "model"
    device: str = "cpu"
    half: bool = False
    _want_half: bool = False
    #: shared list of fallback events (set by ``build_models``), e.g. ``{"role": "body", ...}``
    events: list[dict[str, Any]] | None = None
    _oom_halved = False

    def _unload(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _on_device_change(self) -> None:
        """Hook: recompute cached precision arguments after ``device``/``half`` changed."""

    @property
    def on_gpu(self) -> bool:
        return str(self.device).startswith("cuda")

    def set_device(self, device: str) -> None:
        """Move this model to ``device`` (weights reload lazily; fp16 only on CUDA)."""
        changed = device != self.device
        self.device = device
        self.half = use_half(device, self._want_half) if device.startswith("cuda") else False
        self._on_device_change()
        if changed:
            self._unload()

    def to_cpu(self) -> None:
        """Free this model's GPU memory and continue on CPU."""
        self.set_device("cpu")
        free_cuda_cache()

    def _record(self, action: str, why: str) -> None:
        log.warning("%s: CUDA out of memory (%s): %s", self.role, why[:80], action)
        if self.events is not None:
            self.events.append({"role": self.role, "action": action})

    def guarded(self, fn: Callable[[], T]) -> T:
        """Run ``fn``; on GPU OOM halve ``self.batch`` once (if any) and retry, then use CPU."""
        try:
            return fn()
        except Exception as exc:
            if not (self.on_gpu and is_oom(exc)):
                raise
            err = str(exc)
        free_cuda_cache()
        batch = getattr(self, "batch", 1)
        if batch > 1 and not self._oom_halved:
            self._oom_halved = True
            self.batch = max(1, batch // 2)  # type: ignore[attr-defined]
            self._record(f"batch {batch} -> {self.batch}, retrying", err)  # type: ignore[attr-defined]
            try:
                return fn()
            except Exception as exc:
                if not is_oom(exc):
                    raise
                err = str(exc)
            free_cuda_cache()
        self._record("moved to CPU, retrying", err)
        self.to_cpu()
        return fn()
