"""Device and precision helpers shared by all model wrappers."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def resolve_device(spec: str) -> str:
    """Map a config device string to a torch device string.

    ``"auto"`` becomes ``"cuda"`` if CUDA is usable, else ``"cpu"``. A ``cuda*`` request without
    a working CUDA falls back to ``"cpu"`` with a warning instead of crashing.
    """
    import torch

    spec = (spec or "auto").lower()
    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if spec.startswith("cuda") and not torch.cuda.is_available():
        log.warning("device %r requested but CUDA is unavailable; using cpu", spec)
        return "cpu"
    return spec


def use_half(device: str, want: bool) -> bool:
    """Whether to run in fp16: only on CUDA devices with fast half (compute capability >= 7).

    Pascal consumer GPUs (sm_61, e.g. GTX 1050) execute fp16 at a fraction of fp32 speed, so
    ``runtime.fp16`` is ignored there.
    """
    import torch

    if not want or not device.startswith("cuda"):
        return False
    major, _ = torch.cuda.get_device_capability(torch.device(device))
    if major < 7:
        log.info("fp16 disabled on %s (compute capability %d.x has no fast half)", device, major)
        return False
    return True


def onnx_providers(device: str) -> tuple[list[str], int]:
    """ONNX Runtime providers and insightface ``ctx_id`` for a torch-style device string.

    CUDA is skipped on pre-Volta GPUs: the onnxruntime-gpu 1.32 wheels fail at ``cublasCreate``
    on a GTX 1050 (verified), so the face models run on CPU there.
    """
    import onnxruntime as ort
    import torch

    avail = ort.get_available_providers()
    if device.startswith("cuda") and torch.cuda.get_device_capability(torch.device(device))[0] < 7:
        log.info("onnxruntime-gpu wheels no longer ship Pascal (sm_6x) kernels: face models on CPU")
        return ["CPUExecutionProvider"], -1
    if device.startswith("cuda") and "CUDAExecutionProvider" in avail:
        idx = int(device.split(":")[1]) if ":" in device else 0
        return [("CUDAExecutionProvider", {"device_id": idx}), "CPUExecutionProvider"], idx  # type: ignore[list-item]
    return ["CPUExecutionProvider"], -1


def weights_dir(configured: str = "") -> Path:
    """Directory where downloaded weights are cached (``~/.cache/reid/weights`` by default)."""
    d = Path(configured).expanduser() if configured else Path.home() / ".cache" / "reid" / "weights"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ultralytics_weights(name: str, configured_dir: str = "") -> str:
    """Local path of an Ultralytics weights file, downloading it into the weights cache.

    Names that already point to an existing file (or contain a directory) are used as given.
    Keeps downloads out of the working directory.
    """
    p = Path(name).expanduser()
    if p.exists() or p.parent != Path("."):
        return str(p)
    target = weights_dir(configured_dir) / name
    if not target.exists():
        from ultralytics.utils.downloads import attempt_download_asset

        attempt_download_asset(str(target))
    return str(target)
