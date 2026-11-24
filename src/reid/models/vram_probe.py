"""Measure the GPU memory one model needs, alone in a fresh process (feeds ``planner.COSTS``).

``python -m reid.models.vram_probe detector yolo26m-pose.pt --imgsz 640`` prints one JSON line.
Numbers are whole-process NVML figures (what ``nvidia-smi`` shows for the process): the CUDA
context alone (``context_mb``) and after loading and running one batch (``used_mb``). The model's
cost is ``used_mb - context_mb``; the planner adds the context once.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _used_mb() -> float:
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    for pr in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
        if pr.pid == os.getpid() and pr.usedGpuMemory is not None:
            return pr.usedGpuMemory / 2**20
    return pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20


def probe(role: str, model: str, batch: int, imgsz: int) -> dict:
    import torch

    from reid.config import load_config

    ov: dict = {"runtime.batch_size": batch, "face.enabled": role == "face"}
    if role == "detector":
        ov.update({"detection.model": model, "detection.imgsz": imgsz})
    elif role == "pose":
        ov.update({"quality.pose_model": model, "quality.pose_imgsz": imgsz})
    elif role == "body":
        dim = 1280 if model.startswith("clip") else 512
        ov.update({"body_reid.model": model, "body_reid.dim": dim})
    cfg = load_config(overrides=ov)
    torch.zeros(1, device="cuda")  # create the CUDA context
    ctx = _used_mb()
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    if role == "detector":
        from reid.models.detector import PersonDetector

        m = PersonDetector(cfg)
        m.set_device("cuda")
        m.load()
        m.detect(frame)
    elif role == "pose":
        from reid.models.pose import PoseEstimator

        m = PoseEstimator(cfg)
        m.set_device("cuda")
        m.load()
        boxes = np.array([[100 + 20 * i, 100, 200 + 20 * i, 400] for i in range(batch)], np.float32)
        m.keypoints(frame, boxes)
    elif role == "body":
        from reid.models.body_reid import BodyReID

        m = BodyReID(cfg)
        m.set_device("cuda")
        m.load()
        m.embed([frame[:300, :150].copy() for _ in range(batch)])
    else:
        raise SystemExit(f"unsupported role {role!r}")
    torch.cuda.synchronize()
    used = _used_mb()
    return {
        "role": role,
        "model": model,
        "batch": batch,
        "imgsz": imgsz,
        "context_mb": round(ctx),
        "used_mb": round(used),
        "cost_mb": round(used - ctx),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("role", choices=["detector", "pose", "body"])
    ap.add_argument("model")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    print(json.dumps(probe(a.role, a.model, a.batch, a.imgsz)))


if __name__ == "__main__":
    main()
