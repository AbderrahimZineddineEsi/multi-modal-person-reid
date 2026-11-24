"""Device planner (fake hardware), auto preset and the runtime OOM fallback. CPU only."""

import numpy as np
import pytest

from reid.config import load_config
from reid.models.placement import Placeable, is_oom
from reid.models.planner import (
    Hardware,
    choose_preset,
    detect_hardware,
    model_cost_mb,
    plan_devices,
)

NO_GPU = Hardware(cpu_count=8, physical_cores=4)


def gpu(total_gb: float, cc=(7, 5), name="Fake GPU", ort=True) -> Hardware:
    mb = total_gb * 1024
    return Hardware(
        cuda=True,
        gpu_name=name,
        total_mb=mb,
        free_mb=mb,
        capability=cc,
        cpu_count=8,
        physical_cores=4,
        ort_cuda=ort,
    )


def devices(plan):
    return {r: p.device for r, p in plan.placements.items()}


def test_no_gpu_everything_on_cpu():
    plan = plan_devices(load_config(preset="balanced"), NO_GPU)
    assert set(devices(plan).values()) == {"cpu"}
    assert "no CUDA" in plan.placements["detector"].reason


def test_t4_15gb_everything_on_gpu():
    plan = plan_devices(load_config(preset="accurate"), gpu(15))
    assert set(devices(plan)) == {"detector", "body", "pose", "face"}
    assert set(devices(plan).values()) == {"cuda"}
    assert plan.adjustments == []


def test_pascal_3gb_faces_on_cpu_rest_on_gpu():
    plan = plan_devices(load_config(preset="balanced"), gpu(2.9, cc=(6, 1)))
    d = devices(plan)
    assert d["detector"] == d["body"] == "cuda" and d["face"] == "cpu"
    assert "sm_61" in plan.placements["face"].reason
    assert plan.gpu_est_mb < 2.9 * 1024


def test_2gb_gpu_balanced_spills_body_to_cpu():
    cfg = load_config(preset="balanced")  # detector 490 + body 529 MB
    plan = plan_devices(cfg, gpu(1.2))
    d = devices(plan)
    assert d["detector"] == "cuda" and d["body"] == "cpu"
    assert "no room" in plan.placements["body"].reason


def test_priority_detector_first_then_smaller_models_still_fit():
    cfg = load_config(overrides={"runtime.vram_budget_gb": 0.4}, preset="lite")
    plan = plan_devices(cfg, gpu(8))
    # lite: detector 180, osnet 33+13*8: all fit in 0.4 GB * 0.88 - 100
    assert devices(plan)["detector"] == "cuda"
    cfg = load_config(overrides={"runtime.vram_budget_gb": 0.4}, preset="balanced")
    d = devices(plan_devices(cfg, gpu(8)))
    assert d["detector"] == "cpu" and d["body"] == "cpu"


def test_budget_overrides_free_vram():
    hw = gpu(15)
    cfg = load_config(overrides={"runtime.vram_budget_gb": 0.2}, preset="balanced")
    plan = plan_devices(cfg, hw)
    assert plan.budget_mb == pytest.approx(0.2 * 1024)
    assert devices(plan)["detector"] == "cpu"


def test_device_plan_cpu_and_gpu_modes():
    cfg = load_config(overrides={"runtime.device_plan": "cpu"}, preset="balanced")
    assert set(devices(plan_devices(cfg, gpu(15))).values()) == {"cpu"}
    cfg = load_config(overrides={"runtime.device_plan": "gpu"}, preset="balanced")
    d = devices(plan_devices(cfg, gpu(0.5)))  # forced even when it does not fit
    assert d["detector"] == d["body"] == d["face"] == "cuda"


def test_explicit_device_pins_and_is_deducted():
    cfg = load_config(overrides={"runtime.embed_device": "cpu"}, preset="balanced")
    d = devices(plan_devices(cfg, gpu(15)))
    assert d["body"] == d["face"] == "cpu" and d["detector"] == "cuda"
    cfg = load_config(overrides={"runtime.detect_device": "cuda:1"}, preset="balanced")
    assert plan_devices(cfg, gpu(15)).placements["detector"].device == "cuda:1"


def test_cpu_downsize_logged_and_optional():
    cfg = load_config(preset="max")  # rfdetr-large, pose 384, antelopev2 640
    plan = plan_devices(cfg, NO_GPU)
    assert plan.overrides["pose"] == {"imgsz": 320}
    assert plan.overrides["face"] == {"det_size": 480}
    assert "detector" not in plan.overrides  # RF-DETR resizes by itself
    assert plan.adjustments
    cfg = load_config(overrides={"runtime.cpu_downsize": False}, preset="max")
    assert plan_devices(cfg, NO_GPU).overrides == {}


def test_plan_to_dict_and_summary():
    plan = plan_devices(load_config(preset="balanced"), gpu(2.9, cc=(6, 1)))
    d = plan.to_dict()
    assert d["placements"]["face"]["device"] == "cpu" and "detector" in d["summary"]
    assert d["hardware"]["capability"] == (6, 1)


def test_cost_scales_with_batch():
    assert model_cost_mb("body", "clipreid_msmt17", 16) > model_cost_mb(
        "body", "clipreid_msmt17", 1
    )
    assert model_cost_mb("detector", "unknown.pt", 1) > 0


@pytest.mark.parametrize(
    ("hw", "preset"),
    [
        (NO_GPU, "lite"),
        (gpu(1.5), "lite"),
        (gpu(3, cc=(6, 1)), "balanced"),
        (gpu(5.9), "balanced"),
        (gpu(15), "accurate"),
    ],
)
def test_choose_preset(hw, preset):
    assert choose_preset(hw)[0] == preset


def test_choose_preset_budget_and_cpu_mode():
    assert choose_preset(gpu(15), budget_gb=1.0)[0] == "lite"
    assert choose_preset(gpu(15), device_plan="cpu")[0] == "lite"


def test_load_config_auto_preset(monkeypatch):
    monkeypatch.setattr("reid.models.planner.detect_hardware", lambda *a: gpu(15))
    assert load_config(preset="auto").detection.model == "rfdetr-medium"  # accurate
    monkeypatch.setattr("reid.models.planner.detect_hardware", lambda *a: NO_GPU)
    cfg = load_config(overrides={"runtime.batch_size": 3}, preset="auto")
    assert cfg.detection.model == "yolo26n-pose.pt" and cfg.runtime.batch_size == 3  # lite


def test_detect_hardware_never_raises():
    hw = detect_hardware()
    assert hw.physical_cores >= 1 and isinstance(hw.describe(), str)


# ------------------------------------------------------------------ OOM fallback
torch = pytest.importorskip("torch")


class FakeModel(Placeable):
    role = "fake"

    def __init__(self, oom_times: int, device="cuda", batch=8):
        self.device, self.batch, self.oom_times = device, batch, oom_times
        self.unloaded = 0
        self.seen: list[tuple[str, int]] = []

    def _unload(self):
        self.unloaded += 1

    def run(self):
        def go():
            self.seen.append((self.device, self.batch))
            if self.device.startswith("cuda") and self.oom_times > 0:
                self.oom_times -= 1
                raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
            return np.ones(2)

        return self.guarded(go)


def test_is_oom():
    assert is_oom(torch.cuda.OutOfMemoryError("x"))
    assert is_oom(RuntimeError("CUDA error: out of memory"))
    assert not is_oom(RuntimeError("shape mismatch"))
    assert not is_oom(ValueError("out of memory"))


def test_oom_once_halves_batch_and_stays_on_gpu():
    m = FakeModel(oom_times=1)
    m.events = []
    assert m.run().sum() == 2
    assert m.seen == [("cuda", 8), ("cuda", 4)] and m.device == "cuda"
    assert m.events[0]["role"] == "fake" and "batch 8 -> 4" in m.events[0]["action"]


def test_oom_twice_moves_to_cpu_and_continues():
    m = FakeModel(oom_times=2)
    m.events = []
    assert m.run().sum() == 2
    assert m.device == "cpu" and m.half is False and m.unloaded == 1
    assert m.seen[-1] == ("cpu", 4)
    assert m.run().sum() == 2  # later calls stay on CPU
    assert len(m.events) == 2


def test_oom_without_batch_goes_straight_to_cpu():
    m = FakeModel(oom_times=1, batch=1)
    assert m.run().sum() == 2 and m.device == "cpu"


def test_non_oom_errors_propagate_and_cpu_oom_is_not_retried():
    m = FakeModel(oom_times=0)
    with pytest.raises(ValueError):
        m.guarded(lambda: (_ for _ in ()).throw(ValueError("boom")))
    cpu = FakeModel(oom_times=0, device="cpu")

    def oom():
        raise torch.cuda.OutOfMemoryError("x")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        cpu.guarded(oom)


def test_pipeline_survives_models_without_guard():
    from reid.models.registry import Models

    class Dummy:
        loaded = False

        def load(self):
            self.loaded = True

    d = Dummy()
    Models(detector=d, pose=None, body=Dummy(), face=None).warm()
    assert d.loaded
