import pytest

from reid.models.gpu_memory import GpuMemoryMonitor, GpuMemoryPeak, format_peak


def test_format_without_measurement():
    assert "not measured" in format_peak(None)
    assert "not measured" in format_peak(GpuMemoryPeak())


def test_required_prefers_process_figure():
    p = GpuMemoryPeak(process_mb=1500.0, device_delta_mb=1800.0, source="process", gpu_name="X")
    assert p.required_mb == 1500.0
    assert "1.46 GB needed on X (process peak" in format_peak(p)
    assert GpuMemoryPeak(device_delta_mb=900.0, source="device").required_mb == 900.0


def test_monitor_never_raises_and_measures_torch_on_gpu():
    torch = pytest.importorskip("torch")
    with GpuMemoryMonitor(interval_s=0.05) as mon:
        if torch.cuda.is_available():
            x = torch.empty(64 * 2**20, dtype=torch.uint8, device="cuda")  # 64 MiB
            del x
    if torch.cuda.is_available():
        assert mon.peak.required_mb is not None and mon.peak.required_mb >= 64
        assert mon.peak.torch_allocated_mb >= 64
