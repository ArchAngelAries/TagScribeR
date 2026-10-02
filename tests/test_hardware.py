import pytest

from core import hardware


def test_cpu_always_present():
    assert hardware.detect_devices()[-1].backend == "cpu"


def test_resolve_dtype_rules():
    torch = pytest.importorskip("torch")
    gpu_bf16 = hardware.DeviceInfo(id="cuda:0", backend="rocm", name="x", bf16=True)
    gpu_old = hardware.DeviceInfo(id="cuda:0", backend="cuda", name="x", bf16=False)
    assert hardware.resolve_dtype(gpu_bf16) == torch.bfloat16
    assert hardware.resolve_dtype(gpu_old) == torch.float16
    assert hardware.resolve_dtype(hardware.CPU) == torch.float32
    assert hardware.resolve_dtype(gpu_old, "bfloat16") == torch.float16
    assert hardware.resolve_dtype(hardware.CPU, "float16") == torch.float32


def test_unknown_device_falls_back():
    assert hardware.resolve_device("cuda:99") == hardware.detect_devices()[0]


def test_describe_environment_runs():
    assert "Python" in hardware.describe_environment()
