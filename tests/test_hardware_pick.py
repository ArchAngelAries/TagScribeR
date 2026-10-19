"""Which GPU is used when several are visible (integrated graphics must not win). Faked device lists, no torch."""
from core.hardware import (DeviceInfo, best_device_index, device_choice_message, is_integrated_gpu, pick_device,
                           pinned_device_env)

GB = 2**30


def gpu(i, name, gb, backend="rocm", arch="", integrated=None):
    return DeviceInfo(id=f"cuda:{i}", backend=backend, name=name, total_memory=int(gb * GB), arch=arch,
                      integrated=integrated)


APU = dict(name="AMD Radeon(TM) Graphics", gb=16, arch="gfx1036")
RX = dict(name="AMD Radeon RX 7900 XTX", gb=24, arch="gfx1100", integrated=False)


def test_apu_first_then_discrete():
    devs = [gpu(0, **APU), gpu(1, **RX)]
    assert best_device_index(devs, {}) == 1
    assert pinned_device_env(devs, {}) == {"HIP_VISIBLE_DEVICES": "1"}
    assert "7900" in device_choice_message(devs, {}) and "integrated" in device_choice_message(devs, {})


def test_discrete_first_then_apu():
    devs = [gpu(0, **RX), gpu(1, **APU)]
    assert best_device_index(devs, {}) == 0
    assert pinned_device_env(devs, {}) == {}


def test_discrete_wins_even_with_less_memory_than_the_igpu():
    devs = [gpu(0, "AMD Radeon(TM) Graphics", 32, integrated=True), gpu(1, "AMD Radeon RX 6600", 8)]
    assert best_device_index(devs, {}) == 1


def test_two_discrete_cards_most_memory_and_ties():
    devs = [gpu(0, "AMD Radeon RX 7600", 8), gpu(1, "AMD Radeon RX 7900 XT", 20)]
    assert best_device_index(devs, {}) == 1
    same = [gpu(0, "AMD Radeon RX 7800 XT", 16), gpu(1, "AMD Radeon RX 7800 XT", 16)]
    assert best_device_index(same, {}) == 0


def test_nvidia_multi_gpu_unchanged_without_igpu():
    devs = [gpu(0, "NVIDIA GeForce RTX 3060", 12, "cuda"), gpu(1, "NVIDIA GeForce RTX 4090", 24, "cuda")]
    assert best_device_index(devs, {}) == 0
    assert pinned_device_env(devs, {}) == {}


def test_nvidia_with_flagged_integrated_gpu_uses_cuda_var():
    devs = [gpu(0, "Some iGPU", 4, "cuda", integrated=True), gpu(1, "NVIDIA GeForce RTX 4070", 12, "cuda")]
    assert best_device_index(devs, {}) == 1
    assert pinned_device_env(devs, {}) == {"CUDA_VISIBLE_DEVICES": "1"}


def test_nvidia_laptop_intel_igpu_invisible_to_torch():
    devs = [gpu(0, "NVIDIA GeForce RTX 4060 Laptop GPU", 8, "cuda")]
    assert best_device_index(devs, {}) == 0
    assert pinned_device_env(devs, {}) == {}


def test_user_set_visible_devices_is_respected():
    devs = [gpu(0, **APU), gpu(1, **RX)]
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        env = {var: "0"}
        assert best_device_index(devs, env) == 0
        assert pinned_device_env(devs, env) == {}
        assert "user" in pick_device(devs, env)[1]


def test_cpu_only_and_non_cuda_devices():
    from core.hardware import CPU
    assert best_device_index([CPU], {}) == 0
    assert pinned_device_env([CPU], {}) == {}
    assert device_choice_message([CPU], {}) == "GPU: none (CPU)"


def test_only_integrated_gpus_picks_most_memory():
    devs = [gpu(0, "AMD Radeon(TM) Graphics", 2, arch="gfx90c"), gpu(1, "AMD Radeon 780M Graphics", 8)]
    assert best_device_index(devs, {}) == 1


def test_integrated_detection_by_name_and_arch():
    def d(name, arch="", integrated=None):
        return gpu(0, name, 8, arch=arch, integrated=integrated)
    for name, arch in [("AMD Radeon(TM) Graphics", ""), ("Radeon Graphics", ""), ("AMD Radeon(TM) Vega 8 Graphics", ""),
                       ("AMD Radeon 780M Graphics", ""), ("AMD Radeon 610M", ""), ("Unknown", "gfx1103"),
                       ("Unknown", "gfx1150:sramecc-:xnack-")]:
        assert is_integrated_gpu(d(name, arch)), (name, arch)
    for name, arch in [("AMD Radeon RX Vega 56", ""), ("AMD Radeon RX 7600M XT", ""), ("AMD Radeon RX 6800", "gfx1030"),
                       ("AMD Radeon PRO W7800", "gfx1100"), ("Radeon VII", "gfx906"), ("AMD Radeon RX 7600", "gfx1102"),
                       ("NVIDIA GeForce RTX 4090", "sm_89")]:
        assert not is_integrated_gpu(d(name, arch)), (name, arch)
    assert is_integrated_gpu(d("Anything", integrated=True))
