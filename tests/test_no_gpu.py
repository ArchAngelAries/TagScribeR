"""The suite hides the GPU (tests/conftest.py), so running it never competes with a training run for VRAM."""
import pytest


def test_suite_cannot_see_a_gpu():
    torch = pytest.importorskip("torch")
    assert not torch.cuda.is_available()
