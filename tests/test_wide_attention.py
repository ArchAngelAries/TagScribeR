"""Attention for heads wider than 256 (the VAEs' single head): the hand-computed path matches the reference maths,
and the VAEs that have such a head use it. CPU only."""
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from training.modules import wide_attention as W  # noqa: E402


def _reference(q, k, v):
    return torch.softmax((q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5, dim=-1) @ v


@pytest.mark.parametrize("dim", [384, 512])
def test_wide_head_is_computed_by_hand_and_matches(dim, monkeypatch):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 1, 300, dim) for _ in range(3))

    def boom(*_a, **_k):
        raise AssertionError("the built-in attention must not be used for a wide head")
    monkeypatch.setattr(W.F, "scaled_dot_product_attention", boom)
    monkeypatch.setattr(W, "_CHUNK_ELEMENTS", 300 * 7)                  # force several row chunks
    out = W.attention(q, k, v)
    assert out.dtype == q.dtype and torch.allclose(out, _reference(q, k, v), atol=1e-5)
    half = W.attention(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16))
    assert half.dtype == torch.bfloat16                                 # computed in float32, returned in the input dtype
    assert torch.allclose(half.float(), _reference(q, k, v), atol=3e-2)


def test_ordinary_heads_use_the_builtin():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 50, 64) for _ in range(3))
    assert torch.allclose(W.attention(q, k, v), F.scaled_dot_product_attention(q, k, v), atol=1e-6)


def test_the_vaes_with_a_wide_head_use_it():
    import inspect

    from training.families.klein import vae as klein_vae
    from training.families.krea2 import vae as krea2_vae
    from training.families.qwen_image21 import vae as qwen_vae
    from training.families.sdxl import vae as sdxl_vae
    for mod in (krea2_vae, qwen_vae, klein_vae):
        src = inspect.getsource(mod)
        assert "wide_attention(q, k, v)" in src and "scaled_dot_product_attention(q, k, v)" not in src, mod.__name__
    assert "set_attn_processor(AttnProcessor())" in inspect.getsource(sdxl_vae)
