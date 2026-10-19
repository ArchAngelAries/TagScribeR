"""MiniMax H3 pre-sm_80 NVIDIA paths (Fizgig b94ffcaf): chunked text-encoder prefill and fp16 attention fallback.
CPU only, tiny random models, a faked CUDA device. Not run on real V100/T4 hardware."""
import types

import pytest
import torch

from training.families.minimax_h3 import embedder as E
from training.families.minimax_h3 import model as M


def _fake_cuda(monkeypatch, cap=(7, 0), hip=None, free=2 ** 31):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx=None: cap)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda idx=None: (free, free * 2))
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)


class _Tok:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1 + (ord(c) % 20) for c in text]])}


def _te(cpu_embed=False):
    pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = E.build_qwen3_te(dict(hidden_size=32, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
                                  head_dim=16, intermediate_size=48, vocab_size=64)).eval()
    return E.H3TextEncoder(model, _Tok(), device="cpu", compute_dtype=torch.float32, cpu_embed=cpu_embed)


@pytest.mark.parametrize("cpu_embed", [False, True])
@pytest.mark.parametrize("chunk", [1, 7, 16])
def test_chunked_prefill_matches_single_pass(monkeypatch, chunk, cpu_embed):
    te = _te(cpu_embed)
    ids = torch.randint(1, 60, (2, 40))
    with torch.no_grad():
        ref = te._forward(ids)
        monkeypatch.setattr(te, "_plan_chunk", lambda i: chunk)
        got = te._forward(ids)
    assert got.shape == ref.shape and torch.allclose(got, ref, atol=1e-5)


def test_chunk_planner_gate(monkeypatch):
    te = _te()
    long_ids = torch.zeros(1, 2400, dtype=torch.long)
    short_ids = torch.zeros(1, 100, dtype=torch.long)
    te.device = torch.device("cuda", 0)
    _fake_cuda(monkeypatch, (7, 0))
    c = te._plan_chunk(long_ids)
    assert c is not None and 128 <= c <= 2048
    assert te._plan_chunk(short_ids) is None                 # small table: single pass
    _fake_cuda(monkeypatch, (8, 6))
    assert te._plan_chunk(long_ids) is None                  # sm_80+
    _fake_cuda(monkeypatch, (7, 0), hip="6.2")
    assert te._plan_chunk(long_ids) is None                  # ROCm/HIP reports a capability too
    te.device = torch.device("cpu")
    _fake_cuda(monkeypatch, (7, 0))
    assert te._plan_chunk(long_ids) is None                  # CPU


def test_fp16_attention_matches_default():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 64, 16).bfloat16() for _ in range(3))
    ref = M.F.scaled_dot_product_attention(q, k, v)
    got = M._sdpa_fp16(q, k, v)
    assert got.dtype == torch.bfloat16 and torch.allclose(got.float(), ref.float(), atol=3e-2, rtol=3e-2)


class _Q:
    """Stands in for a CUDA bf16 [1, H, S, D] tensor: only the attributes the gate reads."""
    def __init__(self, s, cuda=True, dtype=torch.bfloat16):
        self.is_cuda, self.dtype = cuda, dtype
        self.device = types.SimpleNamespace(index=0)
        self.shape = (1, 8, s, 16)


@pytest.fixture
def fresh_state():
    M._BF16_SDPA.update(state="unknown", max_s=0, logged=set())
    yield
    M._BF16_SDPA.update(state="unknown", max_s=0, logged=set())


def test_attention_gate_selects_fp16_only_on_fake_sm70(monkeypatch, fresh_state):
    _fake_cuda(monkeypatch, (7, 0), free=2 ** 30)            # 1 GiB free -> a modest fit boundary
    assert M._use_fp16_attention(_Q(100)) is False           # fits bf16
    smax = M._BF16_SDPA["max_s"]
    assert M._BF16_SDPA["state"] == "active" and smax > 100
    assert M._use_fp16_attention(_Q(smax + 1)) is True       # too long -> fp16
    assert M._use_fp16_attention(_Q(smax + 1, dtype=torch.float16)) is False
    assert M._use_fp16_attention(_Q(smax + 1, cuda=False)) is False   # CPU

    M._BF16_SDPA.update(state="unknown", max_s=0, logged=set())
    _fake_cuda(monkeypatch, (7, 0), hip="6.2", free=2 ** 30)  # HIP never
    assert M._use_fp16_attention(_Q(10 ** 6)) is False
    assert M._BF16_SDPA["state"] == "unknown"                # never even resolved

    _fake_cuda(monkeypatch, (8, 0), free=2 ** 30)            # sm_80 disables
    assert M._use_fp16_attention(_Q(10 ** 6)) is False
    assert M._BF16_SDPA["state"] == "disabled"


def test_h3_attention_on_cpu_is_the_plain_call(monkeypatch, fresh_state):
    _fake_cuda(monkeypatch, (7, 0))
    called = []
    monkeypatch.setattr(M, "_sdpa_fp16", lambda *a: called.append(1))
    q, k, v = (torch.randn(1, 2, 8, 16) for _ in range(3))
    assert torch.equal(M.h3_attention(q, k, v), M.F.scaled_dot_product_attention(q, k, v))
    assert not called
    # and with the faked sm_70 + a tiny budget the fp16 route is taken
    monkeypatch.setattr(M, "_use_fp16_attention", lambda q: True)
    M.h3_attention(q, k, v)
    assert called


def test_is_pre_sm80_cuda(monkeypatch):
    _fake_cuda(monkeypatch, (7, 0))
    assert E.is_pre_sm80_cuda("cuda:0") and not E.is_pre_sm80_cuda("cpu")
    _fake_cuda(monkeypatch, (7, 0), hip="6.2")
    assert not E.is_pre_sm80_cuda("cuda:0")
    _fake_cuda(monkeypatch, (8, 0))
    assert not E.is_pre_sm80_cuda("cuda:0")
