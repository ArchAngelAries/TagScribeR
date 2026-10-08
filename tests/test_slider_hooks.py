"""Slider hooks of every family driver (Fizgig 7.0.1): training_loss's diff_ref / diff_weight image-pair weighting, and
noise_latents / predict for prompt-pair sliders. On each family's tiny random model (the family tests' own builders):

* diff_ref=None (or diff_weight 0) is the plain loss of the prediction against the family's target;
* an identical pair at diff_weight 1 degenerates to uniform weights (Fizgig's guard), and a pair that differs in one
  region weights that region more, matching a hand computation of Fizgig's formula for that family;
* noise_latents returns a dict with a float "t", and a prompt-slider step (Fizgig families/train.py
  _prompt_slider_step's target, v_neutral + guidance * (v_pos - v_neg)) backpropagates to the LoRA;
* predict at a noise_latents state is exactly the prediction training_loss makes for the same seed.
CPU only, no weights."""
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from training import registry  # noqa: E402
from training.lora import FamilyLoRA  # noqa: E402

SEED = 3


def _gen(seed=SEED):
    return torch.Generator().manual_seed(seed)


def _seeded(shape, seed=SEED):
    """The first draw of a seeded generator: every family here draws its noise first."""
    return torch.randn(shape, generator=_gen(seed))


def _token_d(diff):
    """|latents - ref| (B, C, h, w) averaged over channels, one value per latent cell -> (B, h*w)."""
    return diff.abs().float().mean(dim=1).flatten(1)


def _patch_d(diff, p=2):
    """... averaged over channels and each p x p patch -> (B, h/p * w/p), in patchify's row-major token order."""
    return F.avg_pool2d(diff.abs().float().mean(dim=1, keepdim=True), p).flatten(1)


def _fizgig_weights(d, dw):
    """Fizgig 7.0.1's weighting (krea2/driver.py loss_at), written out: r = d / mean d capped at 8, w = (1 - dw) +
    dw * r renormalised to mean 1, uniform for an identical pair."""
    out = torch.ones_like(d)
    for i in range(d.shape[0]):
        m = d[i].mean()
        if float(m) > 1e-6:
            w = (1.0 - dw) + dw * torch.clamp(d[i] / m, max=8.0)
            out[i] = w / w.mean()
    return out


# ---- the families --------------------------------------------------------------------------------------------------
class Case:
    """One family: a driver, its tiny model, latents (1, C, h, w) and conditioning; `target(lat, state)` the training
    target for the seeded draw, `se(pred, target)` the per-cell squared error and `d(lat, ref)` Fizgig's per-cell
    difference, both (B, cells); `region` a slice that covers whole patches."""
    snr = None

    def __init__(self, name, driver, dit, lat, cond, conds, target, se, d, window=True):
        self.name, self.driver, self.dit, self.lat, self.cond, self.conds = name, driver, dit, lat, cond, conds
        self.target, self.se, self.d, self.window = target, se, d, window


def _krea2():
    import tests.test_training_krea2 as K
    from training.families.krea2 import sampling as S
    d = K.Krea2Driver()
    d.description = registry.get("krea2")
    bf = torch.bfloat16
    tgt = lambda lat, st: S.patchify(_seeded(lat.shape).to(bf) - lat.to(bf), 2).float()   # noqa: E731
    return Case("krea2", d, K._dit(), torch.randn(1, 16, 8, 8, generator=_gen(11)), K._cond(1),
                [K._cond(1) for _ in range(3)], tgt,
                lambda p, t: (p.float() - t).pow(2).mean(dim=-1),
                lambda lat, ref: _patch_d(lat.to(bf) - ref.to(bf)))


def _qwen():
    import tests.test_training_smoke as Q
    from training.families.qwen_image21 import sampling as S
    dit, d = Q._tiny_qwen()

    def tgt(lat, st):
        x0 = S.pack(lat.float())
        return _seeded(x0.shape) - x0
    c = lambda: {"hidden_states": torch.randn(1, 5, 32)}                # noqa: E731
    return Case("qwen_image21", d, dit, torch.randn(1, 64, 4, 4, generator=_gen(11)), c(), [c() for _ in range(3)],
                tgt, lambda p, t: (p.float() - t).pow(2).mean(dim=-1), lambda lat, ref: _token_d(lat - ref))


def _klein():
    import tests.test_training_klein as KL
    d = KL.KleinDriver()
    d.description = registry.get("klein9b")
    return Case("klein", d, KL._dit(), torch.randn(1, 16, 4, 4, generator=_gen(11)), KL._cond(1),
                [KL._cond(1) for _ in range(3)], lambda lat, st: _seeded(lat.shape) - lat.float(),
                lambda p, t: (p.float() - t).pow(2).mean(dim=1).flatten(1), lambda lat, ref: _token_d(lat - ref))


def _sdxl(vpred=False):
    import tests.test_training_sdxl as SX
    from training.families.sdxl import sampling as S
    d = SX._driver(SX.TinyVPredDriver, min_snr_gamma=5.0, noise_offset=0.05) if vpred else SX._driver()
    c = Case("sdxl_vpred_snr_offset" if vpred else "sdxl", d, SX._tiny_unet(), torch.randn(1, 4, 8, 8, generator=_gen(11)),
             SX._cond(1), [SX._cond(1, seed=s) for s in (4, 5, 6)],
             lambda lat, st: S.target_for(d.prediction, st["x0"], st["noise"], st["ac"]),
             lambda p, t: (p.float() - t).pow(2).mean(dim=1).flatten(1), lambda lat, ref: _token_d(lat - ref))
    if vpred:
        c.snr = lambda st: S.min_snr_weights(st["ac"], 5.0, d.prediction)
    return c


def _anima():
    import tests.test_training_anima as A
    d = A.AnimaDriver()
    d.description = registry.get("anima")
    return Case("anima", d, A._dit(), torch.randn(1, 16, 8, 8, generator=_gen(11)), A._cond(1),
                [A._cond(1) for _ in range(3)], lambda lat, st: _seeded(lat.shape) - lat.float(),
                lambda p, t: (p.float() - t).pow(2).mean(dim=1).flatten(1), lambda lat, ref: _token_d(lat - ref))


def _minimax():
    import tests.test_training_minimax as M
    d = M.MiniMaxH3Driver()
    d.description = registry.get("minimax_h3")
    d.compute_dtype = torch.float32
    d.configure(likeness_mode=M.OFF, blocks="all")

    def d_cells(lat, ref):                       # one value per 2 x 2 patch token, spread back over its 4 cells
        tok = F.avg_pool2d((lat - ref).abs().float().mean(dim=1, keepdim=True), 2)
        return F.interpolate(tok, scale_factor=2, mode="nearest").flatten(1)
    return Case("minimax_h3", d, M._dit(), torch.randn(1, 24, 8, 8, generator=_gen(11)), M._cond(),
                [M._cond() for _ in range(3)], lambda lat, st: (st["x0"] - st["noise"]).float(),
                lambda p, t: (p.float() - t).pow(2).mean(dim=1).flatten(1), d_cells, window=False)


CASES = {"krea2": _krea2, "qwen_image21": _qwen, "klein": _klein, "sdxl": _sdxl,
         "sdxl_vpred_snr_offset": lambda: _sdxl(vpred=True), "anima": _anima, "minimax_h3": _minimax}


@pytest.fixture(params=list(CASES))
def case(request):
    torch.manual_seed(0)
    return CASES[request.param]()


def _raw(out):
    return out.sample if hasattr(out, "sample") else out


def _state_pred(case, seed=SEED):
    st = case.driver.noise_latents(case.lat, _gen(seed))
    with torch.no_grad():
        return st, case.driver.predict(case.dit, st, case.cond)


def _expected(case, ref, dw, seed=SEED):
    """Fizgig's weighted loss by hand from predict's output, the seeded target and the written-out weights."""
    st, pred = _state_pred(case, seed)
    se = case.se(pred, case.target(case.lat, st))
    w = _fizgig_weights(case.d(case.lat, ref), dw) if ref is not None and dw > 0 else torch.ones_like(se)
    loss = (se * w).mean(dim=1)
    if case.snr is not None:
        loss = loss * case.snr(st)
    return loss.mean(), w


def _loss(case, **kw):
    with torch.no_grad():
        return case.driver.training_loss(case.dit, case.lat, case.cond, _gen(), **kw)


# ---- (d) predict = training_loss's prediction ----------------------------------------------------------------------
def test_predict_at_a_noise_latents_state_is_the_training_prediction(case):
    seen = []
    hook = case.dit.register_forward_hook(lambda m, i, o: seen.append(_raw(o).detach().clone()))
    try:
        _loss_, info = _loss(case)
        st, _pred = _state_pred(case)
    finally:
        hook.remove()
    assert len(seen) == 2 and torch.equal(seen[0], seen[1])
    assert isinstance(st["t"], float) and st["t"] == info["t"]


# ---- (a) no pair: the plain loss ------------------------------------------------------------------------------------
def test_without_a_pair_the_loss_is_the_plain_objective(case):
    loss, _ = _loss(case)
    want, _ = _expected(case, None, 0.0)
    assert torch.allclose(loss, want, rtol=1e-5, atol=1e-7)
    ref = case.lat + torch.randn(case.lat.shape, generator=_gen(9))
    off, _ = _loss(case, diff_ref=ref, diff_weight=0.0)                  # weight 0: the plain path, bit for bit
    assert torch.equal(off, loss)


# ---- (b) Fizgig's weighting -----------------------------------------------------------------------------------------
def test_identical_pair_gives_uniform_weights(case):
    loss, _ = _loss(case)
    same, _ = _loss(case, diff_ref=case.lat.clone(), diff_weight=1.0)
    assert torch.isfinite(same) and torch.allclose(same, loss, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("dw", [1.0, 0.5])
def test_a_pair_differing_in_one_region_weights_it_more(case, dw):
    # small differences everywhere, a large uneven one in the top-left 2 x 2 patch (so the cap at 8 and the per-patch
    # mean both matter)
    ref = case.lat + 0.05 * torch.randn(case.lat.shape, generator=_gen(21))
    ref[..., :2, :2] += 1.0 + torch.rand(case.lat[..., :2, :2].shape, generator=_gen(22))
    ref[..., 0, 0] += 30.0                                               # one spike, past the cap
    loss, info = _loss(case, diff_ref=ref, diff_weight=dw)
    want, w = _expected(case, ref, dw)
    assert torch.allclose(loss, want, rtol=1e-5, atol=1e-7), (loss, want)
    assert 0.0 <= info["t"] <= 1.0
    # the region's cells carry the weight
    mark = torch.zeros_like(case.lat)
    mark[..., :2, :2] = 1.0
    region = case.d(mark, torch.zeros_like(mark))[0] > 0.5
    assert region.any() and not region.all()
    d = case.d(case.lat, ref)[0]
    assert float(d.max() / d.mean()) > 8.0                                # the cap is in play
    assert float(w[0][region].min()) > float(w[0][~region].max())
    assert torch.allclose(w.mean(dim=1), torch.ones(1), atol=1e-5)
    if dw == 1.0:
        plain, _ = _loss(case)
        assert not torch.allclose(loss, plain)


def test_fizgig_weight_formula_by_hand():
    """The written-out formula itself: a 4-cell pair differing in one cell by 1 -> d = [1, 0, 0, 0], mean 0.25, r = [4,
    0, 0, 0]; at weight 1, w = r / mean r = [4, 0, 0, 0]; at weight 0.5, w = [2.5, .5, .5, .5] / 1 = same."""
    d = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(_fizgig_weights(d, 1.0), torch.tensor([[4.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(_fizgig_weights(d, 0.5), torch.tensor([[2.5, 0.5, 0.5, 0.5]]))
    big = torch.zeros(1, 20)
    big[0, 0] = 1.0                                                      # r = 20 caps at 8
    w = _fizgig_weights(big, 1.0)
    assert torch.allclose(w[0, 0], torch.tensor(8.0 / (8.0 / 20)))
    assert torch.equal(_fizgig_weights(torch.zeros(1, 4), 1.0), torch.ones(1, 4))


# ---- (c) noise_latents + a prompt-slider step ------------------------------------------------------------------------
def test_noise_latents_state_and_the_timestep_window(case):
    st = case.driver.noise_latents(case.lat, _gen(5))
    assert isinstance(st, dict) and isinstance(st["t"], float) and 0.0 <= st["t"] <= 1.0
    if case.window:
        for seed in range(6):
            t = case.driver.noise_latents(case.lat, _gen(seed), min_t=0.2, max_t=0.4)["t"]
            assert 0.2 - 1e-3 <= t <= 0.4 + 1e-3


def test_a_prompt_slider_step_backpropagates_to_the_lora(case):
    """Fizgig families/train.py _prompt_slider_step's arithmetic: the frozen predictions for the neutral, positive and
    negative prompts at one practice latent, then the trained prediction toward v_n + guidance * (v_pos - v_neg)."""
    drv, dit = case.driver, case.dit
    net = FamilyLoRA(dit, drv)
    net.add_trainable(4, 4)
    params = [p for p in net.parameters() if p.requires_grad]
    assert params
    st = drv.noise_latents(case.lat, _gen(7))
    n_c, p_c, g_c = case.conds
    with torch.no_grad():
        v_n = drv.predict(dit, st, n_c).float()
        delta = 3.0 * (drv.predict(dit, st, p_c).float() - drv.predict(dit, st, g_c).float())
    loss = F.mse_loss(drv.predict(dit, st, n_c).float(), v_n + delta)
    loss.backward()
    grads = [p.grad for p in params if p.grad is not None]
    assert torch.isfinite(loss) and float(loss.detach()) > 0
    assert grads and all(torch.isfinite(g).all() for g in grads) and any(float(g.abs().sum()) > 0 for g in grads)
    # the target's shape is the training target's (the space mse(predict, target) lives in)
    assert v_n.shape == case.target(case.lat, st).shape
