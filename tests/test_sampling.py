import importlib

import pytest
import torch

from annadca.rbm.sampling import sample
from benchmarks.sampling import inputs


@pytest.mark.parametrize("kind", ["binary", "categorical"])
@pytest.mark.parametrize("conditioned", [False, True])
@pytest.mark.parametrize("beta", [0.0, 0.3, 1.0, 2.0])
def test_torch_equivalence(kind, conditioned, beta):
    torch.manual_seed(7)
    v, l, p = inputs(kind, 13, 7, 11, 5, 3, "cpu", torch.float64)
    baseline = importlib.import_module(f"benchmarks._baseline_{kind}")
    fn = baseline._sample_conditioned if conditioned else baseline._sample
    torch.manual_seed(9)
    expected = fn(3, v, l, p, beta)
    torch.manual_seed(9)
    actual = sample(3, v, l, p, beta, conditioned=conditioned)
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y)


@pytest.mark.parametrize("kind", ["binary", "categorical"])
def test_invalid_steps(kind):
    v, l, p = inputs(kind, 3, 7, 11, 5, 3, "cpu")
    for steps in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            sample(steps, v, l, p)
    with pytest.raises(ValueError, match="CUDA"):
        sample(1, v, l, p, backend="triton")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("beta", [0.0, 0.3, 1.0, 5.0])
def test_triton_probabilities_and_distribution(categorical, beta):
    from annadca.rbm.triton_sampling import draw

    torch.manual_seed(11)
    bias = torch.tensor([-0.7, 0.2, 1.2, -2.0, 0.5], device="cuda")
    logits = torch.zeros(50000, 5, device="cuda")
    torch.manual_seed(12)
    s, p = draw(logits, bias, beta, categorical)
    expected = (beta * bias).softmax(-1) if categorical else (beta * bias).sigmoid()
    torch.testing.assert_close(p, expected.expand_as(p), atol=2e-6, rtol=2e-6)
    assert torch.all((s == 0) | (s == 1))
    if categorical:
        torch.testing.assert_close(s.sum(-1), torch.ones_like(s[:, 0]))
    torch.testing.assert_close(s.mean(0), expected, atol=0.012, rtol=0)
    torch.manual_seed(12)
    torch.testing.assert_close(draw(logits, bias, beta, categorical)[0], s)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["binary", "categorical"])
@pytest.mark.parametrize("conditioned", [False, True])
def test_triton_chain(kind, conditioned):
    v, l, p = inputs(kind, 13, 7, 11, 5, 3, "cuda")
    original = (v.clone(), l.clone())
    out = sample(3, v, l, p, conditioned=conditioned, backend="triton")
    assert all(torch.isfinite(x).all() for x in out)
    assert out[0].shape == v.shape
    if kind == "categorical":
        torch.testing.assert_close(out[0].sum(-1), torch.ones_like(v[..., 0]))
    torch.testing.assert_close(v, original[0])
    torch.testing.assert_close(l, original[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["binary", "categorical"])
@pytest.mark.parametrize("name", ["hiddens", "visibles", "labels"])
def test_triton_conditional_probabilities(kind, name):
    from annadca.rbm.sampling import get_sampler

    v, l, p = inputs(kind, 13, 7, 11, 5, 3, "cuda")
    h = torch.rand(13, 11, device="cuda")
    ref = getattr(importlib.import_module(f"benchmarks._baseline_{kind}"), "_sample_" + name)
    args = (v, l, p, 0.7) if name == "hiddens" else (h, p, 0.7)
    torch.testing.assert_close(
        get_sampler(name, ref, "triton")(*args)[1], ref(*args)[1], atol=2e-6, rtol=2e-6
    )


@pytest.mark.parametrize("kind", ["binary", "categorical"])
def test_public_sample_skips_initialization(kind, monkeypatch):
    module = importlib.import_module(f"annadca.rbm.{kind}.classes")
    cls = getattr(module, "annaRBMbin" if kind == "binary" else "annaRBMcat")
    v, l, p = inputs(kind, 3, 7, 11, 5, 3, "cpu")
    model = cls(params=p)

    def unexpected(*args, **kwargs):
        raise AssertionError("Unnecessary chain initialization")

    monkeypatch.setattr(model, "init_chains", unexpected)
    assert model.sample(2, visible=v, label=l)["visible"].shape == v.shape
