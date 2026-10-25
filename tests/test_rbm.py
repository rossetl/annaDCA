import itertools
import math
import torch
import pytest
from annadca.rbm import AnnaRBM, get_rbm
from annadca.layers import BernoulliLayer, PottsLayer, GaussianLayer, ReLULayer, CategoricalLabelLayer
from annadca.utils.functions import mm_left

DTYPE = torch.float64


def _randomize(rbm: AnnaRBM, std: float = 0.5, seed: int = 0) -> AnnaRBM:
    """Draws random parameters for the model, keeping the scales of the continuous layers positive."""
    gen = torch.Generator().manual_seed(seed)
    for name, p in rbm.named_parameters():
        if name.endswith("scale"):
            p.copy_(0.5 + torch.rand(p.shape, generator=gen, dtype=p.dtype))
        else:
            p.copy_(std * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    return rbm


def _all_binary(n: int) -> torch.Tensor:
    return torch.tensor(list(itertools.product([0.0, 1.0], repeat=n)), dtype=DTYPE)


def _small_bernoulli_rbm(num_visibles=4, num_hiddens=3, num_classes=2) -> AnnaRBM:
    rbm = get_rbm("bernoulli", "bernoulli", num_visibles, num_hiddens, num_classes).to(dtype=DTYPE)
    return _randomize(rbm)


def test_marginal_energies_match_enumeration():
    rbm = _small_bernoulli_rbm()
    V, H, C = 4, 3, 2
    all_v, all_h, all_l = _all_binary(V), _all_binary(H), torch.eye(C, dtype=DTYPE)
    # E[v, h, l] on the full configuration space
    vv, hh, ll = torch.meshgrid(torch.arange(len(all_v)), torch.arange(len(all_h)), torch.arange(C), indexing="ij")
    E = rbm.compute_energy(all_v[vv.flatten()], all_h[hh.flatten()], all_l[ll.flatten()]).view(len(all_v), len(all_h), C)

    E_vl = - torch.logsumexp(-E, dim=1) # (num_v, C)
    v_rep, l_rep = all_v.repeat_interleave(C, 0), all_l.repeat(len(all_v), 1)
    assert torch.allclose(rbm.compute_energy_visibles_labels(v_rep, l_rep), E_vl.flatten())
    assert torch.allclose(rbm.compute_energy_visibles(all_v), - torch.logsumexp(-E_vl, dim=1))
    assert torch.allclose(rbm.predict_labels(all_v), torch.softmax(-E_vl, dim=1))
    E_h = - torch.logsumexp(-E.permute(1, 0, 2).reshape(len(all_h), -1), dim=1)
    assert torch.allclose(rbm.compute_energy_hiddens(all_h), E_h)


@pytest.mark.parametrize("layer_cls", [GaussianLayer, ReLULayer])
def test_continuous_nonlinearity_matches_integration(layer_cls):
    layer = layer_cls(3).to(dtype=DTYPE)
    layer.bias.copy_(torch.tensor([-1.0, 0.3, 2.0]))
    layer.scale.copy_(torch.tensor([0.7, 1.0, 2.5]))
    I = torch.tensor([[0.5, -2.0, 1.0]], dtype=DTYPE)
    x = torch.linspace(-30 if layer_cls is GaussianLayer else 0.0, 30, 200001, dtype=DTYPE).view(-1, 1)
    integrand = torch.exp(- 0.5 * layer.scale * x**2 + (layer.bias + I) * x)
    expected = torch.log(torch.trapezoid(integrand, x, dim=0))
    assert torch.allclose(layer.nonlinearity(I).view(-1), expected, atol=1e-6)


@pytest.mark.parametrize("layer", [BernoulliLayer(3), GaussianLayer(3), ReLULayer(3), PottsLayer((3, 4)), CategoricalLabelLayer(4)])
def test_sampling_matches_meanvar(layer):
    torch.manual_seed(0)
    layer = layer.to(dtype=DTYPE)
    layer.bias.copy_(torch.randn(layer.shape, dtype=DTYPE))
    I = torch.randn(layer.shape, dtype=DTYPE).expand((200000,) + layer.shape)
    samples = layer(I)
    mean, var = layer.meanvar(I[:1])
    assert torch.allclose(samples.mean(0), mean[0], atol=1e-2)
    assert torch.allclose(samples.var(0), var[0], atol=2e-2)


def test_init_params_from_data():
    torch.manual_seed(0)
    potts = PottsLayer((5, 3)).to(dtype=DTYPE)
    data = torch.nn.functional.one_hot(torch.randint(0, 3, (1000, 5)), 3).to(DTYPE)
    potts.init_params_from_data(data)
    assert torch.allclose(torch.softmax(potts.bias, -1), data.mean(0))

    label = CategoricalLabelLayer(4).to(dtype=DTYPE)
    labels = torch.nn.functional.one_hot(torch.randint(0, 4, (1000,)), 4).to(DTYPE)
    label.init_params_from_data(labels)
    assert torch.allclose(torch.softmax(label.bias, -1), labels.mean(0))

    bernoulli = BernoulliLayer(5).to(dtype=DTYPE)
    binary = torch.bernoulli(torch.full((1000, 5), 0.3, dtype=DTYPE))
    bernoulli.init_params_from_data(binary)
    assert torch.allclose(torch.sigmoid(bernoulli.bias), binary.mean(0))

    gaussian = GaussianLayer(2).to(dtype=DTYPE)
    x = torch.randn(1000, 2, dtype=DTYPE) * 2.0 + 1.0
    gaussian.init_params_from_data(x)
    mean, var = gaussian.meanvar(torch.zeros(1, 2, dtype=DTYPE))
    assert torch.allclose(mean[0], x.mean(0), atol=1e-4)
    assert torch.allclose(var[0], x.var(0, unbiased=False), atol=1e-4)


# Gradient tests: the gradient of the log-likelihood estimated by PCD is the gradient of the surrogate
#   S(θ) = - <F(v, l)>_data + <F(v, l)>_chains,
# where F(v, l) is the energy of (v, l) after marginalizing over the hidden units.

GRADIENT_CASES = [
    ("bernoulli", 4, "bernoulli", False),
    ("potts", (4, 3), "gaussian", False),
    ("potts", (4, 3), "relu", False),
    ("gaussian", 4, "relu", True),
    ("bernoulli", 4, "bernoulli", True),
]


def _gradient_setup(visible_type, visible_shape, hidden_type, continuous_label, seed=0):
    torch.manual_seed(seed)
    num_classes = 1 if continuous_label else 3
    rbm = get_rbm(visible_type, hidden_type, visible_shape, 5, num_classes, continuous_label=continuous_label).to(dtype=DTYPE)
    if isinstance(rbm.hidden_layer, ReLULayer):
        rbm.hidden_layer.learn_scale = True
    # Models with continuous visible units are normalizable only for weak couplings
    std = 0.1 if visible_type in ("gaussian", "relu") else 0.5
    _randomize(rbm, std=std, seed=seed)
    data_model = get_rbm(visible_type, "bernoulli", visible_shape, 5, num_classes, continuous_label=continuous_label).to(dtype=DTYPE)
    data = _randomize(data_model, std=2 * std, seed=seed + 1).sample(gibbs_steps=5, num_samples=64)
    assert all(torch.isfinite(x).all() for x in data.values())
    data["weight"] = torch.rand(64, 1, dtype=DTYPE)
    chains = rbm.sample(gibbs_steps=5, num_samples=48)
    return rbm, data, chains


class _FreeEnergy(torch.nn.Module):
    def __init__(self, rbm):
        super().__init__()
        self.rbm = rbm

    def forward(self, visible, label):
        return self.rbm.compute_energy_visibles_labels(visible, label)


def _surrogate(rbm, params, data, chains):
    """S(θ) evaluated with the parameters `params` (a dict name -> tensor)."""
    free_energy = _FreeEnergy(rbm)
    params = {f"rbm.{k}": v for k, v in params.items()}
    w = data["weight"].view(-1) / data["weight"].sum()
    F_data = torch.func.functional_call(free_energy, params, (data["visible"], data["label"]))
    F_chains = torch.func.functional_call(free_energy, params, (chains["visible"], chains["label"]))
    return - (w * F_data).sum() + F_chains.mean()


def _autograd_gradient(rbm, data, chains):
    params = {k: p.detach().clone().requires_grad_(True) for k, p in rbm.named_parameters()}
    with torch.enable_grad():
        S = _surrogate(rbm, params, data, chains)
        grads = torch.autograd.grad(S, list(params.values()))
    return dict(zip(params.keys(), grads))


@pytest.mark.parametrize("case", GRADIENT_CASES)
def test_plain_gradient_matches_autograd(case):
    rbm, data, chains = _gradient_setup(*case)
    expected = _autograd_gradient(rbm, data, chains)
    rbm.apply_gradient(data, chains, pseudo_count=0.0, standardize=False, centered=False)
    for name, p in rbm.named_parameters():
        assert torch.allclose(p.grad, expected[name], atol=1e-10), name


def _standardized_to_original(rbm, tilde):
    """Maps the parameters of the standardized model to the original ones (see AnnaRBM.apply_gradient)."""
    o_v, o_h, o_l = rbm.offset_v, rbm.offset_h, rbm.offset_l
    s_v, s_h, s_l = rbm.scale_v, rbm.scale_h, rbm.scale_l
    params = dict(tilde)
    W = tilde["weight_matrix"] / (s_v.unsqueeze(-1) * s_h)
    L = tilde["label_matrix"] / (s_l.unsqueeze(-1) * s_h)
    params["weight_matrix"], params["label_matrix"] = W, L
    params["visible_layer.bias"] = tilde["visible_layer.bias"] - W @ o_h
    params["label_layer.bias"] = tilde["label_layer.bias"] - L @ o_h
    params["hidden_layer.bias"] = tilde["hidden_layer.bias"] - mm_left(W, o_v) - o_l @ L
    return params


def _original_to_standardized(rbm, params):
    o_v, o_h, o_l = rbm.offset_v, rbm.offset_h, rbm.offset_l
    s_v, s_h, s_l = rbm.scale_v, rbm.scale_h, rbm.scale_l
    tilde = dict(params)
    W, L = params["weight_matrix"], params["label_matrix"]
    tilde["weight_matrix"] = W * (s_v.unsqueeze(-1) * s_h)
    tilde["label_matrix"] = L * (s_l.unsqueeze(-1) * s_h)
    tilde["visible_layer.bias"] = params["visible_layer.bias"] + W @ o_h
    tilde["label_layer.bias"] = params["label_layer.bias"] + L @ o_h
    tilde["hidden_layer.bias"] = params["hidden_layer.bias"] + mm_left(W, o_v) + o_l @ L
    return tilde


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("case", GRADIENT_CASES)
def test_standardized_gradient_is_gradient_of_reparametrized_model(case, standardize):
    """The update must be Δθ = J ∇_θ' S, with θ = θ(θ') the (linear) map from the standardized parameters θ'
    to the original ones and J its Jacobian."""
    rbm, data, chains = _gradient_setup(*case)
    rbm.apply_gradient(data, chains, pseudo_count=0.0, standardize=standardize, centered=True)
    names = [k for k, _ in rbm.named_parameters()]
    params = {k: p.detach().clone() for k, p in rbm.named_parameters()}
    tilde0 = _original_to_standardized(rbm, params)
    tilde = {k: v.clone().requires_grad_(True) for k, v in tilde0.items()}
    with torch.enable_grad():
        S = _surrogate(rbm, _standardized_to_original(rbm, tilde), data, chains)
        grad_tilde = torch.autograd.grad(S, [tilde[k] for k in names])
    _, expected = torch.func.jvp(
        lambda *t: tuple(_standardized_to_original(rbm, dict(zip(names, t)))[k] for k in names),
        tuple(tilde0[k] for k in names),
        tuple(grad_tilde),
    )
    for name, value in zip(names, expected):
        assert torch.allclose(dict(rbm.named_parameters())[name].grad, value, atol=1e-10), name


@pytest.mark.parametrize("visible_type, visible_shape", [("bernoulli", 6), ("potts", (6, 3))])
def test_l1l2_regularization_matches_autograd(visible_type, visible_shape):
    rbm = _randomize(get_rbm(visible_type, "bernoulli", visible_shape, 4, 2).to(dtype=DTYPE))
    rbm.weight_matrix.grad = torch.zeros_like(rbm.weight_matrix)
    rbm.regularize(l1l2_strength=0.3)
    W = rbm.weight_matrix.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        N = W[..., 0].numel()
        penalty = 0.3 / (2 * N) * (W.abs().reshape(N, -1).sum(0) ** 2).sum()
        expected = - torch.autograd.grad(penalty, W)[0]
    assert torch.allclose(rbm.weight_matrix.grad, expected)


def test_biased_model_identity():
    rbm = _small_bernoulli_rbm()
    biased = rbm.get_biased_model(gen_strength=1.0, label_strength=1.0, wt_strength=0.0, wt=torch.zeros(4))
    v = _all_binary(4)
    assert torch.allclose(biased.compute_energy_visibles(v), rbm.compute_energy_visibles(v))
    biased = rbm.get_biased_model(gen_strength=2.0, label_strength=0.0, wt_strength=1.0, wt=torch.ones(4))
    assert torch.allclose(biased.visible_layer.bias, 2.0 * rbm.visible_layer.bias + 1.0)
    assert torch.all(biased.label_matrix == 0)


def test_state_dict_roundtrip():
    rbm = _small_bernoulli_rbm()
    other = get_rbm("bernoulli", "bernoulli", 4, 3, 2).to(dtype=DTYPE)
    other.load_state_dict(rbm.state_dict())
    v = _all_binary(4)
    assert torch.allclose(other.compute_energy_visibles(v), rbm.compute_energy_visibles(v))
    # Offsets and scales are buffers: they are saved but not optimized
    assert "offset_v" in rbm.state_dict()
    assert all("offset" not in k and "scale_" not in k for k, _ in rbm.named_parameters())


def test_invalid_hidden_layer():
    with pytest.raises(AssertionError):
        get_rbm("bernoulli", "potts", 4, (3, 2), 2)
