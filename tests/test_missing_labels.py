"""Compare marginalized updates with explicitly enumerated latent states."""

import importlib
import itertools

import pytest
import torch

from annadca.rbm.missing_labels import marginalize_missing_labels
from annadca.scripts.train import create_parser


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("missing", ["none", "mixed", "all"])
@pytest.mark.parametrize("pseudo_count", [0.0, 0.1])
def test_enumerated_positive_phase(binary, centered, missing, pseudo_count):
    torch.manual_seed(31)
    dtype = torch.float64
    params = {
        "weight_matrix": torch.randn((2, 2) if binary else (2, 3, 2), dtype=dtype),
        "label_matrix": torch.randn(3, 2, dtype=dtype) * 2,
        "hbias": torch.randn(2, dtype=dtype),
        "vbias": torch.randn((2,) if binary else (2, 3), dtype=dtype),
        "lbias": torch.randn(3, dtype=dtype),
    }
    for p in params.values():
        p.grad = torch.zeros_like(p)
    visible = torch.tensor([[0, 1], [1, 0], [1, 1]], dtype=dtype)
    if not binary:
        visible = torch.nn.functional.one_hot(visible.long(), 3).to(dtype)
    labels = torch.eye(3, dtype=dtype)
    if missing == "mixed":
        labels[1] = 0
    elif missing == "all":
        labels.zero_()
    data = {
        "visible": visible,
        "label": labels,
        "weight": torch.tensor([[1.0], [3.0], [2.0]], dtype=dtype),
    }
    original_labels = labels.clone()
    actual = marginalize_missing_labels(data, params)
    torch.testing.assert_close(data["label"], original_labels)
    assert "hidden" not in data

    # Independent reference: enumerate label AND hidden states using joint scores.
    xs = torch.tensor(list(itertools.product([0.0, 1.0], repeat=2)), dtype=dtype)
    expanded = {k: [] for k in ("visible", "label", "hidden", "weight")}
    for n in range(3):
        ks = range(3) if labels[n].sum() == 0 else [n]
        states = [(k, x) for k in ks for x in xs]
        scores = torch.stack(
            [
                params["lbias"][k]
                + x
                @ (
                    params["hbias"]
                    + visible[n].reshape(-1) @ params["weight_matrix"].reshape(-1, 2)
                    + params["label_matrix"][k]
                )
                for k, x in states
            ]
        )
        probabilities = scores.softmax(0)
        for probability, (k, x) in zip(probabilities, states):
            expanded["visible"].append(visible[n])
            expanded["label"].append(torch.eye(3, dtype=dtype)[k])
            expanded["hidden"].append(x)
            expanded["weight"].append(data["weight"][n] * probability)
    reference = {k: torch.stack(v) for k, v in expanded.items()}
    weights = reference["weight"] / reference["weight"].sum()
    expected_joint = (reference["label"] * weights).T @ reference["hidden"]
    actual_weights = data["weight"] / data["weight"].sum()
    actual_joint = (actual["label"] * actual_weights).T @ actual["hidden"] + actual[
        "label_hidden_correction"
    ]
    torch.testing.assert_close(actual_joint, expected_joint)

    chains = {
        "visible": visible.flip(0),
        "label": torch.eye(3, dtype=dtype).flip(0),
        "hidden": xs[:3],
    }
    kind = "binary" if binary else "categorical"
    gradient = importlib.import_module(f"annadca.rbm.{kind}.grad")._compute_gradient
    options = dict(
        centered=centered, pseudo_count=pseudo_count, eta=0.7, lambda_l1=0.03, lambda_l2=0.02
    )
    gradient(actual, chains, params, **options)
    actual_grad = {k: p.grad.clone() for k, p in params.items()}
    gradient(reference, chains, params, **options)
    for k, p in params.items():
        torch.testing.assert_close(actual_grad[k], p.grad, msg=lambda msg: f"{k}: {msg}")


def test_flag_default_and_opt_in():
    parser = create_parser()
    assert not parser.parse_args(["-d", "data.csv"]).marginalize_missing_labels
    assert parser.parse_args(
        ["-d", "data.csv", "--marginalize_missing_labels"]
    ).marginalize_missing_labels


def test_large_fields_are_finite():
    params = {
        "weight_matrix": torch.tensor([[1000.0, -1000.0]]),
        "hbias": torch.zeros(2),
        "label_matrix": torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]]),
        "lbias": torch.zeros(2),
    }
    data = {"visible": torch.ones(1, 1), "label": torch.zeros(1, 2), "weight": torch.ones(1, 1)}
    result = marginalize_missing_labels(data, params)
    for tensor in result.values():
        assert torch.isfinite(tensor).all()
    torch.testing.assert_close(result["label"].sum(-1), torch.ones(1))
