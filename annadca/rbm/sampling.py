"""Shared Gibbs driver with optional fused Triton sampling."""

import torch


def draw(logits, bias, beta, categorical, backend):
    if backend == "triton":
        from annadca.rbm.triton_sampling import draw as triton_draw

        return triton_draw(logits, bias, beta, categorical)
    return _torch_draw(logits, bias, beta, categorical)


@torch.jit.script
def _torch_draw(logits: torch.Tensor, bias: torch.Tensor, beta: float, categorical: bool):
    probs = (
        (beta * (logits + bias)).softmax(-1)
        if categorical
        else torch.sigmoid(beta * (logits + bias))
    )
    if categorical:
        indices = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1)
        samples = torch.zeros_like(probs).reshape(-1, probs.shape[-1])
        samples.scatter_(1, indices, 1)
        return samples.reshape_as(probs), probs
    return torch.bernoulli(probs), probs


def sample(gibbs_steps, visible, label, params, beta=1.0, *, conditioned=False, backend="torch"):
    if backend not in ("torch", "triton"):
        raise ValueError("backend must be 'torch' or 'triton'")
    if gibbs_steps < 1:
        raise ValueError("gibbs_steps must be positive")
    if backend == "triton" and (visible.device.type != "cuda" or visible.dtype != torch.float32):
        raise ValueError("Triton sampling requires float32 CUDA tensors")
    w = params["weight_matrix"]
    categorical = w.ndim == 3
    flat_w = w.reshape(-1, w.shape[-1])
    # The label field is invariant throughout a conditioned chain.
    label_field = label @ params["label_matrix"] if conditioned else None
    for _ in range(gibbs_steps):
        field = label_field if conditioned else label @ params["label_matrix"]
        hidden, hm = draw(
            visible.reshape(visible.shape[0], -1) @ flat_w + field,
            params["hbias"],
            beta,
            False,
            backend,
        )
        if not conditioned:
            label, lm = draw(
                hidden @ params["label_matrix"].T, params["lbias"], beta, True, backend
            )
        if categorical and backend == "torch":
            # Keep the scripted categorical projection/softmax together: splitting
            # this large operation regresses memory-heavy workloads.
            from annadca.rbm.categorical.sampling import _sample_visibles

            visible, vm = _sample_visibles(hidden, params, beta)
        else:
            logits = hidden @ flat_w.T
            if categorical:
                logits = logits.reshape(hidden.shape[0], w.shape[0], w.shape[1])
            visible, vm = draw(logits, params["vbias"], beta, categorical, backend)
            del logits
    if conditioned:
        return visible, hidden, vm, hm
    return visible, hidden, label, vm, hm, lm


def sample_hiddens(visible, label, params, beta=1.0):
    w = params["weight_matrix"].reshape(-1, params["hbias"].numel())
    return draw(
        visible.reshape(visible.shape[0], -1) @ w + label @ params["label_matrix"],
        params["hbias"],
        beta,
        False,
        "triton",
    )


def sample_visibles(hidden, params, beta=1.0):
    w = params["weight_matrix"]
    logits = hidden @ w.reshape(-1, w.shape[-1]).T
    if w.ndim == 3:
        logits = logits.reshape(hidden.shape[0], w.shape[0], w.shape[1])
    return draw(logits, params["vbias"], beta, w.ndim == 3, "triton")


def sample_labels(hidden, params, beta=1.0):
    return draw(hidden @ params["label_matrix"].T, params["lbias"], beta, True, "triton")


def get_sampler(name, baseline, backend):
    if backend == "torch":
        return baseline
    if backend != "triton":
        raise ValueError("backend must be 'torch' or 'triton'")
    return {"hiddens": sample_hiddens, "visibles": sample_visibles, "labels": sample_labels}[name]
