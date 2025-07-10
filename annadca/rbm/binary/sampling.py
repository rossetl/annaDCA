from typing import Dict, Tuple
import torch
from adabmDCA.functional import one_hot


@torch.jit.script
def _sample_hiddens(
    visible: torch.Tensor,
    label: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:

    mh = torch.sigmoid(
        beta * (params["hbias"] + (visible @ params["weight_matrix"]) + (label @ params["label_matrix"]))
    )
    h = torch.bernoulli(mh)
    return (h, mh)


@torch.jit.script
def _sample_visibles(
    hidden: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:

    mv = torch.sigmoid(
        beta * (params["vbias"] + (hidden @ params["weight_matrix"].T))
    )
    v = torch.bernoulli(mv)
    return (v, mv)


@torch.jit.script
def _sample_labels(
    hidden: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:

    ml = torch.softmax(
        beta * (params["lbias"] + (hidden @ params["label_matrix"].T)),
        dim=-1,
    )
    l = one_hot(torch.multinomial(ml, 1), num_classes=ml.shape[-1]).to(params["weight_matrix"].dtype).view(-1, ml.shape[-1])
    
    return (l, ml)


def _sample(
    gibbs_steps: int,
    visible: torch.Tensor,
    label: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    for _ in range(gibbs_steps):
        hidden, _ = _sample_hiddens(visible=visible, label=label, params=params, beta=beta)
        label, _ = _sample_labels(hidden=hidden, params=params, beta=beta)
        visible, _ = _sample_visibles(hidden=hidden, params=params, beta=beta)
    
    return (visible, hidden, label)


def _sample_conditioned(
    gibbs_steps: int,
    visible: torch.Tensor,
    label: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
) -> torch.Tensor:
    
    for _ in range(gibbs_steps):
        hidden, _ = _sample_hiddens(visible=visible, label=label, params=params, beta=beta)
        visible = _sample_visibles(hidden=hidden, params=params, beta=beta)
    
    return (visible, hidden)


@torch.jit.script
def _predict_labels(
    visible: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,    
) -> torch.Tensor:
    """Returns the probability of each label given the visible units: p(l|v).

    Args:
        visible (torch.Tensor): Visible units.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        beta (float, optional): Inverse temperature. Defaults to 1.0.

    Returns:
        torch.Tensor: p(l|v).
    """
    if visible.dim() == 1:
        visible = visible.unsqueeze(0)
    num_hiddens = params["hbias"].shape[0]
    num_labels = params["lbias"].shape[0]
    
    exponent = (
        params["hbias"].view(1, 1, num_hiddens) +
        (visible @ params["weight_matrix"]).view(-1, 1, num_hiddens) +
        params["label_matrix"].view(1, num_labels, num_hiddens)
    )
    logits = params["lbias"] + torch.where(exponent > 10, exponent, torch.log1p(torch.exp(exponent))).sum(-1) # (num_samples, num_labels)
    
    return torch.softmax(beta * logits, dim=-1)