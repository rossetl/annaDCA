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

    L, q, P = params["weight_matrix"].shape
    weight_matrix_oh = params["weight_matrix"].view(L * q, P)
    visible_oh = visible.view(-1, L * q)
    mh = torch.sigmoid(
        beta * (params["hbias"] + (visible_oh @ weight_matrix_oh) + (label @ params["label_matrix"]))
    )
    h = torch.bernoulli(mh).to(params["weight_matrix"].dtype)
    return (h, mh)


@torch.jit.script
def _sample_visibles(
    hidden: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:

    num_visibles, num_states, _ = params["weight_matrix"].shape
    mv = torch.softmax(
        beta * (params["vbias"] + torch.einsum("np,lqp->nlq", hidden, params["weight_matrix"])),
        dim=-1
    )
    v = one_hot(
        torch.multinomial(mv.view(-1, num_states), 1).view(-1, num_visibles),
        num_classes=num_states,
    ).to(params["weight_matrix"].dtype)
    return (v, mv)


@torch.jit.script
def _sample_labels_sigmoid(
    hidden: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:

    ml = torch.sigmoid(
        beta * (params["lbias"] + (hidden @ params["label_matrix"].T))
    )
    l = torch.bernoulli(ml)
    return (l, ml)


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
        visible, visible_mag = _sample_visibles(hidden=hidden, params=params, beta=beta)
    
    return visible_mag


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
    if visible.dim() == 2:
        visible = visible.unsqueeze(0)
    num_visibles, num_states, num_hiddens = params["weight_matrix"].shape
    num_labels = params["label_matrix"].shape[0]
    
    v_flat = visible.view(visible.shape[0], -1)
    w_oh = params["weight_matrix"].view(num_visibles * num_states, num_hiddens)
    exponent = (
        params["hbias"].view(1, 1, num_hiddens) +
        (v_flat @ w_oh).view(-1, 1, num_hiddens) +
        params["label_matrix"].view(1, num_labels, num_hiddens)
    )
    logits = params["lbias"] + torch.where(exponent > 10, exponent, torch.log1p(torch.exp(exponent))).sum(-1) # (num_samples, num_labels)
    
    return torch.softmax(beta * logits, dim=-1)