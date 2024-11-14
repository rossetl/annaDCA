from typing import Dict, Tuple
import torch


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

    ml = torch.sigmoid(
        beta * (params["lbias"] + (hidden @ params["label_matrix"].T))
    )
    l = torch.bernoulli(ml)
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


def _predict_labels(
    gibbs_steps: int,
    visible: torch.Tensor,
    label: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,    
) -> torch.Tensor:
    
    for _ in range(gibbs_steps):
        hidden, _ = _sample_hiddens(visible=visible, label=label, params=params, beta=beta)
        label, label_mag = _sample_labels(hidden=hidden, params=params, beta=beta)
    
    return label_mag