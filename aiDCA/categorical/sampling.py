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
    """Sample the hidden layer conditionally to the visible one and the labels.

    Args:
        visible (torch.Tensor): Visible units.
        label (torch.Tensor): Labels.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        beta (float, optional): Inverse temperature. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sampled hidden units and hidden magnetization.
    """
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
    """Sample the visible layer conditionally to the hidden one.

    Args:
        hidden (torch.Tensor): Hidden units.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        beta (float, optional): Inverse temperature. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sampled visible units and visible magnetization.
    """
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
def _sample_labels(
    hidden: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample the labels conditionally to the hidden layer.

    Args:
        hidden (torch.Tensor): Hidden units.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        beta (float, optional): Inverse temperature. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sampled labels and label magnetization.
    """
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
    """Samples from the binary aiRBM.

    Args:
        gibbs_steps (int): Number of Gibbs steps.
        visible (torch.Tensor): Visible units initial configuration.
        label (torch.Tensor): Labels initial configuration.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        beta (float, optional): Inverse temperature. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Sampled visible units, hidden units and labels.
    """
    for _ in range(gibbs_steps):
        hidden, _ = _sample_hiddens(visible=visible, label=label, params=params, beta=beta)
        label, _ = _sample_labels(hidden=hidden, params=params, beta=beta)
        visible, _ = _sample_visibles(hidden=hidden, params=params, beta=beta)
    
    return (visible, hidden, label)