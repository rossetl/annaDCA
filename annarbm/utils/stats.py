import torch
from typing import Optional, Tuple


def _normalized_weights(
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Returns the sample weights normalized to one and reshaped to broadcast against `x`.

    Args:
        x (torch.Tensor): Input tensor of shape (M, ...).
        weights (Optional[torch.Tensor], optional): Weights of the samples of shape (M,) or (M, 1). If None, uniform weights are used.

    Returns:
        torch.Tensor: Normalized weights of shape (M, 1, ..., 1).
    """
    M = x.shape[0]
    if weights is None:
        weights = torch.ones(M, device=x.device, dtype=x.dtype)
    weights = weights.reshape(M).to(device=x.device, dtype=x.dtype)
    weights = weights / weights.sum()
    return weights.view((M,) + (1,) * (x.dim() - 1))


def get_mean(
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes the (weighted) average of the input tensor along the first dimension.

    Args:
        x (torch.Tensor): Input tensor of shape (M, ...).
        weights (Optional[torch.Tensor], optional): Weights of the samples. If None, uniform weights are used.

    Returns:
        torch.Tensor: Mean tensor of shape x.shape[1:].
    """
    return (x * _normalized_weights(x, weights)).sum(dim=0)


def get_meanvar(
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the (weighted) mean and variance of the input tensor along the first dimension.

    Args:
        x (torch.Tensor): Input tensor of shape (M, ...).
        weights (Optional[torch.Tensor], optional): Weights of the samples. If None, uniform weights are used.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Mean and variance tensors of shape x.shape[1:].
    """
    w = _normalized_weights(x, weights)
    mean = (x * w).sum(dim=0)
    var = ((x - mean) ** 2 * w).sum(dim=0)
    return mean, var


def get_freq(
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    pseudo_count: float = 0.0,
    num_states: Optional[int] = None,
) -> torch.Tensor:
    """Computes the (weighted) frequencies of discrete variables, regularized with a pseudo count.
    To be used only for binary (0/1) or one-hot encoded variables.

    Args:
        x (torch.Tensor): Input tensor of shape (M, L) for binary variables or (M, L, q) for one-hot variables.
        weights (Optional[torch.Tensor], optional): Weights of the samples. If None, uniform weights are used.
        pseudo_count (float, optional): Pseudo count to be added to the frequencies. Defaults to 0.0.
        num_states (Optional[int], optional): Number of states of each variable. If None, it is 2 for (M, L) tensors and q for (M, L, q) tensors.

    Returns:
        torch.Tensor: Frequencies of shape x.shape[1:].
    """
    if x.dim() not in (2, 3):
        raise ValueError(f"Expected data to be a 2D or 3D tensor, but got {x.dim()}D tensor instead")
    if num_states is None:
        num_states = 2 if x.dim() == 2 else x.shape[-1]
    freq = get_mean(x, weights)
    torch.clamp_(freq, min=0.0)  # Set to zero the negative frequencies. Used for the reintegration.
    return (1.0 - pseudo_count) * freq + (pseudo_count / num_states)
