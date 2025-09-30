import torch
from typing import Optional


def get_mean(
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    pseudo_count: float = 0.0,
) -> torch.Tensor:
    """
    Compute the mean of the input tensor, optionally weighted by the given weights.

    Args:
        x (torch.Tensor): The input tensor.
        weights (Optional[torch.Tensor], optional): The weights for each element in the input tensor. Defaults to None.
        pseudo_count (float, optional): A small value added to the mean to avoid zero counts. Defaults to 0.0.

    Returns:
        torch.Tensor: The computed mean.
    """
    if len(x.shape) == 2:
        M, q = len(x), 2
        if weights is not None:
            weights = weights.reshape(M, 1) / weights.sum()
        else:
            weights = torch.ones((M, 1), device=x.device) / M
    elif len(x.shape) == 3:
        M, _, q = x.shape
        if weights is not None:
            weights = weights.reshape(M, 1, 1) / weights.sum()
        else:
            weights = torch.ones((M, 1, 1), device=x.device, dtype=x.dtype) / M
    else:
        raise ValueError(f"Expected data to be a 2D or 3D tensor, but got {x.dim()}D tensor instead")

    mean = (x * weights).sum(dim=0)
    torch.clamp_(mean, min=0.0)  # Set to zero the negative frequencies. Used for the reintegration.
    return (1. - pseudo_count) * mean + (pseudo_count / q)


def get_meanvar(
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    pseudo_count: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the mean and the variance of the input tensor, optionally weighted by the given weights.

    Args:
        x (torch.Tensor): The input tensor.
        weights (Optional[torch.Tensor], optional): The weights for each element in the input tensor. Defaults to None.
        pseudo_count (float, optional): A small value added to the variance to avoid zero counts. Defaults to 0.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The computed mean and variance.
    """
    if len(x.shape) == 2:
        M, q = len(x), 2
        if weights is not None:
            weights = weights.reshape(M, 1) / weights.sum()
        else:
            weights = torch.ones((M, 1), device=x.device) / M
    elif len(x.shape) == 3:
        M, _, q = x.shape
        if weights is not None:
            weights = weights.reshape(M, 1, 1) / weights.sum()
        else:
            weights = torch.ones((M, 1, 1), device=x.device, dtype=x.dtype) / M
    else:
        raise ValueError(f"Expected data to be a 2D or 3D tensor, but got {x.dim()}D tensor instead")

    mean = get_mean(x, weights=weights, pseudo_count=pseudo_count)
    var = ((x - mean) ** 2 * weights).sum(dim=0)
    var = (1. - pseudo_count) * var + (pseudo_count * (q - 1) / (q ** 2))
    return mean, var