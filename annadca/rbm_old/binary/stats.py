import torch

@torch.jit.script
def _get_freq_single_point(
    data: torch.Tensor,
    weights: torch.Tensor | None,
    pseudo_count: float = 0.0,
) -> torch.Tensor:    
    M = len(data)
    if weights is not None:
        norm_weights = weights.reshape(M, 1) / weights.sum()
    else:
        norm_weights = torch.ones((M, 1), device=data.device) / M

    frequencies = (data * norm_weights).sum(dim=0)

    return (1. - pseudo_count) * frequencies + (pseudo_count / 2)


def get_freq_single_point(
    data: torch.Tensor,
    weights: torch.Tensor | None,
    pseudo_count: float = 0.,
) -> torch.Tensor:
    """Computes the single point frequencies of the input binary data.
    Args:
        data (torch.Tensor): Binary data array.
        weights (torch.Tensor | None): Weights of the sequences.
        pseudo_count (float, optional): Pseudo count to be added to the frequencies. Defaults to 0.0.
    
    Raises:
        ValueError: If the input data is not a 2D tensor.

    Returns:
        torch.Tensor: Single point frequencies.
    """
    if data.dim() != 2:
        raise ValueError(f"Expected data to be a 2D tensor, but got {data.dim()}D tensor instead")
    
    return _get_freq_single_point(data, weights, pseudo_count)


@torch.jit.script
def _get_freq_two_points(
    data: torch.Tensor,
    weights: torch.Tensor | None,
    pseudo_count: float=0.0,
) -> torch.Tensor:
    
    M, L = data.shape
    
    if weights is not None:
        norm_weights = weights.reshape(M, 1) / weights.sum()
    else:
        norm_weights = torch.ones((M, 1), device=data.device) / M
    
    fij = (data * norm_weights).T @ data
    # Apply the pseudo count
    fij = (1. - pseudo_count) * fij + (pseudo_count / 4)
    # Diagonal terms must represent the single point frequencies
    fi = get_freq_single_point(data, weights, pseudo_count).ravel()
    # Apply the pseudo count on the single point frequencies
    fij_diag = (1. - pseudo_count) * fi + (pseudo_count / 2)
    # Set the diagonal terms of fij to the single point frequencies
    fij = torch.diagonal_scatter(fij, fij_diag, dim1=0, dim2=1)
    
    return fij


def get_freq_two_points(
    data: torch.Tensor,
    weights: torch.Tensor | None,
    pseudo_count: float=0.0,
) -> torch.Tensor:
    """
    Computes the 2-points statistics of the input binary matrix

    Args:
        data (torch.Tensor): Binary data array.
        weights (torch.Tensor | None): Array of weights to assign to the sequences of shape.
        pseudo_count (float, optional): Pseudo count for the single and two points statistics. Acts as a regularization. Defaults to 0.0.
    
    Raises:
        ValueError: If the input data is not a 2D tensor.

    Returns:
        torch.Tensor: Matrix of two-point frequencies.
    """
    if data.dim() != 2:
        raise ValueError(f"Expected data to be a 2D tensor, but got {data.dim()}D tensor instead")
    
    return _get_freq_two_points(data, weights, pseudo_count)