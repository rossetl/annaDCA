import torch
from typing import Optional, Tuple, Dict
import math


def get_freq_single_point(
    data: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    pseudo_count: float = 0.0,
) -> torch.Tensor:
    """Compute the frequency of a single point in the data. Works for binary data.

    Args:
        data (torch.Tensor): The input data tensor.
        weights (Optional[torch.Tensor], optional): The weights for each data point. Defaults to None.
        pseudo_count (float, optional): A small value added to the frequencies to avoid zero counts. Defaults to 0.0.

    Returns:
        torch.Tensor: The computed frequencies for the single point.
    """
    M = len(data)
    if weights is not None:
        norm_weights = weights.reshape(M, 1) / weights.sum()
    else:
        norm_weights = torch.ones((M, 1), device=data.device) / M
    frequencies = (data * norm_weights).sum(dim=0)
    return (1. - pseudo_count) * frequencies + (pseudo_count / 2)


def get_freq_two_points(
    data: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    pseudo_count: float = 0.0,
) -> torch.Tensor:
    """Compute the frequency of two points in the data. Works for binary data.

    Args:
        data (torch.Tensor): The input data tensor.
        weights (Optional[torch.Tensor], optional): The weights for each data point. Defaults to None.
        pseudo_count (float, optional): A small value added to the frequencies to avoid zero counts. Defaults to 0.0.

    Returns:
        torch.Tensor: The computed frequencies for the two points.
    """
    M = data.shape[0]
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


def zerosum_gauge(W: torch.Tensor) -> torch.Tensor:
    """Applies the zero-sum gauge to the weight matrix of the model."""
    return W - W.mean(1, keepdim=True)

# ReLU-specific functions

def phi(x: torch.Tensor) -> torch.Tensor:
    """Applies the phi function to the input tensor: phi(x) = exp(x^2/2) * (1 - erf(x/sqrt(2))) * sqrt(2*pi)"""
    
    def phi_asym(x: torch.Tensor) -> torch.Tensor:
        return torch.pow(x, -1) - torch.pow(x, -3) + 3.0 * torch.pow(x, -5)
    
    def phi_def(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.pow(x, 2) / 2) * (1.0 - torch.erf(x / 1.4142)) * 1.2533

    return torch.where(x < 5.0, phi_def(x), phi_asym(x))


def truncated_normal(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor
) -> torch.Tensor:
    """
    Computes the probability density function of a truncated normal distribution.
    
    Args:
        x (torch.Tensor): Points at which to evaluate the PDF.
        mean (torch.Tensor): Mean of the normal distribution.
        std (torch.Tensor): Standard deviation of the normal distribution.
        a (torch.Tensor): Lower truncation bound.
        b (torch.Tensor): Upper truncation bound.
        
    Returns:
        torch.Tensor: PDF values at x.
    """
    # Standardize bounds and input
    alpha = (a - mean) / std
    beta = (b - mean) / std
    z = (x - mean) / std
    
    # Standard normal PDF: φ(z) = (1/√(2π)) * exp(-z²/2)
    sqrt_2pi = math.sqrt(2 * math.pi)
    phi_z = torch.exp(-0.5 * z**2) / sqrt_2pi
    
    # Standard normal CDF: Φ(z) = 0.5 * (1 + erf(z/√2))
    sqrt_2 = math.sqrt(2)
    Phi_alpha = 0.5 * (1 + torch.erf(alpha / sqrt_2))
    Phi_beta = 0.5 * (1 + torch.erf(beta / sqrt_2))
    
    # Normalization constant (probability mass in [a,b])
    Z = Phi_beta - Phi_alpha
    
    # Handle edge case where Z approaches 0
    Z = torch.clamp(Z, min=1e-8)
    
    # Calculate PDF: φ((x-μ)/σ) / (σ * Z)
    pdf_unnormalized = phi_z / std
    pdf = pdf_unnormalized / Z
    
    # Set PDF to 0 outside truncation bounds
    mask = (x >= a) & (x <= b)
    pdf = torch.where(mask, pdf, torch.zeros_like(pdf))
    return pdf


def sample_truncated_normal(
    mean: torch.Tensor,
    std: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor
) -> torch.Tensor:
    """Vectorized truncated normal sampling using inverse CDF.
    
    Args:
        mean (torch.Tensor): Mean of the normal distribution.
        std (torch.Tensor): Standard deviation of the normal distribution.
        a (torch.Tensor): Lower bound for truncation.
        b (torch.Tensor): Upper bound for truncation.
        
    Returns:
        torch.Tensor: Samples from the truncated normal distribution.
    """    
    # Compute standardized bounds
    alpha = (a - mean) / std
    beta = (b - mean) / std
    
    # Compute CDF values
    sqrt_2 = math.sqrt(2.0)
    alpha_cdf = 0.5 * (1 + torch.erf(alpha / sqrt_2))
    beta_cdf = 0.5 * (1 + torch.erf(beta / sqrt_2))
    
    # Sample uniform in transformed space
    u = torch.rand_like(mean)
    p = alpha_cdf + (beta_cdf - alpha_cdf) * u
    
    # Apply inverse normal CDF
    x = sqrt_2 * torch.erfinv(2 * p - 1)
    
    # Transform to original scale
    samples = mean + std * x
    
    # Ensure bounds (extra safety)
    return torch.clamp(samples, min=a, max=b)