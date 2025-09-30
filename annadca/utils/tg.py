import torch
import math


def zerosum_gauge(W: torch.Tensor) -> torch.Tensor:
    """Applies the zero-sum gauge to the weight matrix of the model."""
    return W - W.mean(1, keepdim=True)

# ReLU-specific functions

def scer(x: torch.Tensor) -> torch.Tensor:
    """Applies the log scaled complementary error function to the input tensor: scer(x) = exp(x^2/2) * (1 - erf(x/sqrt(2))) * sqrt(2*pi)"""
    return torch.special.erfcx(x / math.sqrt(2.0)) * math.sqrt(math.pi / 2.0)


def logscer(x: torch.Tensor) -> torch.Tensor:
    """Applies the log scaled complementary error function to the input tensor: logscer(x) = x^2/2 + log((1 - erf(x/sqrt(2))) * sqrt(2*pi))"""
    sqrt2 = math.sqrt(2.0)
    sqrtpi_2 = math.sqrt(math.pi / 2.0)
    
    def logphi_asym(x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.pow(x, -1) - torch.pow(x, -3) + 3.0 * torch.pow(x, -5))

    def logphi_def(x: torch.Tensor) -> torch.Tensor:
        return (torch.pow(x, 2) / 2.0) + torch.log(torch.erfc(x / sqrt2) * sqrtpi_2)

    return torch.where(x < 4.0, logphi_def(x), logphi_asym(x))


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
    Phi_alpha = 0.5 * torch.erfc(alpha / sqrt_2)
    Phi_beta = 0.5 * torch.erfc(beta / sqrt_2)

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
    # eps value to avoid numerical issues
    eps = 1e-7
    std_abs = torch.abs(std)
    # Compute standardized bounds
    alpha = (a - mean) / std_abs
    beta = (b - mean) / std_abs

    # Compute CDF values
    sqrt_2 = math.sqrt(2.0)
    alpha_cdf = 0.5 * torch.erfc(alpha / sqrt_2)
    beta_cdf = 0.5 * torch.erfc(beta / sqrt_2)

    # Sample uniform in transformed space
    u = torch.rand_like(mean)
    p = alpha_cdf + (beta_cdf - alpha_cdf) * u
    
    # Apply inverse normal CDF
    x = sqrt_2 * torch.erfinv(torch.clamp(2.0 * p - 1, min=-1 + eps, max=1 - eps))
    
    # Transform to original scale
    samples = mean + std_abs * x
    
    # Ensure bounds (extra safety)
    return torch.clamp(samples, min=a, max=b)