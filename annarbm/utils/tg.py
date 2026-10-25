import torch
import math


def zerosum_gauge(W: torch.Tensor) -> torch.Tensor:
    """Applies the zero-sum gauge to the weight matrix of the model."""
    return W - W.mean(1, keepdim=True)

# ReLU-specific functions

def _std_normal_cdf(z: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF Φ(z) = erfc(-z/√2) / 2, accurate also in the lower tail."""
    return 0.5 * torch.erfc(- z / math.sqrt(2.0))


def scer(x: torch.Tensor) -> torch.Tensor:
    """Scaled complementary error function (Mills ratio): scer(x) = exp(x^2/2) * erfc(x/sqrt(2)) * sqrt(pi/2) = (1 - Φ(x)) / φ(x)."""
    return torch.special.erfcx(x / math.sqrt(2.0)) * math.sqrt(math.pi / 2.0)


def logscer(x: torch.Tensor) -> torch.Tensor:
    """Logarithm of the scaled complementary error function: logscer(x) = x^2/2 + log(erfc(x/sqrt(2)) * sqrt(pi/2))."""
    # erfcx overflows for large negative x, while erfc underflows for large positive x
    log_direct = torch.pow(x, 2) / 2.0 + torch.log(torch.erfc(x / math.sqrt(2.0)) * math.sqrt(math.pi / 2.0))
    return torch.where(x < 0.0, log_direct, torch.log(scer(x)))


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
    phi_z = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)

    # Normalization constant (probability mass in [a,b])
    Z = _std_normal_cdf(beta) - _std_normal_cdf(alpha)

    # Handle edge case where Z approaches 0
    Z = torch.clamp(Z, min=1e-8)

    # Calculate PDF: φ((x-μ)/σ) / (σ * Z)
    pdf = phi_z / (std * Z)

    # Set PDF to 0 outside truncation bounds
    mask = (x >= a) & (x <= b)
    return torch.where(mask, pdf, torch.zeros_like(pdf))


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

    # When the interval lies in the upper tail, sample the reflected variable -z in [-beta, -alpha],
    # so that the CDF values stay small and are computed accurately.
    flip = alpha > 0
    lower = torch.where(flip, -beta, alpha)
    upper = torch.where(flip, -alpha, beta)
    lower_cdf = _std_normal_cdf(lower)
    upper_cdf = _std_normal_cdf(upper)

    # Sample uniform in transformed space and apply the inverse normal CDF
    u = torch.rand_like(mean)
    p = lower_cdf + (upper_cdf - lower_cdf) * u
    z = math.sqrt(2.0) * torch.erfinv(torch.clamp(2.0 * p - 1, min=-1 + eps, max=1 - eps))
    z = torch.where(flip, -z, z)

    # Transform to original scale
    samples = mean + std_abs * z

    # Ensure bounds (extra safety)
    return torch.clamp(samples, min=a, max=b)
