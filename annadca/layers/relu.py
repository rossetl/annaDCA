from annadca.layers.gaussian import GaussianLayer
import torch
from typing import Optional, Tuple
from annadca.utils.tg import scer, logscer, sample_truncated_normal
from annadca.utils.stats import get_meanvar


class ReLULayer(GaussianLayer):
    """Layer of non-negative real-valued units with potential U(x) = |scale| * x^2 / 2 - bias * x for x >= 0
    (and +inf for x < 0), i.e. a Gaussian potential truncated to the positive axis.

    Args:
        shape: Shape of the layer.
        learn_scale (bool, optional): Whether the scale (precision) of the units is learned. Defaults to False.
    """
    layer_type = "relu"

    def __init__(
        self,
        shape: int | Tuple[int, ...] | torch.Size,
        learn_scale: bool = False,
    ):
        super().__init__(shape=shape, learn_scale=learn_scale)


    def _mu_sigma(self, I: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mean and standard deviation of the (untruncated) Gaussian distribution given the input."""
        abs_scale = self._abs_scale()
        mu = (I + self.bias) / abs_scale
        sigma = torch.sqrt(1.0 / (beta * abs_scale)).expand_as(mu)
        return mu, sigma


    def init_params_from_data(
        self,
        data: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0
    ):
        raise NotImplementedError("Initialization from data not implemented for ReLU layer.")


    def init_chains(
        self,
        num_samples: int,
        data: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ) -> torch.Tensor:
        if data is None:
            mu, sigma = self._mu_sigma(torch.zeros_like(self.bias))
        else:
            mu, var = get_meanvar(data, weights=weights)
            mu, sigma = mu.to(self.bias), torch.sqrt(var + 1e-10).to(self.bias)
        mu = mu.expand((num_samples,) + self.shape)
        sigma = sigma.expand((num_samples,) + self.shape)
        return sample_truncated_normal(mu, sigma, torch.zeros_like(mu), torch.full_like(mu, float("inf")))


    def sample(self, I: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        mu, sigma = self._mu_sigma(I, beta)
        return sample_truncated_normal(mu, sigma, torch.zeros_like(mu), torch.full_like(mu, float("inf")))


    def meanvar(self, I: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, sigma = self._mu_sigma(I)
        # mean and variance of the normal distribution truncated to [0, inf)
        alpha = - mu / sigma
        inv_mills = 1.0 / scer(alpha)
        mean = mu + sigma * inv_mills
        var = torch.pow(sigma, 2) * (1 - (inv_mills - alpha) * inv_mills)
        return mean, var.clamp_min(0.0)


    def nonlinearity(self, I: torch.Tensor) -> torch.Tensor:
        """Γ(I) = log ∫_0^inf dx exp(-|scale| x^2 / 2 + (I + bias) x) = log(σ) + logscer(-μ/σ), with μ = (I + bias)/|scale| and σ = 1/√|scale|."""
        mu, sigma = self._mu_sigma(I)
        return torch.log(sigma) + logscer(- mu / sigma)
