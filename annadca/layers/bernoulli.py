from annadca.layers.layer import Layer
import torch
from typing import Optional, Tuple
from torch.nn.functional import softplus
from annadca.utils.stats import get_freq, get_mean


class BernoulliLayer(Layer):
    """Layer of binary {0, 1} units with potential U(x) = - bias * x."""
    layer_type = "bernoulli"

    def __init__(
        self,
        shape: int | Tuple[int, ...] | torch.Size,
    ):
        super().__init__(shape=shape)
        assert len(self.shape) == 1, f"Bernoulli layer shape must be one-dimensional, got {self.shape}."


    def init_params_from_data(
        self,
        data: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0
    ):
        """Initializes the bias to the independent-site model of the data: bias = logit(<x>_data).

        Args:
            data (torch.Tensor): Input data tensor.
            weights (Optional[torch.Tensor], optional): Optional weight tensor for the data.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
        """
        mean = get_freq(data, weights=weights, pseudo_count=pseudo_count)
        self.bias.copy_(torch.logit(mean, eps=1e-6))


    def init_chains(
        self,
        num_samples: int,
        data: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ) -> torch.Tensor:
        if data is None:
            mean = torch.sigmoid(self.bias)
        else:
            mean = get_freq(data, weights=weights, pseudo_count=pseudo_count).to(self.bias)
        return torch.bernoulli(mean.expand((num_samples,) + self.shape))


    def sample(self, I: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        p = torch.sigmoid(beta * (I + self.bias))
        return torch.bernoulli(p)


    def meanvar(self, I: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = torch.sigmoid(I + self.bias)
        var = mean * (1 - mean)
        return mean, var


    def nonlinearity(self, I: torch.Tensor) -> torch.Tensor:
        """Γ(I) = log(1 + exp(bias + I))."""
        return softplus(I + self.bias)


    def layer_energy(self, x: torch.Tensor) -> torch.Tensor:
        return - x @ self.bias


    def uniform_mean(self) -> torch.Tensor:
        return torch.full_like(self.bias, 0.5)


    def compress(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.uint8)


    def data_meanvar(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = get_freq(x, weights=weights, pseudo_count=pseudo_count)
        return mean, mean * (1 - mean)


    def apply_gradient_visible(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        self.bias.grad = get_freq(x_pos, weights=weights, pseudo_count=pseudo_count) - x_neg.mean(0)


    def apply_gradient_hidden(
        self,
        mean_h_pos: torch.Tensor,
        mean_h_neg: torch.Tensor,
        var_h_pos: torch.Tensor,
        var_h_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        self.bias.grad = get_mean(mean_h_pos, weights=weights) - mean_h_neg.mean(0)
