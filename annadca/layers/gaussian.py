from annadca.layers.layer import Layer
import torch
from typing import Optional, Tuple
from torch.nn import Parameter
from annadca.utils.stats import get_mean, get_meanvar
import math


class GaussianLayer(Layer):
    """Layer of real-valued units with quadratic potential U(x) = |scale| * x^2 / 2 - bias * x.

    Args:
        shape: Shape of the layer.
        learn_scale (bool, optional): Whether the scale (precision) of the units is learned. Defaults to True.
    """
    layer_type = "gaussian"

    def __init__(
        self,
        shape: int | Tuple[int, ...] | torch.Size,
        learn_scale: bool = True,
    ):
        super().__init__(shape=shape)
        assert len(self.shape) == 1, f"{self.__class__.__name__} shape must be one-dimensional, got {self.shape}."
        self.scale = Parameter(torch.ones(self.shape), requires_grad=False)
        self.learn_scale = learn_scale


    def config(self) -> dict:
        return {**super().config(), "learn_scale": self.learn_scale}


    def _abs_scale(self) -> torch.Tensor:
        return torch.abs(self.scale) + 1e-10


    def init_params_from_data(
        self,
        data: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        """Initializes the layer parameters to the independent Gaussian model of the data:
        scale = 1 / var, bias = mean / var.

        Args:
            data (torch.Tensor): Input data tensor.
            weights (Optional[torch.Tensor], optional): Optional weight tensor for the data. Defaults to None.
            pseudo_count (float, optional): Not used for continuous variables.
        """
        mean, var = get_meanvar(data, weights=weights)
        precision = 1.0 / (var + 1e-6)
        self.scale.copy_(precision)
        self.bias.copy_(mean * precision)


    def init_chains(
        self,
        num_samples: int,
        data: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ) -> torch.Tensor:
        if data is None:
            mean = self.bias / self._abs_scale()
            std = torch.sqrt(1.0 / self._abs_scale())
        else:
            mean, var = get_meanvar(data, weights=weights)
            mean, std = mean.to(self.bias), torch.sqrt(var + 1e-10).to(self.bias)
        return torch.normal(mean.expand((num_samples,) + self.shape), std.expand((num_samples,) + self.shape))


    def sample(self, I: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        abs_scale = self._abs_scale()
        mu = (I + self.bias) / abs_scale
        sigma = torch.sqrt(1.0 / (beta * abs_scale))
        return mu + sigma * torch.randn_like(mu)


    def meanvar(self, I: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        abs_scale = self._abs_scale()
        mean = (I + self.bias) / abs_scale
        var = torch.ones_like(mean) / abs_scale
        return mean, var


    def nonlinearity(self, I: torch.Tensor) -> torch.Tensor:
        """Γ(I) = (I + bias)^2 / (2 |scale|) + 1/2 log(2π / |scale|)."""
        abs_scale = self._abs_scale()
        return torch.pow(I + self.bias, 2) / (2.0 * abs_scale) + 0.5 * torch.log(2.0 * math.pi / abs_scale)


    def layer_energy(self, x: torch.Tensor) -> torch.Tensor:
        u = 0.5 * torch.abs(self.scale) * torch.pow(x, 2) - self.bias * x
        return torch.sum(u, dim=1)


    def _set_gradient(
        self,
        mean_pos: torch.Tensor,
        mean_neg: torch.Tensor,
        sq_pos: torch.Tensor,
        sq_neg: torch.Tensor,
    ):
        """Sets the gradients given the first and second moments of the units: ∂U/∂bias = -x, ∂U/∂scale = sign(scale) x^2 / 2."""
        self.bias.grad = mean_pos - mean_neg
        if self.learn_scale:
            self.scale.grad = - 0.5 * torch.sign(self.scale) * (sq_pos - sq_neg)
        else:
            self.scale.grad = torch.zeros_like(self.scale)


    def apply_gradient_visible(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        self._set_gradient(
            mean_pos=get_mean(x_pos, weights=weights),
            mean_neg=x_neg.mean(0),
            sq_pos=get_mean(torch.pow(x_pos, 2), weights=weights),
            sq_neg=torch.pow(x_neg, 2).mean(0),
        )


    def apply_gradient_hidden(
        self,
        mean_h_pos: torch.Tensor,
        mean_h_neg: torch.Tensor,
        var_h_pos: torch.Tensor,
        var_h_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        # <h^2 | I> = <h | I>^2 + Var(h | I)
        self._set_gradient(
            mean_pos=get_mean(mean_h_pos, weights=weights),
            mean_neg=mean_h_neg.mean(0),
            sq_pos=get_mean(torch.pow(mean_h_pos, 2) + var_h_pos, weights=weights),
            sq_neg=(torch.pow(mean_h_neg, 2) + var_h_neg).mean(0),
        )
