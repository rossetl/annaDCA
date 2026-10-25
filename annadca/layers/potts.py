from annadca.layers.layer import Layer
import torch
from typing import Optional, Tuple
from torch.nn.functional import one_hot
from annadca.utils.stats import get_freq


class PottsLayer(Layer):
    """Layer of categorical units in one-hot encoding. The last dimension of the shape is the number of states,
    e.g. (L, q) for L Potts variables with q states each. The potential is U(x) = - sum_a bias_a * x_a.
    """
    layer_type = "potts"
    can_be_hidden = False

    def __init__(
        self,
        shape: int | Tuple[int, ...] | torch.Size,
    ):
        super().__init__(shape=shape)
        assert len(self.shape) == 2, f"Potts layer shape must be two-dimensional (L, q), got {self.shape}."


    @property
    def num_states(self) -> int:
        return self.shape[-1]


    def _freq(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ) -> torch.Tensor:
        return get_freq(x, weights=weights, pseudo_count=pseudo_count, num_states=self.num_states)


    def init_params_from_data(
        self,
        data: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0
    ):
        """Initializes the bias to the independent-site model of the data, in the zero-sum gauge:
        bias_a = log f_a - 1/q sum_b log f_b.

        Args:
            data (torch.Tensor): Input data tensor.
            weights (Optional[torch.Tensor], optional): Optional weight tensor for the data.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
        """
        log_freq = torch.log(self._freq(data, weights=weights, pseudo_count=pseudo_count).clamp_min(1e-10))
        self.bias.copy_(log_freq - log_freq.mean(dim=-1, keepdim=True))


    def _sample_from_probs(self, p: torch.Tensor) -> torch.Tensor:
        """Samples one-hot configurations from the probabilities p of shape (..., q)."""
        indices = torch.multinomial(p.reshape(-1, self.num_states), num_samples=1).view(p.shape[:-1])
        return one_hot(indices, num_classes=self.num_states).to(dtype=self.bias.dtype)


    def init_chains(
        self,
        num_samples: int,
        data: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ) -> torch.Tensor:
        if data is None:
            p = torch.softmax(self.bias, dim=-1)
        else:
            p = self._freq(data, weights=weights, pseudo_count=pseudo_count).to(self.bias)
            p = p / p.sum(dim=-1, keepdim=True)
        return self._sample_from_probs(p.expand((num_samples,) + self.shape))


    def sample(self, I: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        p = torch.softmax(beta * (I + self.bias), dim=-1)
        return self._sample_from_probs(p)


    def meanvar(self, I: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = torch.softmax(I + self.bias, dim=-1)
        var = mean * (1 - mean)
        return mean, var


    def nonlinearity(self, I: torch.Tensor) -> torch.Tensor:
        """Γ(I) = log sum_a exp(bias_a + I_a), computed for each categorical variable."""
        return torch.logsumexp(I + self.bias, dim=-1)


    def layer_energy(self, x: torch.Tensor) -> torch.Tensor:
        return - (x * self.bias).reshape(len(x), -1).sum(dim=1)


    def uniform_mean(self) -> torch.Tensor:
        return torch.full_like(self.bias, 1.0 / self.num_states)


    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Index of the active state of each variable, -1 for the all-zero vectors (e.g. missing labels)."""
        indices = x.argmax(dim=-1).to(torch.int16)
        indices[x.sum(dim=-1) == 0] = -1
        return indices


    def decompress(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_floating_point() and x.shape[1:] == self.shape:
            return x.to(self.bias)
        x = x.long()
        return (one_hot(x.clamp_min(0), num_classes=self.num_states) * (x >= 0).unsqueeze(-1)).to(self.bias)


    def data_meanvar(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self._freq(x, weights=weights, pseudo_count=pseudo_count)
        return mean, mean * (1 - mean)


    def apply_gradient_visible(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        self.bias.grad = self._freq(x_pos, weights=weights, pseudo_count=pseudo_count) - x_neg.mean(0)
