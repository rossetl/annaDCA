from annarbm.layers.layer import Layer
from annarbm.layers.potts import PottsLayer
import torch
from typing import Optional, Tuple


class CategoricalLabelLayer(PottsLayer):
    """Label layer for categorical annotations: a single categorical variable with `num_classes` states,
    in one-hot encoding. An all-zero vector represents a missing label.
    """
    layer_type = "categorical"

    def __init__(
        self,
        shape: int | Tuple[int, ...] | torch.Size,
    ):
        # Skip the PottsLayer shape check: the label is a single categorical variable of shape (num_classes,)
        Layer.__init__(self, shape=shape)
        assert len(self.shape) == 1, f"Label layer shape must be one-dimensional, got {self.shape}."



class LabelNullLayer(Layer):
    """Dummy layer for labels. Providing this label is like training a simple RBM with no label conditioning:
    the label units are always zero and do not contribute to the energy.
    """
    layer_type = "null"
    can_be_hidden = False

    def __init__(
        self,
        shape: int | Tuple[int, ...] | torch.Size,
    ):
        super().__init__(shape=shape)
        assert len(self.shape) == 1, f"LabelNull layer shape must be one-dimensional, got {self.shape}."


    def init_params_from_data(
        self,
        data: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        self.bias.zero_()


    def init_chains(
        self,
        num_samples: int,
        data: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ) -> torch.Tensor:
        return torch.zeros((num_samples,) + self.shape, device=self.bias.device, dtype=self.bias.dtype)


    def sample(self, I: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        return torch.zeros_like(I)


    def meanvar(self, I: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros_like(I), torch.zeros_like(I)


    def nonlinearity(self, I: torch.Tensor) -> torch.Tensor:
        return torch.zeros(I.shape[:-1], device=I.device, dtype=I.dtype)


    def layer_energy(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(x), device=x.device, dtype=x.dtype)


    def apply_gradient_visible(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        self.bias.grad = torch.zeros_like(self.bias)
