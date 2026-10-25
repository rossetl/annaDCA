import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from torch.nn import Parameter
from annarbm.utils.stats import get_meanvar


LAYER_REGISTRY = {}


def build_layer(config: dict) -> "Layer":
    """Builds a layer from its configuration, as returned by `Layer.config()`."""
    config = dict(config)
    layer_type = config.pop("type")
    if layer_type not in LAYER_REGISTRY:
        raise ValueError(f"Unknown layer type: {layer_type}. Available types are: {list(LAYER_REGISTRY)}")
    return LAYER_REGISTRY[layer_type](**config)


class Layer(ABC, torch.nn.Module):
    """Base class for the layers of the RBM.

    A layer defines the single-unit potential U(x) = - bias * x + (layer-specific terms). Its energy contribution is
    sum_i U(x_i), and the conditional distribution given an input field I is p(x | I) ∝ exp(-beta * (U(x) - I * x)).

    The RBM interacts with the layers only through the methods defined here. Each subclass declares whether it can be
    used as a visible (or label) layer and/or as a hidden layer through the class attributes `can_be_visible` and
    `can_be_hidden`. Subclasses that define `layer_type` are registered, so that they can be rebuilt from their
    configuration with `build_layer`.
    """
    layer_type: Optional[str] = None
    can_be_visible: bool = True
    can_be_hidden: bool = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("layer_type") is not None:
            LAYER_REGISTRY[cls.layer_type] = cls


    def __init__(
        self,
        shape: int | Tuple[int, ...] | torch.Size,
    ):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        assert isinstance(shape, (tuple, list, torch.Size)), "Shape must be integer, tuple, list, or torch.Size."
        self.shape = torch.Size(shape)
        self.bias = Parameter(torch.zeros(self.shape), requires_grad=False)


    def config(self) -> dict:
        """Configuration of the layer: the arguments needed to rebuild it with `build_layer`."""
        return {"type": self.layer_type, "shape": list(self.shape)}


    @abstractmethod
    def init_params_from_data(
        self,
        data: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        """Initializes the layer parameters using the input data statistics.

        Args:
            data (torch.Tensor): Input data tensor.
            weights (Optional[torch.Tensor], optional): Optional weight tensor for the data.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
        """
        pass


    @abstractmethod
    def init_chains(
        self,
        num_samples: int,
        data: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ) -> torch.Tensor:
        """Initializes the Markov chains for Gibbs sampling.

        Args:
            num_samples (int): Number of Markov chains to initialize.
            data (torch.Tensor, optional): Empirical data tensor. If provided, the chains are initialized using the data statistics.
                Otherwise, they are sampled from the layer's distribution with zero input.
            weights (torch.Tensor, optional): Weights for the data samples. If None, uniform weights are assumed.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.

        Returns:
            torch.Tensor: Initialized Markov chains tensor of shape (num_samples, *shape).
        """
        pass


    @abstractmethod
    def sample(self, I: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Samples from the layer's distribution given the activation input tensor.
        If `p(x)` is the layer distribution and 'I' the activation input tensor, this method samples from 'p(x|I + bias)'.

        Args:
            I (torch.Tensor): Activation input tensor of shape (batch_size, *shape).
            beta (float): Inverse temperature parameter.

        Returns:
            torch.Tensor: Sampled output tensor of shape (batch_size, *shape).
        """
        pass


    def forward(self, I: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Same as `sample`."""
        return self.sample(I, beta)


    @abstractmethod
    def meanvar(self, I: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the mean and variance of the layer's distribution given the activation input tensor.

        Args:
            I (torch.Tensor): Activation input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and variance of the layer's units.
        """
        pass


    @abstractmethod
    def nonlinearity(self, I: torch.Tensor) -> torch.Tensor:
        """Log-partition function of each unit given the input, Γ(I) = log ∫ dx exp(-U(x) + I * x), obtained when
        marginalizing over the layer's units. For Potts-like layers the sum is taken over the states of each unit.

        Args:
            I (torch.Tensor): Activation input tensor.

        Returns:
            torch.Tensor: Γ(I) for each unit of the layer.
        """
        pass


    @abstractmethod
    def layer_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the energy contribution of the layer given the configurations: sum_i U(x_i).

        Args:
            x (torch.Tensor): Configurations tensor of shape (batch_size, *shape).

        Returns:
            torch.Tensor: Energy contribution of the layer of shape (batch_size,).
        """
        pass


    def uniform_mean(self) -> Optional[torch.Tensor]:
        """Mean of the units under the uniform distribution, used to regularize the data statistics of discrete layers
        with a pseudo count (mixture of the data with the uniform distribution). None for continuous layers, which are
        not regularized with the pseudo count."""
        return None


    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compact representation of the configurations, used to store the chains."""
        return x


    def decompress(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of `compress`. Also accepts the full (uncompressed) representation."""
        return x.to(self.bias)


    def data_meanvar(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mean and variance of the layer's units over a dataset. Used for centering and standardizing the gradient.

        Args:
            x (torch.Tensor): Configurations tensor of shape (batch_size, *shape).
            weights (torch.Tensor, optional): Weights for the samples. If None, uniform weights are assumed.
            pseudo_count (float, optional): Pseudo count used by discrete layers. Defaults to 0.0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and variance tensors of shape `shape`.
        """
        return get_meanvar(x, weights=weights)


    def apply_gradient_visible(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        """Computes the gradient of the layer parameters, when used as visible (or label) layer, using the positive (data)
        and negative (generated) samples, and stores it in the `.grad` attribute of the parameters.

        Args:
            x_pos (torch.Tensor): Positive samples tensor.
            x_neg (torch.Tensor): Negative samples tensor.
            weights (torch.Tensor, optional): Weights for the positive samples. If None, uniform weights are assumed.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
        """
        raise NotImplementedError(f"{self.__class__.__name__} cannot be used as a visible layer.")


    def apply_gradient_hidden(
        self,
        mean_h_pos: torch.Tensor,
        mean_h_neg: torch.Tensor,
        var_h_pos: torch.Tensor,
        var_h_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        """Computes the gradient of the layer parameters, when used as hidden layer, using the conditional mean and variance
        of the hidden units given the positive (data) and negative (generated) samples, and stores it in the `.grad`
        attribute of the parameters.

        Args:
            mean_h_pos (torch.Tensor): Mean activity of the hidden units given the positive samples.
            mean_h_neg (torch.Tensor): Mean activity of the hidden units given the negative samples.
            var_h_pos (torch.Tensor): Variance of the hidden units given the positive samples.
            var_h_neg (torch.Tensor): Variance of the hidden units given the negative samples.
            weights (torch.Tensor, optional): Weights for the positive samples. If None, uniform weights are assumed.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
        """
        raise NotImplementedError(f"{self.__class__.__name__} cannot be used as a hidden layer.")


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={tuple(self.shape)}, device={self.bias.device}, dtype={self.bias.dtype})"
