from annadca.layers.layer import Layer
import torch
import numpy as np
from typing import Optional, Tuple, Dict
from torch.nn import Parameter
from adabmDCA.fasta import import_from_fasta, write_fasta
from annadca.utils.stats import get_mean
import math


class GaussianLayer(Layer):
    def __init__(
        self,
        shape: int | Tuple[int, ...] | torch.Size,
        **kwargs
    ):
        super().__init__(shape=shape, **kwargs)
        assert len(self.shape) == 1, f"Gaussian layer shape must be one-dimensional, got {self.shape}."
        self.bias = Parameter(torch.zeros(self.shape), requires_grad=False)
        self.scale = Parameter(torch.ones(self.shape), requires_grad=False)


    def init_from_frequencies(
        self,
        frequencies: torch.Tensor,
    ):
        """Initializes the layer bias using the empirical frequencies of the dataset.

        Args:
            frequencies (torch.Tensor): Empirical frequencies tensor.
        """
        raise NotImplementedError("Initialization from frequencies not implemented for Gaussian layer.")


    def init_chains(
        self,
        num_samples: int,
        frequencies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Initializes the Markov chains for Gibbs sampling.

        Args:
            num_samples (int): Number of Markov chains to initialize.
            frequencies (torch.Tensor, optional): Empirical frequencies tensor to sample the chains from.

        Returns:
            torch.Tensor: Initialized Markov chains tensor.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        if frequencies is None:
            frequencies = torch.full(self.shape, 0.0, device=device, dtype=dtype)
        assert frequencies.shape == self.shape, f"Frequencies shape ({frequencies.shape}) must match layer shape ({self.shape})."
        return torch.normal(frequencies.expand((num_samples,) + self.shape).to(device=device, dtype=dtype), 1.0)


    def mm_right(self, W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Layer-specific matrix multiplication operation W @ x between weight tensor W and hidden input tensor x.

        Args:
            W (torch.Tensor): Weight tensor.
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after layer-specific matrix multiplication: W @ x.
        """
        return x @ W.t()
    
    
    def mm_left(self, W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Layer-specific matrix multiplication operation x @ W between weight tensor W and visible input tensor x.

        Args:
            W (torch.Tensor): Weight tensor.
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after layer-specific matrix multiplication: x @ W.
        """
        return x @ W
    
    
    def outer(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Layer-specific outer product operation between input tensors x and y.

        Args:
            x (torch.Tensor): First input tensor of shape (batch_size, l).
            y (torch.Tensor): Second input tensor of shape (batch_size, h).
        Returns:
            torch.Tensor: Output tensor after layer-specific outer product.
        """
        return torch.einsum("nl,nh->lh", x, y)


    def multiply(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Layer-specific element-wise multiplication operation between input tensors x and y.

        Args:
            x (torch.Tensor): First input tensor of shape (batch_size, l).
            y (torch.Tensor): Second input tensor of shape (batch_size, ...).

        Returns:
            torch.Tensor: Output tensor after layer-specific element-wise multiplication.
        """
        return x * y.view(y.shape[0], 1)


    def forward(self, I: torch.Tensor, beta: float) -> torch.Tensor:
        """Samples from the layer's distribution given the activation input tensor.

        Args:
            I (torch.Tensor): Activation input tensor.
            beta (float): Inverse temperature parameter.
            
        Returns:
            torch.Tensor: Sampled output tensor.
        """
        abs_scale = torch.abs(self.scale)
        mu = (I + self.bias) / abs_scale
        sigma = torch.sqrt(1.0 / (beta * abs_scale + 1e-10))
        x = torch.normal(mu, sigma)
        return x


    def nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the non-linear activation function for the layer: x -> log(1/ √(scale) * Phi((-x + bias)/ √(scale))).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying nonlinearity.
        """
        abs_scale = torch.abs(self.scale)
        return torch.pow((x + self.bias), 2) / (2.0 * abs_scale + 1e-10) + 0.5 * torch.log(2.0 * math.pi / (abs_scale + 1e-10))


    def layer_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the energy contribution of the layer for a given configuration: sum(0.5 * scale * x^2 + bias * x, axis=1).

        Args:
            x (torch.Tensor): Configuration tensor.
            
        Returns:
            torch.Tensor: Energy contribution of the layer.
        """
        u = 0.5 * torch.abs(self.scale.unsqueeze(0)) * torch.pow(x, 2) - self.bias.unsqueeze(0) * x
        return torch.sum(u, dim=1)
    
    
    def save_configurations(
        self,
        chains: Dict[str, torch.Tensor],
        filepath: str):
        """Saves the configurations of the layer to a file.
        Args:
            chains (Dict[str, torch.Tensor]): Dictionary containing the chain configurations.
            filepath (str): Path to the file where configurations will be saved.
        """
        tokens = "01"
        headers = np.vectorize(lambda x: "".join([str(i) for i in x]), signature="(l) -> ()")(chains["label"].cpu().numpy())
        write_fasta(
            fname=filepath,
            headers=headers,
            sequences=chains["visible"].cpu().numpy(),
            numeric_input=True,
            tokens=tokens,
            remove_gaps=False,
        )
    
    
    def load_configurations(
        self,
        filepath: str,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        **kwargs,
        ) -> Dict[str, torch.Tensor]:
        """Loads configurations from a fasta file.

        Args:
            filepath (str): Path to the fasta file containing the configurations.
            dtype (Optional[torch.dtype], optional): Desired data type of the loaded tensor. If None, uses the default dtype.
            device (Optional[torch.device], optional): Desired device of the loaded tensor. If None, uses the default device.

        Returns:
            Dict[str, torch.Tensor]: Loaded configurations dictionary.
        """
        
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        headers, sequences = import_from_fasta(filepath)
        label = np.vectorize(lambda x: np.array([int(i) for i in x]), signature="() -> (l)")(headers)
        visible = np.vectorize(lambda x: np.array([int(i) for i in x]), signature="() -> (l)")(sequences)
        label = torch.tensor(label, device=device, dtype=dtype)
        visible = torch.tensor(visible, device=device, dtype=dtype)
        hidden = torch.zeros((visible.shape[0],), device=device, dtype=dtype)
        return {"visible": visible, "hidden": hidden, "label": label}
    
    
    def meanvar(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        abs_scale = torch.abs(self.scale) + 1e-10
        mean = (x + self.bias) / abs_scale
        var = 1.0 / abs_scale
        return mean, var


    def apply_gradient_hidden(
        self,
        I_pos: torch.Tensor,
        I_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        mean_pos, var_pos = self.meanvar(I_pos)
        mean_neg, var_neg = self.meanvar(I_neg)
        # ∂Γ/∂θ = <h> = μ
        grad_bias = get_mean(mean_pos, weights, pseudo_count) - mean_neg.mean(0)
        # ∂Γ/∂γ = -0.5 * <h²> = -0.5 * (μ² + σ²)
        grad_scale_pos = - 0.5 * get_mean(torch.pow(mean_pos, 2) + var_pos, weights, pseudo_count)
        grad_scale_neg = - 0.5 * (torch.pow(mean_neg, 2) + var_neg).mean(0)
        grad_scale = grad_scale_pos - grad_scale_neg
        # Update gradients
        self.bias.grad = grad_bias
        self.scale.grad = grad_scale
        
    
    def apply_gradient_visible(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        raise NotImplementedError("Gradient w.r.t. visible layer not implemented for Gaussian layer.")


    def __repr__(self) -> str:
        return f"GaussianLayer(shape={self.shape}, device={self.device}, dtype={self.dtype})"