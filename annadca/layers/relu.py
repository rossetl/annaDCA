from annadca.layers.layer import Layer
import torch
import numpy as np
from typing import Optional, Tuple, Dict
from torch.nn import Parameter
from adabmDCA.fasta import import_from_fasta, write_fasta
from annadca.utils.tg import scer, logscer, sample_truncated_normal
from annadca.utils.stats import get_mean


class ReLULayer(Layer):
    def __init__(
        self,
        shape: int | Tuple[int, ...] | torch.Size,
        **kwargs
    ):
        super().__init__(shape=shape, **kwargs)
        assert len(self.shape) == 1, f"ReLU layer shape must be one-dimensional, got {self.shape}."
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
        raise NotImplementedError("Initialization from frequencies not implemented for ReLU layer.")


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
        abs_scale = torch.abs(self.scale)
        mu = self.bias / (abs_scale + 1e-10)
        mu = mu.unsqueeze(0).repeat(num_samples, 1)
        sigma = torch.sqrt(1.0 / (abs_scale + 1e-10))
        sigma = sigma.unsqueeze(0).repeat(num_samples, 1)
        a = torch.zeros_like(mu)
        b = torch.full_like(mu, float('inf'))
        return sample_truncated_normal(mu, sigma, a, b)


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
            x (torch.Tensor): First input tensor of shape (l,) or (batch_size, l).
            y (torch.Tensor): Second input tensor of shape (h,) or (batch_size, h).
        Returns:
            torch.Tensor: Output tensor after layer-specific outer product.
        """
        if len(x.shape) == 1 and len(y.shape) == 1:
            return torch.outer(x, y)
        elif len(x.shape) == 2 and len(y.shape) == 2:
            return torch.einsum("nl,nh->lh", x, y)
        else:
            raise ValueError(f"Invalid input shapes: {x.shape}, {y.shape}")


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
        a = torch.zeros_like(mu)
        b = torch.full_like(mu, float('inf'))
        x = sample_truncated_normal(mu, sigma, a, b)
        return x
    
    
    def meanvar(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute mean and std for a normal distribution
        mu = (x + self.bias) / torch.abs(self.scale + 1e-10)
        sigma = torch.sqrt(1.0 / (torch.abs(self.scale + 1e-10)))

        # compute mean and variance for truncated normal distribution
        alpha = - mu / sigma
        scer_inv = 1.0 / scer(alpha)
        mu_t = mu + sigma * scer_inv
        var_t = torch.pow(sigma, 2) * (1 - (scer_inv - alpha) * scer_inv)
        return mu_t, var_t


    def nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the non-linear activation function for the layer: x -> log(1/ √(scale) * Phi((x + bias)/ √(scale))).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying nonlinearity.
        """
        mu = (x + self.bias) / torch.abs(self.scale + 1e-10)
        sigma = torch.sqrt(1.0 / (torch.abs(self.scale) + 1e-10))
        return torch.log(sigma) + logscer(- mu / sigma)


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


    def apply_gradient_hidden(
        self,
        mean_h_pos: torch.Tensor,
        mean_h_neg: torch.Tensor,
        var_h_pos: torch.Tensor,
        var_h_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):        
        # ∂Γ/∂θ = <h> = μ_t
        grad_bias = get_mean(mean_h_pos, weights, pseudo_count) - mean_h_neg.mean(0)

        # ∂Γ/∂γ = -0.5 * <h²> = -0.5 * (μ² + σ²)
        grad_scale_pos = - 0.5 * get_mean(torch.pow(mean_h_pos, 2) + var_h_pos, weights, pseudo_count)
        grad_scale_neg = - 0.5 * (torch.pow(mean_h_neg, 2) + var_h_neg).mean(0)
        grad_scale = grad_scale_pos - grad_scale_neg
        # Update gradients
        self.bias.grad = grad_bias
        self.scale.grad = torch.zeros_like(grad_scale)


    def apply_gradient_visible(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        raise NotImplementedError("Gradient w.r.t. visible layer not implemented for ReLU layer.")
    
    
    def standardize_gradient_visible(
        self,
        dW: torch.Tensor,
        c_h: torch.Tensor,
        **kwargs,
    ):
        """Transforms the gradient of the layer's parameters, mapping it from the standardized space back to the original space.

        Args:
            dW (torch.Tensor): Gradient of the weight matrix.
            c_h (torch.Tensor): Centering tensor for the hidden layer.
        """
        raise NotImplementedError("Standardization of gradient w.r.t. visible layer not implemented for ReLU layer.")
            
    
    def standardize_gradient_hidden(
        self,
        dW: torch.Tensor,
        dL: torch.Tensor,
        c_v: torch.Tensor,
        c_l: torch.Tensor,
        **kwargs,
    ):
        if self.bias.grad is not None:
            grad_bias = self.bias.grad / self.scale_stnd - c_v @ dW - c_l @ dL + (self.bias_stnd * self.scale) / torch.pow(self.scale_stnd, 2)
            self.bias.grad = grad_bias
        if self.scale.grad is not None:
            self.scale.grad /= torch.pow(self.scale_stnd, 2)

    def __repr__(self) -> str:
        return f"ReLULayer(shape={self.shape}, device={self.bias.device}, dtype={self.bias.dtype})"