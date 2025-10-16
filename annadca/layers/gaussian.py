from annadca.layers.layer import Layer
import torch
import numpy as np
from typing import Optional, Tuple, Dict
from torch.nn import Parameter
from adabmDCA.fasta import import_from_fasta, write_fasta
from annadca.utils.stats import get_mean, get_meanvar
from annadca.utils.functions import mm_left
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


    def init_params_from_data(
        self,
        data: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        """Initializes the layer parameters using the dataset statistics.

        Args:
            data (torch.Tensor): Input data tensor.
            weights (Optional[torch.Tensor], optional): Optional weight tensor for the data. Defaults to None.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
        """
        mean, var = get_meanvar(data, weights=weights, pseudo_count=pseudo_count)
        self.scale.copy_(1 / torch.sqrt(var + 1e-10))
        self.bias.copy_(mean / (var + 1e-10))
        

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
            weights (torch.Tensor, optional): Weights for the data samples. If None, uniform weights are assumed.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.

        Returns:
            torch.Tensor: Initialized Markov chains tensor.
        """
        if data is None:
            mean = self.bias / (torch.abs(self.scale) + 1e-10)
            std = torch.sqrt(1.0 / (torch.abs(self.scale) + 1e-10))
        else:
            mean, var = get_meanvar(data, weights=weights, pseudo_count=pseudo_count)
            std = torch.sqrt(var + 1e-10)
        return torch.normal(mean.expand((num_samples,) + self.shape), std.expand((num_samples,) + self.shape))


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
        """Computes the non-linear activation function for the layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying nonlinearity.
        """
        abs_scale = torch.abs(self.scale) + 1e-10
        return torch.pow((x + self.bias), 2) / (2.0 * abs_scale) + 0.5 * torch.log(2.0 * math.pi / abs_scale)


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
        mean_h_pos: torch.Tensor,
        mean_h_neg: torch.Tensor,
        var_h_pos: torch.Tensor,
        var_h_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        # ∂Γ/∂θ = <h> = μ
        grad_bias = get_mean(mean_h_pos, weights, pseudo_count) - mean_h_neg.mean(0)
        # ∂Γ/∂γ = -0.5 * <h²> = -0.5 * (μ² + σ²)
        grad_scale_pos = - 0.5 * get_mean(torch.pow(mean_h_pos, 2) + var_h_pos, weights, pseudo_count)
        grad_scale_neg = - 0.5 * (torch.pow(mean_h_neg, 2) + var_h_neg).mean(0)
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
        mean_pos, var_pos = get_meanvar(x_pos, weights=weights, pseudo_count=pseudo_count)
        mean_neg, var_neg = get_meanvar(x_neg)
        # ∂Γ/∂θ = <h> = μ
        grad_bias = mean_pos - mean_neg
        # ∂Γ/∂γ = -0.5 * <h²> = -0.5 * (μ² + σ²)
        grad_scale_pos = - 0.5 * (mean_pos**2 + var_pos)
        grad_scale_neg = - 0.5 * (mean_neg**2 + var_neg)
        grad_scale = grad_scale_pos - grad_scale_neg
        # Update gradients
        self.bias.grad = grad_bias
        self.scale.grad = grad_scale
        
        
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
        if self.bias.grad is not None:
            grad_bias = self.bias.grad / self.scale_stnd - dW @ c_h
            self.bias.grad = grad_bias
            
    
    def standardize_gradient_hidden(
        self,
        dW: torch.Tensor,
        dL: torch.Tensor,
        c_v: torch.Tensor,
        c_l: torch.Tensor,
        **kwargs,
    ):
        if self.bias.grad is not None:
            grad_bias = self.bias.grad / self.scale_stnd - mm_left(dW, c_v) - c_l @ dL
            self.bias.grad = grad_bias


    def __repr__(self) -> str:
        return f"GaussianLayer(shape={self.shape}, device={self.bias.device}, dtype={self.bias.dtype})"