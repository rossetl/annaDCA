from annadca.layers.layer import Layer
import torch
import numpy as np
from typing import Optional, Tuple, Dict
from torch.nn import Parameter
from adabmDCA.fasta import import_from_fasta, write_fasta
from annadca.utils.stats import get_mean
from annadca.utils.functions import mm_left, outer


class BernoulliLayer(Layer):
    def __init__(
        self,
        shape: int | Tuple[int, ...] | torch.Size,
        **kwargs
    ):
        super().__init__(shape=shape, **kwargs)
        assert len(self.shape) == 1, f"Bernoulli layer shape must be one-dimensional, got {self.shape}."
        self.bias = Parameter(torch.zeros(self.shape), requires_grad=False, )


    def init_params_from_data(
        self,
        data: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0
    ):
        """Initializes the layer parameters using the input data statistics.

        Args:
            data (torch.Tensor): Input data tensor.
            weights (Optional[torch.Tensor], optional): Optional weight tensor for the data.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.

        """
        mean = get_mean(data, weights=weights, pseudo_count=pseudo_count)
        self.bias.copy_(torch.log(mean / (1 - mean) + 1e-10))


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
            mean = torch.sigmoid(self.bias)
        else:
            mean = get_mean(data, weights=weights, pseudo_count=pseudo_count)
        return torch.bernoulli(mean.expand((num_samples,) + self.shape))


    def forward(self, I: torch.Tensor, beta: float) -> torch.Tensor:
        """Samples from the layer's distribution given the activation input tensor.

        Args:
            I (torch.Tensor): Activation input tensor.
            beta (float): Inverse temperature parameter.
            
        Returns:
            torch.Tensor: Sampled output tensor.
        """
        p = torch.sigmoid(beta * (I + self.bias))
        x = torch.bernoulli(p)
        return x
    
    
    def meanvar(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the mean and variance of the layer's distribution given the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and variance tensors.
        """
        mean = torch.sigmoid(x + self.bias)
        var = mean * (1 - mean)
        return mean, var
        
    
    def nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the non-linear activation function for the layer: x -> log(1 + exp(bias + x)).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying nonlinearity.
        """
        return torch.where(x < 10, torch.log1p(torch.exp(x + self.bias)), x + self.bias)


    def layer_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the energy contribution of the layer for a given configuration.

        Args:
            x (torch.Tensor): Configuration tensor.
            
        Returns:
            torch.Tensor: Energy contribution of the layer.
        """
        return - x @ self.bias
    
    
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


    def apply_gradient_visible(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        grad_bias = get_mean(x_pos, weights=weights, pseudo_count=pseudo_count) - x_neg.mean(0)
        self.bias.grad = grad_bias
        
    
    def apply_gradient_hidden(
        self,
        mean_h_pos: torch.Tensor,
        mean_h_neg: torch.Tensor,
        var_h_pos: torch.Tensor,
        var_h_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        grad_bias = get_mean(mean_h_pos, weights=weights, pseudo_count=pseudo_count) - mean_h_neg.mean(0)
        self.bias.grad = grad_bias


    def standardize_params_visible(
        self,
        scale_v: torch.Tensor,
        scale_h: torch.Tensor,
        offset_h: torch.Tensor,
        W: torch.Tensor,
        **kwargs,
    ):
        """Transforms the parameters of the layer, mapping it from the original space to the standardized space.
        
        Args:
            scale_v (torch.Tensor): Scaling tensor for the visible layer.
            offset_h (torch.Tensor): Centering tensor for the hidden layer.
            W (torch.Tensor): Weight matrix.
        """
        self.bias.copy_(self.bias / scale_v - (W / outer(scale_v, scale_h)) @ offset_h)
        
    
    def unstandardize_params_visible(
        self,
        scale_v: torch.Tensor,
        scale_h: torch.Tensor,
        offset_h: torch.Tensor,
        W: torch.Tensor,
        **kwargs,
    ):
        self.bias.copy_(scale_v * (self.bias + (W / outer(scale_v, scale_h)) @ offset_h))
        

    def standardize_params_hidden(
        self,
        scale_h: torch.Tensor,
        scale_v: torch.Tensor,
        scale_l: torch.Tensor,
        offset_v: torch.Tensor,
        offset_l: torch.Tensor,
        W: torch.Tensor,
        L: torch.Tensor,
        **kwargs,
    ):
        """Transforms the parameters of the layer, mapping it from the original space to the standardized space.
        Args:
            scale_h (torch.Tensor): Scaling tensor for the hidden layer.
            offset_v (torch.Tensor): Centering tensor for the visible layer.
            offset_l (torch.Tensor): Centering tensor for the label layer.
            W_std (torch.Tensor): Standardized weight matrix.
            L_std (torch.Tensor): Standardized label matrix.
        """
        self.bias.copy_(self.bias / scale_h - mm_left(W / outer(scale_v, scale_h), offset_v) - offset_l @ (L / outer(scale_l, scale_h)))
        
    
    def unstandardize_params_hidden(
        self,
        scale_h: torch.Tensor,
        scale_v: torch.Tensor,
        scale_l: torch.Tensor,
        offset_v: torch.Tensor,
        offset_l: torch.Tensor,
        W: torch.Tensor,
        L: torch.Tensor,
        **kwargs,
    ):
        """Transforms the parameters of the layer, mapping it from the standardized space to the original space.
        
        Args:
            scale_h (torch.Tensor): Scaling tensor for the hidden layer.
            offset_v (torch.Tensor): Centering tensor for the visible layer.
            offset_l (torch.Tensor): Centering tensor for the label layer.
            W_std (torch.Tensor): Standardized weight matrix.
            L_std (torch.Tensor): Standardized label matrix.
        """
        self.bias.copy_(scale_h * (self.bias + mm_left(W / outer(scale_v, scale_h), offset_v) + offset_l @ (L / outer(scale_l, scale_h))))
        

    def __repr__(self) -> str:
        return f"BernoulliLayer(shape={self.shape}, device={self.bias.device}, dtype={self.bias.dtype})"