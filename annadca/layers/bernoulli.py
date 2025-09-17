from annadca.layers.layer import Layer
import torch
import numpy as np
from typing import Optional, Tuple, Dict
from torch.nn import Parameter
from adabmDCA.fasta import import_from_fasta, write_fasta


def get_freq_single_point(
    data: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    pseudo_count: float = 0.0,
) -> torch.Tensor:    
    M = len(data)
    if weights is not None:
        norm_weights = weights.reshape(M, 1) / weights.sum()
    else:
        norm_weights = torch.ones((M, 1), device=data.device) / M
    frequencies = (data * norm_weights).sum(dim=0)
    return (1. - pseudo_count) * frequencies + (pseudo_count / 2)


def get_freq_two_points(
    data: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    pseudo_count: float = 0.0,
) -> torch.Tensor:
    M = data.shape[0]
    if weights is not None:
        norm_weights = weights.reshape(M, 1) / weights.sum()
    else:
        norm_weights = torch.ones((M, 1), device=data.device) / M
    
    fij = (data * norm_weights).T @ data
    # Apply the pseudo count
    fij = (1. - pseudo_count) * fij + (pseudo_count / 4)
    # Diagonal terms must represent the single point frequencies
    fi = get_freq_single_point(data, weights, pseudo_count).ravel()
    # Apply the pseudo count on the single point frequencies
    fij_diag = (1. - pseudo_count) * fi + (pseudo_count / 2)
    # Set the diagonal terms of fij to the single point frequencies
    fij = torch.diagonal_scatter(fij, fij_diag, dim1=0, dim2=1)
    return fij


class BernoulliLayer(Layer):
    def __init__(
        self,
        shape: int | Tuple[int, ...] | torch.Size,
        **kwargs
    ):
        super().__init__(shape=shape, **kwargs)
        self.bias = Parameter(torch.zeros(self.shape), requires_grad=False)


    def init_from_frequencies(
        self,
        frequencies: torch.Tensor,
    ):
        """Initializes the layer bias using the empirical frequencies of the dataset.

        Args:
            frequencies (torch.Tensor): Empirical frequencies tensor.
        """
        assert frequencies.shape == self.shape, f"Frequencies shape ({frequencies.shape}) must match layer shape ({self.shape})."
        self.bias.copy_(torch.log(frequencies / (1 - frequencies) + 1e-10))


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
            frequencies = torch.full(self.shape, 0.5, device=device, dtype=dtype)
        assert frequencies.shape == self.shape, f"Frequencies shape ({frequencies.shape}) must match layer shape ({self.shape})."
        return torch.bernoulli(frequencies.expand((num_samples,) + self.shape).to(device=device, dtype=dtype))


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
    
    
    def get_freq_single_point(
        self,
        data: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ) -> torch.Tensor:
        """Computes the single-point frequencies of the input tensor.

        Args:
            data (torch.Tensor): Input tensor.
            weights (torch.Tensor, optional): Weights for the samples. If None, uniform weights are assumed.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.

        Returns:
            torch.Tensor: Computed single-point frequencies.
        """
        return get_freq_single_point(data, weights=weights, pseudo_count=pseudo_count)
    
    
    def get_freq_two_points(
        self,
        data: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0,
    ) -> torch.Tensor:
        """Computes the two-point frequencies of the input tensor.

        Args:
            data (torch.Tensor): Input tensor.
            weights (torch.Tensor, optional): Weights for the samples. If None, uniform weights are assumed.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.

        Returns:
            torch.Tensor: Computed two-point frequencies.
        """
        return get_freq_two_points(data, weights=weights, pseudo_count=pseudo_count)


    def forward(self, I: torch.Tensor, beta: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples from the layer's distribution given the activation input tensor.

        Args:
            I (torch.Tensor): Activation input tensor.
            beta (float): Inverse temperature parameter.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled output tensor and the probabilities.
        """
        p = torch.sigmoid(beta * (I + self.bias))
        x = torch.bernoulli(p)
        return (x, p)
    
    
    def nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the non-linear activation function for the layer: x -> log(1 + exp(bias + x)).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying nonlinearity.
        """
        I = x + self.bias
        return torch.where(I < 10, torch.log1p(torch.exp(I)), I)


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
    
    
    def apply_gradient(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        x_pos_mean = get_freq_single_point(x_pos, weights=weights, pseudo_count=pseudo_count)
        x_neg_mean = x_neg.mean(0)
        grad_bias = x_pos_mean - x_neg_mean
        self.bias.grad = grad_bias
        

    def __repr__(self) -> str:
        return f"BernoulliLayer(shape={self.shape}, device={self.device}, dtype={self.dtype})"