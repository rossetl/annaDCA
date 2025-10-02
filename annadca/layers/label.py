from annadca.layers.layer import Layer
import torch
import numpy as np
from typing import Optional, Tuple, Dict
from torch.nn import Parameter
from torch.nn.functional import one_hot
from adabmDCA.fasta import import_from_fasta, write_fasta
from annadca.utils.stats import get_mean


class LabelLayer(Layer):
    def __init__(
        self,
        shape: int | Tuple[int, ...] | torch.Size,
        **kwargs
    ):
        super().__init__(shape=shape, **kwargs)
        assert len(self.shape) == 1, f"Label layer shape must be one-dimensional, got {self.shape}."
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
            frequencies = torch.full(self.shape, 0.5)
        assert frequencies.shape == self.shape, f"Frequencies shape ({frequencies.shape}) must match layer shape ({self.shape})."
        frequencies = frequencies.to(device=device, dtype=dtype)
        p = frequencies / frequencies.sum(dim=-1, keepdim=True)
        indices = torch.multinomial(
            p.view(-1, p.size(-1)),
            num_samples=num_samples,
            replacement=True
        ).t()  # Transpose to get (num_samples, l)
        return torch.nn.functional.one_hot(indices, num_classes=p.size(-1)).to(dtype=dtype).squeeze(1)


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
        p = torch.softmax(beta * (I + self.bias), dim=-1)
        indices = torch.multinomial(p.view(-1, p.size(-1)), num_samples=1).view(p.shape[:-1])
        x = one_hot(indices, num_classes=p.size(-1)).to(dtype=self.bias.dtype)
        return x
    
    
    def meanvar(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Mean and variance not implemented for Label layer.")
    
    
    def nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the non-linear activation function for the layer: x -> logsumexp(x + bias).

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after applying the non-linear activation function: x -> logsumexp(x + bias).
        """
        return torch.logsumexp(x + self.bias, dim=-1)


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


    def apply_gradient_visible(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        grad_bias = get_mean(x_pos, weights=weights, pseudo_count=pseudo_count) - x_neg.mean(0)
        self.bias.grad = grad_bias.view(-1)
        
    
    def apply_gradient_hidden(
        self,
        mean_h_pos: torch.Tensor,
        mean_h_neg: torch.Tensor,
        var_h_pos: torch.Tensor,
        var_h_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        raise NotImplementedError("Gradient for hidden layer not implemented.")
    
    
    def standardize_gradient_visible(
        self,
        dW: torch.Tensor,
        c: torch.Tensor,
        **kwargs,
    ):
        """Transforms the gradient of the layer's parameters, mapping it from the standardized space back to the original space.

        Args:
            dW (torch.Tensor): Gradient of the weight matrix.
            c (torch.Tensor): Centering tensor.
        """
        if self.bias.grad is not None:
            grad_bias = self.bias.grad - dW @ c
            self.bias.grad = grad_bias
            
    
    def standardize_gradient_hidden(
        self,
        dW: torch.Tensor,
        dL: torch.Tensor,
        c_v: torch.Tensor,
        c_l: torch.Tensor,
        **kwargs,
    ):
        raise NotImplementedError("Standardize gradient for hidden layer not implemented.")


    def __repr__(self) -> str:
        return f"LabelLayer(shape={self.shape}, device={self.device}, dtype={self.dtype})"
    
    
    
class LabelNullLayer(Layer):
    """Dummy layer for labels. Providing this label is like training a simple RBM with no label conditioning."""
    def __init__(
        self,
        shape: int | Tuple[int, ...] | torch.Size,
        **kwargs
    ):
        super().__init__(shape=shape, **kwargs)
        assert len(self.shape) == 1, f"LabelNull layer shape must be one-dimensional, got {self.shape}."
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
        self.bias.copy_(torch.zeros_like(frequencies))


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
            frequencies = torch.full(self.shape, 0.5)
        assert frequencies.shape == self.shape, f"Frequencies shape ({frequencies.shape}) must match layer shape ({self.shape})."
        frequencies = frequencies.to(device=device, dtype=dtype)
        p = frequencies / frequencies.sum(dim=-1, keepdim=True)
        indices = torch.multinomial(
            p.view(-1, p.size(-1)),
            num_samples=num_samples,
            replacement=True
        ).t()  # Transpose to get (num_samples, l)
        chains = torch.nn.functional.one_hot(indices, num_classes=p.size(-1)).to(dtype=dtype).squeeze(1)
        return torch.zeros_like(chains)


    def mm_right(self, W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Layer-specific matrix multiplication operation W @ x between weight tensor W and hidden input tensor x.

        Args:
            W (torch.Tensor): Weight tensor.
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after layer-specific matrix multiplication: W @ x.
        """
        return torch.zeros_like(x @ W.t())
    
    
    def mm_left(self, W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Layer-specific matrix multiplication operation x @ W between weight tensor W and visible input tensor x.

        Args:
            W (torch.Tensor): Weight tensor.
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after layer-specific matrix multiplication: x @ W.
        """
        return torch.zeros_like(x @ W)
    
    
    def outer(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Layer-specific outer product operation between input tensors x and y.

        Args:
            x (torch.Tensor): First input tensor of shape (batch_size, l).
            y (torch.Tensor): Second input tensor of shape (batch_size, h).
        Returns:
            torch.Tensor: Output tensor after layer-specific outer product.
        """
        return torch.zeros_like(torch.einsum("nl,nh->lh", x, y))


    def multiply(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Layer-specific element-wise multiplication operation between input tensors x and y.

        Args:
            x (torch.Tensor): First input tensor of shape (batch_size, l).
            y (torch.Tensor): Second input tensor of shape (batch_size, ...).

        Returns:
            torch.Tensor: Output tensor after layer-specific element-wise multiplication.
        """
        return torch.zeros_like(x * y.view(y.shape[0], 1))


    def forward(self, I: torch.Tensor, beta: float) -> torch.Tensor:
        """Samples from the layer's distribution given the activation input tensor.

        Args:
            I (torch.Tensor): Activation input tensor.
            beta (float): Inverse temperature parameter.
            
        Returns:
            torch.Tensor: Sampled output tensor.
        """
        p = torch.softmax(beta * (I + self.bias), dim=-1)
        indices = torch.multinomial(p.view(-1, p.size(-1)), num_samples=1).view(p.shape[:-1])
        x = one_hot(indices, num_classes=p.size(-1)).to(dtype=self.bias.dtype)
        return torch.zeros_like(x)
    
    
    def meanvar(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Mean and variance not implemented for LabelNull layer.")


    def nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the non-linear activation function for the layer: x -> logsumexp(x + bias).

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after applying the non-linear activation function: x -> logsumexp(x + bias).
        """
        return x


    def layer_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the energy contribution of the layer for a given configuration.

        Args:
            x (torch.Tensor): Configuration tensor.
            
        Returns:
            torch.Tensor: Energy contribution of the layer.
        """
        return torch.zeros_like(x @ self.bias)


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


    def apply_gradient_visible(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        pass
        
    
    def apply_gradient_hidden(
        self,
        I_pos: torch.Tensor,
        I_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        raise NotImplementedError("Gradient for hidden layer not implemented.")
    
    
        

    def __repr__(self) -> str:
        return f"LabelNullLayer(shape={self.shape}, device={self.device}, dtype={self.dtype})"