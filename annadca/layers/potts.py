from annadca.layers import Layer
import torch
from typing import Optional, Tuple

class PottsLayer(Layer):
    def __init__(
        self,
        shape: int | Tuple[int, ...],
        device: Optional[torch.device] = torch.device("cpu"),
        dtype: Optional[torch.dtype] = torch.float32,
        **kwargs
    ):
        assert isinstance(shape, tuple) and len(shape) == 2, "Shape must be a tuple of two integers (L, q) for Potts layer."
        super().__init__(shape=shape, device=device, dtype=dtype, **kwargs)


    def init_from_frequencies(
        self,
        frequencies: torch.Tensor,
    ):
        """Initializes the layer bias using the empirical frequencies of the dataset.

        Args:
            frequencies (torch.Tensor): Empirical frequencies tensor.
        """
        assert frequencies.shape == self.shape, f"Frequencies shape ({frequencies.shape}) must match layer shape ({self.shape})."
        self.params["bias"] = (torch.log(frequencies) - 1.0 / self.shape[1] * torch.sum(torch.log(frequencies), 0)).to(device=self.device, dtype=self.dtype)


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
        if frequencies is None:
            frequencies = torch.full(self.shape, 0.5, device=self.device, dtype=self.dtype)
        assert frequencies.shape == self.shape, f"Frequencies shape ({frequencies.shape}) must match layer shape ({self.shape})."
        return torch.bernoulli(frequencies.expand((num_samples,) + self.shape).to(device=self.device, dtype=self.dtype))


    def mm(self, W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Layer-specific matrix multiplication operation between weight tensor W and input tensor x.

        Args:
            W (torch.Tensor): Weight tensor.
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after layer-specific matrix multiplication.
        """
        return torch.einsum('nlq,lqh->nh', x, W)
    
    
    def sample(self, I: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples from the layer's distribution given the activation input tensor.

        Args:
            I (torch.Tensor): Activation input tensor.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled output tensor and the probabilities.
        """
        p = torch.softmax(I + self.params["bias"], dim=-1)
        x = torch.zeros_like(p).to(device=self.device, dtype=self.dtype)
        indices = torch.multinomial(p, num_samples=1).squeeze()
        x.scatter_(1, indices.unsqueeze(1), 1)
        return (x, p)
    
    
    def nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the non-linear activation function for the layer.

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after applying the non-linear activation function.
        """
        raise NotImplementedError("Nonlinearity is not implemented for Potts layer.")
    
    
    def layer_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the energy contribution of the layer given the state tensor.

        Args:
            x (torch.Tensor): State tensor.
        Returns:
            torch.Tensor: Energy contribution of the layer.
        """
        bias_term = torch.einsum('nlq,lq->n', x, self.params["bias"])
        return -bias_term
    
    
    def __repr__(self) -> str:
        return f"PottsLayer(shape={self.shape}, device={self.device}, dtype={self.dtype})"