import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, List
import copy

class Layer(ABC):
    def __init__(
        self,
        shape: int | Tuple[int, ...],
        device: Optional[torch.device] = torch.device("cpu"),
        dtype: Optional[torch.dtype] = torch.float32,
        **kwargs
    ):
        self.shape = torch.Size(shape) if isinstance(shape, tuple) else torch.Size((shape,))
        self.device = device
        self.dtype = dtype
        self.kwargs = kwargs
        self.params: Dict[str, torch.Tensor] = {}
        self.params["bias"] = torch.zeros(self.shape, device=device, dtype=dtype)
    
    
    @abstractmethod
    def init_from_frequencies(
        self,
        frequencies: torch.Tensor,
    ):
        """Initializes the layer bias using the empirical frequencies of the dataset.

        Args:
            frequencies (torch.Tensor): Empirical frequencies tensor.
        """
        assert frequencies.shape == self.shape, f"Frequencies shape ({frequencies.shape}) must match layer shape ({self.shape})."
        pass
    
    
    
    @abstractmethod
    def init_chains(
        self,
        num_chains: Optional[int] = None,
        frequencies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Initializes the Markov chains for Gibbs sampling.

        Args:
            num_chains (int): Number of Markov chains to initialize. Ignored if `frequencies` is provided.
            frequencies (torch.Tensor, optional): Empirical frequencies tensor to initialize the chains. If provided, `num_chains` is ignored.

        Returns:
            torch.Tensor: Initialized Markov chains tensor.
        """
        pass
    
    

    @abstractmethod
    def mm(self, W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Layer-specific matrix multiplication operation between weight tensor W and input tensor x.

        Args:
            W (torch.Tensor): Weight tensor.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after layer-specific matrix multiplication.
        """
        pass
    
    
    @abstractmethod
    def sample(self, I: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples from the layer's distribution given the activation input tensor.
        If `p(x)` is the layer distribution and 'I' the activation input tensor, this method samples from 'p(x|I + bias)'.

        Args:
            I (torch.Tensor): Activation input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled output tensor and the probabilities.
        """
        pass
    
    
    @abstractmethod
    def nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the nonlinearity function derived from the marginalization over the latent variables to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying nonlinearity.
        """
        pass
    
    
    @abstractmethod
    def layer_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the energy contribution of the layer given the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Energy contribution of the layer.
        """
        pass
    
    
    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> "Layer":
        """Moves the layer's parameters to the specified device and/or dtype.

        Args:
            device (Optional[torch.device], optional): Device to move the parameters to. Defaults to None.
            dtype (Optional[torch.dtype], optional): Data type to convert the parameters to. Defaults to None.

        Returns:
            Layer: The layer with updated device and dtype.
        """
        if device is not None:
            self.params = {k: v.to(device) for k, v in self.params.items()}
            self.device = device
        if dtype is not None:
            self.params = {k: v.to(dtype) for k, v in self.params.items()}
            self.dtype = dtype
        return self
    
    
    def clone(self) -> "Layer":
        """Creates a deep copy of the layer.

        Returns:
            Layer: A deep copy of the layer.
        """
        return copy.deepcopy(self)