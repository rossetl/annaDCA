import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict


class Layer(ABC, torch.nn.Module):
    def __init__(
        self,
        shape: int | Tuple[int, ...],
        **kwargs
    ):
        super(Layer, self).__init__()
        assert isinstance(shape, int) or isinstance(shape, torch.Size) or (isinstance(shape, tuple) and len((shape,)) == 1), "Shape must be an integer or a tuple of one integer"
        self.shape = torch.Size(shape) if isinstance(shape, tuple) else torch.Size((shape,))
        self.kwargs = kwargs
        
        
    @abstractmethod
    def init_chains(
        self,
        num_samples: Optional[int] = None,
        frequencies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Initializes the Markov chains for Gibbs sampling.

        Args:
            num_samples (int): Number of Markov chains to initialize. Ignored if `frequencies` is provided.
            frequencies (torch.Tensor, optional): Empirical frequencies tensor to initialize the chains. If provided, `num_chains` is ignored.

        Returns:
            torch.Tensor: Initialized Markov chains tensor.
        """
        pass
    
    
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
    def mm_right(self, W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Layer-specific matrix multiplication W @ x operation between weight tensor W and input tensor x.

        Args:
            W (torch.Tensor): Weight tensor.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after layer-specific matrix multiplication W @ x.
        """
        pass
    
    
    @abstractmethod
    def mm_left(self, W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Layer-specific matrix multiplication x @ W operation between weight tensor W and input tensor x.

        Args:
            W (torch.Tensor): Weight tensor.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after layer-specific matrix multiplication x @ W.
        """
        pass
    
    
    @abstractmethod
    def outer(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Layer-specific outer product operation between input tensors x and y.

        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.
        Returns:
            torch.Tensor: Output tensor after layer-specific outer product.
        """
        pass
    
    
    @abstractmethod
    def multiply(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Layer-specific element-wise multiplication operation between input tensors x and y.

        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.
        Returns:
            torch.Tensor: Output tensor after layer-specific element-wise multiplication.
        """
        pass
    
    
    @abstractmethod
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
        pass
    
    
    @abstractmethod
    def get_freq_two_points(
        self,
        data: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ) -> torch.Tensor:
        """Computes the two-point frequencies of the input tensor.

        Args:
            data (torch.Tensor): Input tensor.
            weights (torch.Tensor, optional): Weights for the samples. If None, uniform weights are assumed.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.

        Returns:
            torch.Tensor: Computed two-point frequencies.
        """
        pass


    @abstractmethod
    def forward(self, I: torch.Tensor, beta: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples from the layer's distribution given the activation input tensor.
        If `p(x)` is the layer distribution and 'I' the activation input tensor, this method samples from 'p(x|I + bias)'.

        Args:
            I (torch.Tensor): Activation input tensor.
            beta (float): Inverse temperature parameter.

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
    
    @abstractmethod
    def save_configurations(
        self,
        x: torch.Tensor,
        filepath: str,
        alphabet: str,
        headers: Optional[List[str]] = None,
    ):
        """Saves the configurations of the layer to a file.

        Args:
            x (torch.Tensor): Configurations tensor to save.
            filepath (str): Path to the file where configurations will be saved.
            headers (List[str], optional): Optional list of strings to include as a header in the file.
        """
        pass
    
    
    @abstractmethod
    def load_configurations(
        self,
        filepath: str,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Loads configurations from a file.

        Args:
            filepath (str): Path to the file from which configurations will be loaded.
            dtype (torch.dtype, optional): Desired data type of the loaded tensor. If None, uses the default dtype.
            device (torch.device, optional): Desired device of the loaded tensor. If None, uses the default device.

        Returns:
            Dict[str, torch.Tensor]: Loaded configurations dictionary.
        """
        pass
    
    
    def apply_gradient(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        """Computes the gradient of the layer parameters using to the positive (data) and negative (generated) samples.

        Args:
            x_pos (torch.Tensor): Positive samples tensor.
            x_neg (torch.Tensor): Negative samples tensor.
            weights (torch.Tensor, optional): Weights for the positive samples. If None, uniform weights are assumed.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.

        Returns:
            torch.Tensor: Computed gradient tensor.
        """
        pass