import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict
from annadca.utils.stats import get_meanvar


class Layer(ABC, torch.nn.Module):
    def __init__(
        self,
        shape: int | Tuple[int, ...] | torch.Size,
        **kwargs
    ):
        super(Layer, self).__init__()
        assert isinstance(shape, int) or isinstance(shape, torch.Size) or isinstance(shape, tuple) or isinstance(shape, list), "Shape must be integer, tuple, list, or torch.Size."
        self.shape = torch.Size(shape) if (isinstance(shape, tuple) or isinstance(shape, list)) else torch.Size((shape,))
        # Batch-specific variables for centering and scaling the inputs
        self.bias_stnd = torch.zeros(self.shape, requires_grad=False)
        self.scale_stnd = torch.ones(self.shape, requires_grad=False)
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
    def forward(self, I: torch.Tensor, beta: float) -> torch.Tensor:
        """Samples from the layer's distribution given the activation input tensor.
        If `p(x)` is the layer distribution and 'I' the activation input tensor, this method samples from 'p(x|I + bias)'.

        Args:
            I (torch.Tensor): Activation input tensor.
            beta (float): Inverse temperature parameter.

        Returns:
            torch.Tensor: Sampled output tensor.
        """
        pass
    
    
    @abstractmethod
    def meanvar(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the mean and variance of the input tensor with respect to the layer's distribution.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and variance of the input tensor.
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
    
    
    @abstractmethod
    def apply_gradient_visible(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        """Computes the gradient of the visible layer parameters using to the positive (data) and negative (generated) samples.

        Args:
            x_pos (torch.Tensor): Positive samples tensor.
            x_neg (torch.Tensor): Negative samples tensor.
            weights (torch.Tensor, optional): Weights for the positive samples. If None, uniform weights are assumed.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
        """
        pass
    
    
    @abstractmethod
    def apply_gradient_hidden(
        self,
        mean_h_pos: torch.Tensor,
        mean_h_neg: torch.Tensor,
        var_h_pos: torch.Tensor,
        var_h_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        """Computes the gradient of the hidden layer parameters using to the positive (data) and negative (generated) samples.

        Args:
            mean_h_pos (torch.Tensor): Mean activity of the positive samples.
            mean_h_neg (torch.Tensor): Mean activity of the negative samples.
            var_h_pos (torch.Tensor): Variance of the positive samples.
            var_h_neg (torch.Tensor): Variance of the negative samples.
            weights (torch.Tensor, optional): Weights for the positive samples. If None, uniform weights are assumed.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
        """
        pass
    
    
    def fit(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0
    ):
        mean, var = get_meanvar(x, weights=weights, pseudo_count=pseudo_count)
        self.bias_stnd, self.scale_stnd = mean, torch.ones_like(torch.sqrt(var + 1e-8))


    def transform(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return (x - self.bias_stnd) / self.scale_stnd
    
    
    def fit_transform(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0
    ) -> torch.Tensor:
        """Fits the layer to the input data and transforms it.

        Args:
            x (torch.Tensor): Input tensor to fit and transform.
            weights (Optional[torch.Tensor], optional): Weights for the input tensor. If None, uniform weights are assumed.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        self.fit(x, weights=weights, pseudo_count=pseudo_count)
        return self.transform(x)


    @abstractmethod
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
        pass
    
    
    @abstractmethod
    def standardize_gradient_hidden(
        self,
        dW: torch.Tensor,
        c_v: torch.Tensor,
        c_l: torch.Tensor,
        **kwargs,
    ):
        """Transforms the gradient of the layer's parameters, mapping it from the standardized space back to the original space.

        Args:
            dW (torch.Tensor): Gradient of the weight matrix.
            c_v (torch.Tensor): Centering tensor for the visible layer.
            c_l (torch.Tensor): Centering tensor for the label layer.
        """
        pass