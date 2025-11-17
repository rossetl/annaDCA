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
        self.kwargs = kwargs

        
    @abstractmethod
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
        pass
    
    
    @abstractmethod
    def init_params_from_data(
        self,
        data: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ):
        """Initializes the layer parameters using the input data statistics.

        Args:
            data (torch.Tensor): Input data tensor.
            weights (Optional[torch.Tensor], optional): Optional weight tensor for the data.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.

        """
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


    @abstractmethod
    def standardize_params_visible(
        self,
        scale_v: torch.Tensor,
        offset_h: torch.Tensor,
        W: torch.Tensor,
        **kwargs,
    ):
        """Transforms the parameters of the layer, mapping it from the original space to the standardized space.

        Args:
            scale_v (torch.Tensor): Scaling tensor for the visible layer.
            offset_h (torch.Tensor): Centering tensor for the hidden layer.
            W_std (torch.Tensor): Standardized weight matrix.
        """
        pass
    
    
    @abstractmethod
    def unstandardize_params_visible(
        self,
        scale_v: torch.Tensor,
        scale_h: torch.Tensor,
        offset_h: torch.Tensor,
        W: torch.Tensor,
        **kwargs,
    ):
        """Transforms the parameters of the layer, mapping it from the standardized space to the original space.

        Args:
            scale_v (torch.Tensor): Scaling tensor for the visible layer.
            offset_h (torch.Tensor): Centering tensor for the hidden layer.
            W_std (torch.Tensor): Standardized weight matrix.
        """
        pass
    
    
    @abstractmethod
    def standardize_params_hidden(
        self,
        offset_h: torch.Tensor,
        scale_h: torch.Tensor,
        offset_v: torch.Tensor,
        scale_v: torch.Tensor,
        offset_l: torch.Tensor,
        scale_l: torch.Tensor,
        W: torch.Tensor,
        L: torch.Tensor,
        **kwargs,
    ):
        """Transforms the parameters of the layer, mapping it from the original space to the standardized space.

        Args:
            offset_h (torch.Tensor): Centering tensor for the hidden layer.
            scale_h (torch.Tensor): Scaling tensor for the hidden layer.
            offset_v (torch.Tensor): Centering tensor for the visible layer.
            offset_l (torch.Tensor): Centering tensor for the label layer.
            W_std (torch.Tensor): Standardized weight matrix.
            L_std (torch.Tensor): Standardized label matrix.
        """
        pass
    
    
    @abstractmethod
    def unstandardize_params_hidden(
        self,
        offset_h: torch.Tensor,
        scale_h: torch.Tensor,
        offset_v: torch.Tensor,
        scale_v: torch.Tensor,
        offset_l: torch.Tensor,
        scale_l: torch.Tensor,
        W: torch.Tensor,
        L: torch.Tensor,
        **kwargs,
    ):
        """Transforms the parameters of the layer, mapping it from the standardized space to the original space.

        Args:
            offset_h (torch.Tensor): Centering tensor for the hidden layer.
            scale_h (torch.Tensor): Scaling tensor for the hidden layer.
            offset_v (torch.Tensor): Centering tensor for the visible layer.
            offset_l (torch.Tensor): Centering tensor for the label layer.
            W_std (torch.Tensor): Standardized weight matrix.
            L_std (torch.Tensor): Standardized label matrix.
        """
        pass