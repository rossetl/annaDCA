from annadca.layers.layer import Layer
import torch
import numpy as np
from typing import Optional, Tuple, Dict
from torch.nn.functional import one_hot
from torch.nn import Parameter
from adabmDCA.fasta import write_fasta, get_tokens, import_from_fasta
from annadca.utils.stats import get_mean
        

class PottsLayer(Layer):
    def __init__(
        self,
        shape: int | Tuple[int, ...] | torch.Size,
        **kwargs
    ):
        super().__init__(shape=shape, **kwargs)
        assert len(self.shape) == 2, f"Potts layer shape must be two-dimensional (L, q), got {self.shape}."
        self.bias = Parameter(torch.zeros(self.shape), requires_grad=False)


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
        self.bias.copy_(torch.log(mean) - 1.0 / self.shape[1] * torch.sum(torch.log(mean), 0))


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
        dtype = next(self.parameters()).dtype
        if data is None:
            mean = torch.softmax(self.bias, dim=-1)
        else:
            mean = get_mean(data, weights=weights, pseudo_count=pseudo_count)
        p = mean / mean.sum(dim=-1, keepdim=True)
        indices = torch.multinomial(
            p.view(-1, p.size(-1)), 
            num_samples=num_samples, 
            replacement=True
        ).t()  # Transpose to get (num_samples, l)
        return torch.nn.functional.one_hot(indices, num_classes=p.size(-1)).to(dtype=dtype)


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
        mean = torch.softmax(x + self.bias, dim=-1)
        var = mean * (1 - mean)
        return mean, var


    def nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the non-linear activation function for the layer: x -> logsumexp(x + bias).

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after applying the non-linear activation function: x -> logsumexp(x + bias).
        """
        return torch.logsumexp(x + self.bias, dim=-1)
    
    
    def layer_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the energy contribution of the layer given the state tensor.

        Args:
            x (torch.Tensor): State tensor.
        Returns:
            torch.Tensor: Energy contribution of the layer.
        """
        bias_term = torch.einsum('nlq,lq->n', x, self.bias)
        return -bias_term
    
    
    def save_configurations(
        self,
        chains: Dict[str, torch.Tensor],
        filepath: str,
        alphabet: str,
    ):
        """Saves the configurations of the layer to a file.
        Args:
            chains (Dict[str, torch.Tensor]): Dictionary containing the chain configurations.
            filepath (str): Path to the file where configurations will be saved.
        """
        tokens = get_tokens(alphabet)
        headers = np.vectorize(lambda x: "".join([str(i) for i in x]), signature="(l) -> ()")(chains["label"].cpu().numpy())
        write_fasta(
            fname=filepath,
            headers=headers,
            sequences=chains["visible"].argmax(dim=-1).cpu().numpy(),
            numeric_input=True,
            tokens=tokens,
            remove_gaps=False,
        )
        
        
    def load_configurations(
        self,
        filepath: str,
        alphabet: str,
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
        tokens = get_tokens(alphabet)
        headers, visible = import_from_fasta(filepath, tokens)
        label = np.vectorize(lambda x: np.array([int(i) for i in x]), signature="() -> (l)")(headers)
        label = torch.tensor(label, device=device, dtype=dtype)
        visible = one_hot(torch.tensor(visible), len(tokens)).to(device=device, dtype=dtype)
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
        I_pos: torch.Tensor,
        I_neg: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0,
    ):
        raise NotImplementedError("Gradient w.r.t. hidden layer not implemented for Potts layer.")
    
    
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
            grad_bias = self.bias.grad / self.scale_stnd - c_v.view(-1) @ dW.view(-1, dW.shape[2]) - c_l @ dL
            self.bias.grad = grad_bias
        
    
    def __repr__(self) -> str:
        return f"PottsLayer(shape={self.shape}, device={self.bias.device}, dtype={self.bias.dtype})"