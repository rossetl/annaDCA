from annadca.layers.layer import Layer
import torch
import numpy as np
from typing import Optional, Tuple, Dict
from torch.nn.functional import one_hot
from torch.nn import Parameter
from adabmDCA.fasta import write_fasta, get_tokens, import_from_fasta

class PottsLayer(Layer):
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
        self.bias.copy_(torch.log(frequencies) - 1.0 / self.shape[1] * torch.sum(torch.log(frequencies), 0))


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
        return torch.nn.functional.one_hot(indices, num_classes=p.size(-1)).to(dtype=dtype)


    def mm(self, W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Layer-specific matrix multiplication operation between weight tensor W and input tensor x.

        Args:
            W (torch.Tensor): Weight tensor.
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after layer-specific matrix multiplication.
        """
        return torch.einsum('nlq,lqh->nh', x, W)
    
    
    def forward(self, I: torch.Tensor, beta: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples from the layer's distribution given the activation input tensor.

        Args:
            I (torch.Tensor): Activation input tensor.
            beta (float): Inverse temperature parameter.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled output tensor and the probabilities.
        """
        p = torch.softmax(beta * (I + self.bias), dim=-1)
        indices = torch.multinomial(p.view(-1, p.size(-1)), num_samples=1).view(p.shape[:-1])
        x = one_hot(indices, num_classes=p.size(-1)).to(dtype=self.bias.dtype)
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
    
    
    def __repr__(self) -> str:
        return f"PottsLayer(shape={self.shape}, device={self.device}, dtype={self.dtype})"