from annadca.layers import Layer
from annadca.layers.potts import PottsLayer
import torch
from typing import Optional, Tuple, Dict

class AnnaRBM():
    def __init__(
        self,
        visible_layer: Layer,
        hidden_layer: Layer,
        num_classes: int,
        device: Optional[torch.device] = torch.device("cpu"),
        dtype: Optional[torch.dtype] = torch.float32,
        **kwargs
    ):
        assert isinstance(visible_layer, Layer), "visible_layer must be an instance of Layer."
        assert isinstance(hidden_layer, Layer), "hidden_layer must be an instance of Layer."
        assert not isinstance(hidden_layer, PottsLayer), "hidden_layer cannot be a PottsLayer."
        assert isinstance(num_classes, int) and num_classes > 1, "num_classes must be a positive integer greater than 1."
        
        self.visible_layer = visible_layer.to(device=device, dtype=dtype)
        self.hidden_layer = hidden_layer.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.kwargs = kwargs
        self.shape = self.visible_layer.shape + self.hidden_layer.shape # (num_visible, num_states, num_hidden)
        self.num_classes = torch.Size((num_classes,))
        
        self.params = {
            "weight_matrix": torch.randn(self.shape, device=device, dtype=dtype) * 1e-4,
            "label_matrix": torch.randn(self.hidden_layer.shape + (num_classes,), device=device, dtype=dtype) * 1e-4,
            "lbias": torch.zeros((num_classes,), device=device, dtype=dtype),
            "vbias": self.visible_layer.params["bias"],
            "hbias": self.hidden_layer.params["bias"],
        }
        
    
    def init_chains(
        self,
        num_samples: int,
        frequencies: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Initializes the Markov chains for Gibbs sampling.

        Args:
            num_samples (int): Number of Markov chains to initialize.
            frequencies (torch.Tensor, optional): Empirical frequencies tensor to sample the visible chains from.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the chain type and the initialized chains tensor.
        """
        chains = {}
        chains["visible"] = self.visible_layer.init_chains(num_chains=num_samples, frequencies=frequencies)
        chains["hidden"] = self.hidden_layer.init_chains(num_chains=num_samples)
        chains["label"] = torch.zeros((num_samples,) + self.num_classes, device=self.device, dtype=self.dtype)
        chains["visible_mag"] = torch.zeros((num_samples,) + self.visible_layer.shape, device=self.device, dtype=self.dtype)
        chains["hidden_mag"] = torch.zeros((num_samples,) + self.hidden_layer.shape, device=self.device, dtype=self.dtype)
        chains["label_mag"] = torch.zeros((num_samples,) + self.num_classes, device=self.device, dtype=self.dtype)
        return chains


    def sample_visibles(
        self,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples the visible units given the hidden units.

        Args:
            h (torch.Tensor): Hidden units tensor of shape (batch_size, num_hiddens).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled visible units and their probabilities.
        """
        I_v = self.visible_layer.mm(self.params["weight_matrix"], hidden)
        return self.visible_layer.sample(I_v)
    
    
    def sample_hiddens(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples the hidden units given the visible units.

        Args:
            visible (torch.Tensor): Visible units tensor of shape (batch_size, num_visibles).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled hidden units and their probabilities.
        """
        I_h = self.hidden_layer.mm(self.params["weight_matrix"], visible) + label @self.params["label_matrix"]
        return self.hidden_layer.sample(I_h)
    
    
    def sample_labels(
        self,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples the label units given the hidden units.

        Args:
            hidden (torch.Tensor): Hidden units tensor of shape (batch_size, num_hiddens).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled label units and their probabilities.
        """
        I_l = hidden @ self.params["label_matrix"].T
        p = torch.softmax(I_l + self.params["lbias"], dim=-1)
        l = torch.zeros_like(p).to(device=self.device, dtype=self.dtype)
        indices = torch.multinomial(p, num_samples=1).squeeze()
        l.scatter_(1, indices.unsqueeze(1), 1)
        return (l, p)
    
    
    def sample(
        self,
        gibbs_steps: int,
        num_samples: int | None = None,
        visible: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        beta: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Generates samples from the RBM using Gibbs sampling.

        Args:
            gibbs_steps (int): Number of Gibbs sampling steps.
            num_samples (int | None, optional): Number of samples to generate. If None, uses the batch size of the provided visible or label tensors. Defaults to None.
            visible (torch.Tensor | None, optional): Initial visible units tensor. If None, initializes randomly. Defaults to None.
            label (torch.Tensor | None, optional): Initial label units tensor. If None, initializes randomly. Defaults to None.
            beta (float, optional): Inverse temperature parameter for sampling. Defaults to 1.0.
        """
        # Infer num_samples if possible
        if visible is not None:
            num_samples = visible.shape[0]
        elif label is not None:
            num_samples = label.shape[0]
        elif num_samples is None or num_samples <= 0:
            raise ValueError("Either visible, label or a positive num_samples must be provided.")
        
        # Initialize missing visible/label
        chains_init = self.init_chains(num_samples=num_samples)
        if visible is None:
            visible = chains_init["visible"]
        if label is None:
            label = chains_init["label"]

        if visible.shape[0] != label.shape[0]:
            raise ValueError(f"The number of visible units ({visible.shape[0]}) and labels ({label.shape[0]}) must be the same.")
        
        for _ in range(gibbs_steps):
            hidden, hidden_mag = self.sample_hiddens(visible, label)
            visible, visible_mag = self.sample_visibles(hidden)
            label, label_mag = self.sample_labels(hidden)
            
        return {
            "visible": visible,
            "hidden": hidden,
            "label": label,
            "visible_mag": visible_mag,
            "hidden_mag": hidden_mag,
            "label_mag": label_mag,
        }

        