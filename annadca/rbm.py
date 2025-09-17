from annadca.layers.layer import Layer
from annadca.layers.bernoulli import BernoulliLayer
from annadca.layers.potts import PottsLayer
import torch
from typing import Optional, Tuple, Dict
from torch.nn import Parameter
from torch.nn.functional import one_hot
from annadca.layers.bernoulli import get_freq_single_point


def get_rbm(
    visible_type: str,
    hidden_type: str,
    visible_shape: int | Tuple[int, ...] | torch.Size,
    hidden_shape: int | torch.Size,
    num_classes: int,
    **kwargs,
) -> "AnnaRBM":
    available_layers = {
        "potts": PottsLayer,
        "bernoulli": BernoulliLayer,
    }
    assert visible_type in available_layers, f"Unknown visible layer type: {visible_type}. Available types are: {list(available_layers.keys())}"
    assert hidden_type in available_layers, f"Unknown hidden layer type: {hidden_type}. Available types are: {list(available_layers.keys())}"
    visible_layer = available_layers[visible_type](shape=visible_shape)
    hidden_layer = available_layers[hidden_type](shape=hidden_shape)
    return AnnaRBM(
        visible_layer=visible_layer,
        hidden_layer=hidden_layer,
        num_classes=num_classes,
        **kwargs
    )


class AnnaRBM(torch.nn.Module):
    def __init__(
        self,
        visible_layer: Layer,
        hidden_layer: Layer,
        num_classes: int,
        **kwargs
    ):
        super(AnnaRBM, self).__init__()
        assert isinstance(visible_layer, Layer), "visible_layer must be an instance of Layer."
        assert isinstance(hidden_layer, Layer), "hidden_layer must be an instance of Layer."
        assert not isinstance(hidden_layer, PottsLayer), "hidden_layer cannot be a PottsLayer."
        assert isinstance(num_classes, int) and num_classes > 1, "num_classes must be a positive integer greater than 1."
        
        self.visible_layer = visible_layer
        self.hidden_layer = hidden_layer
        self.kwargs = kwargs
        self.shape = self.visible_layer.shape + self.hidden_layer.shape # (num_visible, num_states, num_hidden)
        self.num_classes = torch.Size((num_classes,))
        
        self.weight_matrix = Parameter(torch.randn(self.shape) * 1e-4, requires_grad=False)
        self.label_matrix = Parameter(torch.randn(self.num_classes + self.hidden_layer.shape) * 1e-4, requires_grad=False)
        self.lbias = Parameter(torch.zeros((num_classes,)), requires_grad=False)
        
    
    def init_from_frequencies(
        self,
        frequencies_visible: torch.Tensor,
        frequencies_label: torch.Tensor,
    ):
        """Initializes the visible layer bias using the empirical frequencies of the dataset.

        Args:
            frequencies (torch.Tensor): Empirical frequencies tensor.
        """
        self.visible_layer.init_from_frequencies(frequencies_visible)
        self.lbias.copy_(torch.log(frequencies_label / (1 - frequencies_label) + 1e-10))


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
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        chains = {}
        chains["visible"] = self.visible_layer.init_chains(num_samples=num_samples, frequencies=frequencies)
        chains["hidden"] = self.hidden_layer.init_chains(num_samples=num_samples)
        chains["label"] = torch.zeros((num_samples,) + self.num_classes, device=device, dtype=dtype)
        chains["visible_mag"] = torch.zeros((num_samples,) + self.visible_layer.shape, device=device, dtype=dtype)
        chains["hidden_mag"] = torch.zeros((num_samples,) + self.hidden_layer.shape, device=device, dtype=dtype)
        chains["label_mag"] = torch.zeros((num_samples,) + self.num_classes, device=device, dtype=dtype)
        return chains
    
    
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
        return self.visible_layer.get_freq_single_point(data, weights=weights, pseudo_count=pseudo_count)
    
    
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
        return self.visible_layer.get_freq_two_points(data, weights=weights, pseudo_count=pseudo_count)


    def sample_visibles(
        self,
        hidden: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples the visible units given the hidden units.

        Args:
            h (torch.Tensor): Hidden units tensor of shape (batch_size, num_hiddens).
            beta (float, optional): Inverse temperature parameter for sampling. Defaults to 1.0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled visible units and their probabilities.
        """
        I_v = self.visible_layer.mm_right(self.weight_matrix, hidden)
        return self.visible_layer.forward(I_v, beta)


    def sample_hiddens(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples the hidden units given the visible units.

        Args:
            visible (torch.Tensor): Visible units tensor of shape (batch_size, num_visibles, ...).
            label (torch.Tensor): Label units tensor of shape (batch_size, num_classes).
            beta (float, optional): Inverse temperature parameter for sampling. Defaults to 1.0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled hidden units and their probabilities.
        """
        I_h = self.visible_layer.mm_left(self.weight_matrix, visible) + label @ self.label_matrix
        return self.hidden_layer.forward(I_h, beta)
    
    
    def sample_labels(
        self,
        hidden: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples the label units given the hidden units.

        Args:
            hidden (torch.Tensor): Hidden units tensor of shape (batch_size, num_hiddens).
            beta (float, optional): Inverse temperature parameter for sampling. Defaults to 1.0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled label units and their probabilities.
        """
        I_l = hidden @ self.label_matrix.T
        p = torch.softmax(beta * (I_l + self.lbias), dim=-1)
        indices = torch.multinomial(p.view(-1, p.size(-1)), num_samples=1).view(p.shape[:-1])
        l = one_hot(indices, num_classes=p.size(-1)).to(dtype=self.weight_matrix.dtype)
        return (l, p)
    
    
    def gibbs_step(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes a forward pass through the RBM, returning the configurations and the probabilities.

        Args:
            visible (torch.Tensor): Visible units tensor of shape (batch_size, num_visibles, ...).
            label (torch.Tensor): Label units tensor of shape (batch_size, num_classes).
            beta (float, optional): Inverse temperature parameter for sampling. Defaults to 1.0.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Visible units, hidden units, label units, visible probabilities, hidden probabilities, label probabilities.
            
        """
        h, h_p = self.sample_hiddens(visible, label, beta)
        v, v_p = self.sample_visibles(h, beta)
        l, l_p = self.sample_labels(h, beta)
        return (v, h, l, v_p, h_p, l_p)
    
    
    def forward(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Alias for the gibbs_step method.

        Args:
            visible (torch.Tensor): Visible units tensor of shape (batch_size, num_visibles, ...).
            label (torch.Tensor): Label units tensor of shape (batch_size, num_classes).
            beta (float, optional): Inverse temperature parameter for sampling. Defaults to 1.0.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Visible units, hidden units, label units, visible probabilities, hidden probabilities, label probabilities.
        """
        return self.gibbs_step(visible, label, beta)
    
    
    def sample(
        self,
        gibbs_steps: int,
        num_samples: Optional[int] = None,
        visible: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        beta: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Generates samples from the RBM using Gibbs sampling.

        Args:
            gibbs_steps (int): Number of Gibbs sampling steps.
            num_samples (int, optional): Number of samples to generate. If None, uses the batch size of the provided visible or label tensors. Defaults to None.
            visible (torch.Tensor, optional): Initial visible units tensor. If None, initializes randomly. Defaults to None.
            label (torch.Tensor, optional): Initial label units tensor. If None, initializes randomly. Defaults to None.
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
        assert visible.shape[0] == label.shape[0], f"The number of visible units ({visible.shape[0]}) and labels ({label.shape[0]}) must be the same."

        for _ in range(gibbs_steps):
            visible, hidden, label, visible_prob, hidden_prob, label_prob = self.gibbs_step(visible, label, beta)
            
        return {
            "visible": visible,
            "hidden": hidden,
            "label": label,
            "visible_prob": visible_prob,
            "hidden_prob": hidden_prob,
            "label_prob": label_prob,
        }
        
    
    def sample_conditioned(
        self,
        gibbs_steps: int,
        targets: torch.Tensor,
        visible: torch.Tensor | None = None,
        beta: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Generates samples from the RBM using Gibbs sampling, conditioning on the provided target labels.

        Args:
            gibbs_steps (int): Number of Gibbs sampling steps.
            targets (torch.Tensor): Target one-hot label units tensor of shape (batch_size, num_classes).
            visible (torch.Tensor | None, optional): Initial visible units tensor. If None, initializes randomly. Defaults to None.
            beta (float, optional): Inverse temperature parameter for sampling. Defaults to 1.0.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        targets = targets.to(device=device, dtype=dtype)
        num_samples = targets.shape[0]
        if visible is None:
            visible = self.visible_layer.init_chains(num_samples=num_samples)
        assert visible.shape[0] == targets.shape[0], f"The number of visible units ({visible.shape[0]}) and target labels ({targets.shape[0]}) must be the same."
        hidden = self.hidden_layer.init_chains(num_samples=num_samples)
        hidden_prob = torch.zeros_like(hidden)
        visible_prob = torch.zeros_like(visible)
        
        for _ in range(gibbs_steps):
            hidden, hidden_prob = self.sample_hiddens(visible, targets, beta)
            visible, visible_prob = self.sample_visibles(hidden, beta)
            
        return {
            "visible": visible,
            "hidden": hidden,
            "visible_prob": visible_prob,
            "hidden_prob": hidden_prob,
        }
        
        
    def predict_labels(
        self,
        visible: torch.Tensor,
        beta: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """Predicts the labels given the visible units by computing p(l|v).

        Args:
            visible (torch.Tensor): Visible units tensor of shape (batch_size, num_visibles, ...).
            beta (float, optional): Inverse temperature parameter for sampling. Defaults to 1.0.

        Returns:
            torch.Tensor: Predicted label probabilities tensor of shape (batch_size, num_classes).
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        visible = visible.to(device=device, dtype=dtype)
        
        def _predict_labels_chain(
            v: torch.Tensor,
        ) -> torch.Tensor:
            I_lh = self.visible_layer.mm_left(self.weight_matrix, v).unsqueeze(0) + self.label_matrix  # (num_classes, num_hiddens)
            log_term = self.hidden_layer.nonlinearity(I_lh)
            I_l = log_term.sum(1) + self.lbias  # (num_classes,)
            return torch.softmax(beta * I_l, dim=0)
        
        return torch.vmap(_predict_labels_chain)(visible)
    
    
    def get_patterns(self) -> torch.Tensor:
        """Return the patterns that the RBM associates to each label class.

        Returns:
            torch.Tensor: Patterns tensor of shape (num_classes, num_visibles, ...).
        """
        return torch.movedim(self.weight_matrix @ self.label_matrix.T, -1, 0) # (num_classes, num_visibles, ...)
        
        
    def compute_energy(
        self,
        visible: torch.Tensor,
        hidden: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the energy of the given configurations.

        Args:
            visible (torch.Tensor): Visible units tensor of shape (batch_size, num_visibles, ...).
            hidden (torch.Tensor): Hidden units tensor of shape (batch_size, num_hiddens).
            label (torch.Tensor): Label units tensor of shape (batch_size, num_classes).

        Returns:
            torch.Tensor: Energy tensor of shape (batch_size,).
        """
        energy_visible = self.visible_layer.layer_energy(visible)
        energy_hidden = self.hidden_layer.layer_energy(hidden)
        energy_interaction = - torch.sum(self.visible_layer.mm_left(self.weight_matrix, visible) * hidden, dim=-1)
        energy_label = - torch.sum(hidden @ self.label_matrix.t() * label, dim=-1) - label @ self.lbias
        return energy_visible + energy_hidden + energy_interaction + energy_label
    
    
    def compute_energy_visibles_labels(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the energy of the visible and label configurations by marginalizing over hidden units.

        Args:
            visible (torch.Tensor): Visible units tensor of shape (batch_size, num_visibles, ...).
            label (torch.Tensor): Label units tensor of shape (batch_size, num_classes).

        Returns:
            torch.Tensor: Energy tensor of shape (batch_size,).
        """
        energy_fields = self.visible_layer.layer_energy(visible) - label @ self.lbias
        I_h = self.visible_layer.mm_left(self.weight_matrix, visible) + label @ self.label_matrix
        log_term = self.hidden_layer.nonlinearity(I_h)
        return energy_fields - log_term.sum(dim=-1)
    
    
    def compute_energy_visibles(
        self,
        visible: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the energy of the visible configurations by marginalizing over hidden and label units.

        Args:
            visible (torch.Tensor): Visible units tensor of shape (batch_size, num_visibles, ...).

        Returns:
            torch.Tensor: Energy tensor of shape (batch_size,).
        """
        def _compute_energy_chain(
            v: torch.Tensor,
        ) -> torch.Tensor:
            energy_fields = self.visible_layer.layer_energy(v)
            I_lh = self.visible_layer.mm_left(self.weight_matrix, v).unsqueeze(0) + self.label_matrix # (num_classes, num_hiddens)
            log_term = self.hidden_layer.nonlinearity(I_lh)
            I_l = log_term.sum(1) + self.lbias  # (num_classes,)
            return energy_fields - torch.logsumexp(I_l, dim=0)
        
        return torch.vmap(_compute_energy_chain)(visible)
    
    
    def apply_gradient(
        self,
        data: Dict[str, torch.Tensor],
        chains: Dict[str, torch.Tensor],
        pseudo_count: float = 0.0,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
    ) -> None:
        """Applies the computed gradient to the model parameters.

        Args:
            data (Dict[str, torch.Tensor]): Data batch.
            chains (Dict[str, torch.Tensor]): Chains.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
            lambda_l1 (float, optional): L1 regularization weight. Defaults to 0.0.
            lambda_l2 (float, optional): L2 regularization weight. Defaults to 0.0.
            eta (float, optional): Relative contribution of the label term. Defaults to 1.0.
        """
        # Normalize the weights
        data["weight"] = data["weight"] / data["weight"].sum()
        nchains = len(chains["visible"])
        
        self.visible_layer.apply_gradient(
            x_pos=data["visible"],
            x_neg=chains["visible"],
            weights=data["weight"],
            pseudo_count=pseudo_count,
        )
        self.hidden_layer.apply_gradient(
            x_pos=data["hidden"],
            x_neg=chains["hidden"],
            weights=data["weight"],
            pseudo_count=pseudo_count,
        )
        
        l_pos_mean = get_freq_single_point(data["label"], weights=data["weight"], pseudo_count=pseudo_count)
        l_neg_mean = chains["label"].mean(0)
        grad_lbias = l_pos_mean - l_neg_mean
        grad_weight_matrix = self.visible_layer.outer(self.visible_layer.multiply(data["visible"], data["weight"]), data["hidden"]) - \
                             self.visible_layer.outer(chains["visible"], chains["hidden"]) / nchains
        grad_label_matrix = (data["label"] * data["weight"].view(nchains, 1)).T @ data["hidden"] - \
                            (chains["label"].T @ chains["hidden"]) / nchains
        
        # Regularization
        grad_weight_matrix -= lambda_l1 * self.weight_matrix.sign() + lambda_l2 * self.weight_matrix
        
        self.lbias.grad = grad_lbias
        self.weight_matrix.grad = grad_weight_matrix
        self.label_matrix.grad = grad_label_matrix
