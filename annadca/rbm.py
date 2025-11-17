import os
from typing import Optional, Tuple, Dict
import torch
import copy
from torch.nn import Parameter
from annadca.layers.layer import Layer
from annadca.layers.bernoulli import BernoulliLayer
from annadca.layers.potts import PottsLayer
from annadca.layers.gaussian import GaussianLayer
from annadca.layers.label import CategoricalLabelLayer, LabelNullLayer
from annadca.layers.relu import ReLULayer
from annadca.utils.functions import batched_mm_left, batched_mm_right, batched_outer, multiply, outer
from annadca.utils.stats import get_meanvar, get_mean


def get_rbm(
    visible_type: str,
    hidden_type: str,
    visible_shape: int | Tuple[int, ...] | torch.Size,
    hidden_shape: int | torch.Size,
    num_classes: int,
    continuous_label: Optional[bool] = False,
    **kwargs,
) -> "AnnaRBM":
    available_layers = {
        "potts": PottsLayer,
        "bernoulli": BernoulliLayer,
        "relu": ReLULayer,
        "gaussian": GaussianLayer,
    }
    assert visible_type in available_layers, f"Unknown visible layer type: {visible_type}. Available types are: {list(available_layers.keys())}"
    assert hidden_type in available_layers, f"Unknown hidden layer type: {hidden_type}. Available types are: {list(available_layers.keys())}"
    visible_layer = available_layers[visible_type](shape=visible_shape)
    hidden_layer = available_layers[hidden_type](shape=hidden_shape)
    if continuous_label:
        label_layer = GaussianLayer(shape=(num_classes,))
    else:
        label_layer = CategoricalLabelLayer(shape=(num_classes,))
    return AnnaRBM(
        visible_layer=visible_layer,
        hidden_layer=hidden_layer,
        label_layer=label_layer,
        num_classes=num_classes,
        **kwargs
    )
    

def save_checkpoint(model, chains, optimizer, update, save_dir="checkpoints"):
    """Save model checkpoint with epoch number."""
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Handles prefixes when the model gets compiled
    orig_state_dict = model.state_dict()
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in orig_state_dict.items()}
    
    checkpoint = {
        'update': update,
        'model_state_dict': state_dict,
        "chains": chains,
        'optimizer_state_dict': optimizer.state_dict(),
        "shape": model.shape,
        "num_classes": model.num_classes,
    }

    # Save with update number in filename
    filename = f"model_update_{update:03d}.pt"
    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)


class AnnaRBM(torch.nn.Module):
    def __init__(
        self,
        visible_layer: Layer,
        hidden_layer: Layer,
        label_layer: Layer,
        num_classes: int,
        **kwargs
    ):
        super(AnnaRBM, self).__init__()
        assert isinstance(visible_layer, Layer), "visible_layer must be an instance of Layer."
        assert isinstance(hidden_layer, Layer), "hidden_layer must be an instance of Layer."
        assert not isinstance(hidden_layer, PottsLayer), "hidden_layer cannot be a PottsLayer."
        assert isinstance(num_classes, int) and num_classes >= 1, "num_classes must be a positive integer greater or equal to 1."

        self.visible_layer = visible_layer
        self.hidden_layer = hidden_layer
        self.label_layer = label_layer
        self.kwargs = kwargs
        self.shape = self.visible_layer.shape + self.hidden_layer.shape # (num_visible, num_states, num_hidden)
        self.num_classes = torch.Size((num_classes,))
        
        self.weight_matrix = Parameter(torch.randn(self.shape) * 1e-4, requires_grad=False)
        self.label_matrix = Parameter(torch.randn(self.num_classes + self.hidden_layer.shape) * 1e-4, requires_grad=False)

        # Scaling parameters for the centered RBM
        self.offset_v = Parameter(torch.zeros(self.visible_layer.shape, dtype=torch.float32), requires_grad=False)
        self.offset_h = Parameter(torch.zeros(self.hidden_layer.shape, dtype=torch.float32), requires_grad=False)
        self.offset_l = Parameter(torch.zeros(self.num_classes, dtype=torch.float32), requires_grad=False)
        self.scale_v = Parameter(torch.ones(self.visible_layer.shape, dtype=torch.float32), requires_grad=False)
        self.scale_h = Parameter(torch.ones(self.hidden_layer.shape, dtype=torch.float32), requires_grad=False)
        self.scale_l = Parameter(torch.ones(self.num_classes, dtype=torch.float32), requires_grad=False)

        # compile functions for performances
        self.gibbs_step = torch.compile(self.gibbs_step, mode="default")
        self.sample_visibles = torch.compile(self.sample_visibles, mode="default")
        self.sample_hiddens = torch.compile(self.sample_hiddens, mode="default")
        self.sample_labels = torch.compile(self.sample_labels, mode="default")
        torch.set_float32_matmul_precision('high')
        
    
    def init_params_from_data(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0
    ):
        """Initializes the visible and label layer parameters using the dataset statistics.

        Args:
            visible (torch.Tensor): Input visible data tensor.
            label (torch.Tensor): Input label data tensor.
            weights (Optional[torch.Tensor], optional): Optional weight tensor for the data. Defaults to None.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
        """
        self.visible_layer.init_params_from_data(visible, weights=weights, pseudo_count=pseudo_count)
        self.label_layer.init_params_from_data(label, weights=weights, pseudo_count=pseudo_count)


    def init_chains(
        self,
        num_samples: int,
        data: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Initializes the Markov chains for Gibbs sampling.

        Args:
            num_samples (int): Number of Markov chains to initialize.
            data (torch.Tensor, optional): Empirical data tensor to sample the visible chains from.
            weights (torch.Tensor, optional): Weights for the data samples. If None, uniform weights are assumed.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the chain type and the initialized chains tensor.
        """
        chains = {}
        chains["visible"] = self.visible_layer.init_chains(num_samples=num_samples, data=data, weights=weights, pseudo_count=pseudo_count)
        chains["hidden"] = self.hidden_layer.init_chains(num_samples=num_samples)
        chains["label"] = self.label_layer.init_chains(num_samples=num_samples)
        return chains


    def sample_visibles(
        self,
        hidden: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Samples the visible units given the hidden units.

        Args:
            h (torch.Tensor): Hidden units tensor of shape (batch_size, num_hiddens).
            beta (float, optional): Inverse temperature parameter for sampling. Defaults to 1.0.

        Returns:
            torch.Tensor: Sampled visible units given the hidden units.
        """
        I_v = batched_mm_right(self.weight_matrix, hidden)
        return self.visible_layer.forward(I_v, beta)


    def sample_hiddens(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Samples the hidden units given the visible units.

        Args:
            visible (torch.Tensor): Visible units tensor of shape (batch_size, num_visibles, ...).
            label (torch.Tensor): Label units tensor of shape (batch_size, num_classes).
            beta (float, optional): Inverse temperature parameter for sampling. Defaults to 1.0.

        Returns:
            torch.Tensor: Sampled hidden units given the visible units.
        """
        I_h = batched_mm_left(self.weight_matrix, visible) + batched_mm_left(self.label_matrix, label)
        return self.hidden_layer.forward(I_h, beta)
    
    
    def sample_labels(
        self,
        hidden: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Samples the label units given the hidden units.

        Args:
            hidden (torch.Tensor): Hidden units tensor of shape (batch_size, num_hiddens).
            beta (float, optional): Inverse temperature parameter for sampling. Defaults to 1.0.

        Returns:
            torch.Tensor: Sampled label units given the hidden units.
        """
        I_l = hidden @ self.label_matrix.T
        return self.label_layer.forward(I_l, beta)


    def gibbs_step(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes a forward pass through the RBM, returning the configurations.

        Args:
            visible (torch.Tensor): Visible units tensor of shape (batch_size, num_visibles, ...).
            label (torch.Tensor): Label units tensor of shape (batch_size, num_classes).
            beta (float, optional): Inverse temperature parameter for sampling. Defaults to 1.0.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Visible units, hidden units, label units.

        """
        h = self.sample_hiddens(visible, label, beta)
        v = self.sample_visibles(h, beta)
        l = self.sample_labels(h, beta)
        return (v, h, l)
    
    
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
        hidden = chains_init["hidden"]
        assert visible.shape[0] == label.shape[0], f"The number of visible units ({visible.shape[0]}) and labels ({label.shape[0]}) must be the same."

        for _ in range(gibbs_steps):
            visible, hidden, label = self.gibbs_step(visible, label, beta)
            
        return {
            "visible": visible,
            "hidden": hidden,
            "label": label,
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
        
        for _ in range(gibbs_steps):
            hidden = self.sample_hiddens(visible, targets, beta)
            visible = self.sample_visibles(hidden, beta)

        return {
            "visible": visible,
            "hidden": hidden,
        }
        
        
    def predict_labels(
        self,
        visible: torch.Tensor,
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
        
        energy_vl = torch.vmap(self.compute_energy_visibles_labels, in_dims=(None, 0))(visible, torch.eye(self.num_classes[0], device=device, dtype=dtype)).t()  # (batch_size, num_classes)
        return torch.softmax(- energy_vl, dim=-1)


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
        energy_interaction = - torch.sum(batched_mm_left(self.weight_matrix, visible) * hidden, dim=-1)
        energy_label = - torch.sum(hidden @ self.label_matrix.t() * label, dim=-1) - self.label_layer.layer_energy(label)
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
        energy_fields = self.visible_layer.layer_energy(visible) + self.label_layer.layer_energy(label)
        I_h = batched_mm_left(self.weight_matrix, visible) + batched_mm_left(self.label_matrix, label) # (batch_size, num_hiddens)
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
        assert not isinstance(self.label_layer, GaussianLayer), "The label layer can't be continuous for this function to work"
        all_labels = torch.eye(self.num_classes[0], device=visible.device, dtype=visible.dtype) # (num_classes, num_classes)
        energies_vl = torch.vmap(self.compute_energy_visibles_labels, in_dims=(None, 0))(visible, all_labels).t()  # (batch_size, num_classes)
        return - torch.logsumexp(- energies_vl, dim=-1)


    def compute_energy_hiddens(
        self,
        hidden: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        assert not isinstance(self.label_layer, GaussianLayer), "The label layer can't be continuous for this function to work"
        energy_hidden = self.hidden_layer.layer_energy(hidden)
        I_v = batched_mm_right(self.weight_matrix, hidden) # (batch_size, num_visibles, num_states)
        I_l = batched_mm_right(self.label_matrix, hidden) # (batch_size, num_classes)
        energy_visibles = - self.visible_layer.nonlinearity(I_v).sum(dim=-1)
        energy_labels = - self.label_layer.nonlinearity(I_l)
        return energy_hidden + energy_visibles + energy_labels


    def mean_hidden_activation(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the mean activity of the hidden layer given the activation input tensor: <h | I>.

        Args:
            visible (torch.Tensor): Visible input tensor.
            label (torch.Tensor): Label input tensor.

        Returns:
            torch.Tensor: Average activity of the hidden layer.
        """
        I = batched_mm_left(self.weight_matrix, visible) + batched_mm_left(self.label_matrix, label)
        return self.hidden_layer.meanvar(I)[0]
    
    
    def var_hidden_activation(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the variance of the hidden layer activity given the activation input tensor: <(h - <h | I>)^2 | I>.

        Args:
            visible (torch.Tensor): Visible input tensor.
            label (torch.Tensor): Label input tensor.

        Returns:
            torch.Tensor: Variance of the hidden layer activity.
        """
        I = batched_mm_left(self.weight_matrix, visible) + batched_mm_left(self.label_matrix, label)
        return self.hidden_layer.meanvar(I)[1]
    
    
    def meanvar_hidden_activation(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the mean and variance of the hidden layer activity given the activation input tensor: <h | I>, <(h - <h | I>)^2 | I>.

        Args:
            visible (torch.Tensor): Visible input tensor.
            label (torch.Tensor): Label input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and variance of the hidden layer activity.
        """
        I = batched_mm_left(self.weight_matrix, visible) + batched_mm_left(self.label_matrix, label)
        return self.hidden_layer.meanvar(I)


    def regularize(
        self,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        l1l2_strength: float = 0.0,
    ):
        if l1_strength > 0:
            if self.weight_matrix.grad is not None:
                self.weight_matrix.grad -= l1_strength * self.weight_matrix.sign()
        if l2_strength > 0:
            if self.weight_matrix.grad is not None:
                self.weight_matrix.grad -= l2_strength * self.weight_matrix
        if l1l2_strength > 0:
            if self.weight_matrix.grad is not None:
                if len(self.weight_matrix.shape) == 3:
                    self.weight_matrix.grad -= l1l2_strength * self.weight_matrix.abs().mean(dim=(1, 2), keepdim=True) * self.weight_matrix.sign()
                else:
                    self.weight_matrix.grad -= l1l2_strength * self.weight_matrix.abs().mean(1, keepdim=True) * self.weight_matrix.sign()
                    

    def apply_gradient_old(
        self,
        data: Dict[str, torch.Tensor],
        chains: Dict[str, torch.Tensor],
        pseudo_count: float = 0.0,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        l1l2_strength: float = 0.0,
        standardize: bool = False,
    ) -> None:
        """Applies the computed gradient to the model parameters.

        Args:
            data (Dict[str, torch.Tensor]): Data batch.
            chains (Dict[str, torch.Tensor]): Chains.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
            l1_strength (float, optional): L1 regularization weight. Defaults to 0.0.
            l2_strength (float, optional): L2 regularization weight. Defaults to 0.0.
            l1l2_strength (float, optional): L1L2 regularization weight. Defaults to 0.0.
            standardize (bool, optional): Whether to apply gradient standardization. Defaults to False.
        """
        # Normalize the weights
        data["weight"] = data["weight"] / data["weight"].sum()
        nchains = len(chains["visible"])
        # Hidden distribution means and variances
        h_pos, var_h_pos = self.meanvar_hidden_activation(data["visible"], data["label"])
        h_neg, var_h_neg = self.meanvar_hidden_activation(chains["visible"], chains["label"])
        
        # compute offsets
        self.offset_v, self.scale_v = get_meanvar(data["visible"], weights=data["weight"], pseudo_count=pseudo_count)
        self.offset_l, self.scale_l = get_meanvar(data["label"], weights=data["weight"], pseudo_count=pseudo_count)
        self.offset_h, self.scale_h = get_meanvar(h_pos, weights=data["weight"], pseudo_count=pseudo_count)

        if standardize:
            v_pos = data["visible"] - self.offset_v
            l_pos = data["label"] - self.offset_l
            v_neg = chains["visible"] - self.offset_v
            l_neg = chains["label"] - self.offset_l
            h_pos = h_pos - self.offset_h
            h_neg = h_neg - self.offset_h
        
        else:
            v_pos, l_pos = data["visible"], data["label"]
            v_neg, l_neg = chains["visible"], chains["label"]

        self.visible_layer.apply_gradient_visible(
            x_pos=v_pos,
            x_neg=v_neg,
            weights=data["weight"],
            pseudo_count=pseudo_count,
        )
        
        self.label_layer.apply_gradient_visible(
            x_pos=l_pos,
            x_neg=l_neg,
            weights=data["weight"],
            pseudo_count=pseudo_count,
        )

        self.hidden_layer.apply_gradient_hidden(
            mean_h_pos=h_pos,
            mean_h_neg=h_neg,
            var_h_pos=torch.ones_like(h_pos),
            var_h_neg=torch.ones_like(h_neg),
            weights=data["weight"],
            pseudo_count=pseudo_count,
        )

        grad_weight_matrix = batched_outer(multiply(v_pos, data["weight"]), h_pos) - batched_outer(v_neg, h_neg) / nchains
        grad_label_matrix = batched_outer(multiply(l_pos, data["weight"]), h_pos) - batched_outer(l_neg, h_neg) / nchains

        if standardize:
            self.visible_layer.standardize_gradient_visible(dW=grad_weight_matrix, c_h=self.offset_h)
            self.label_layer.standardize_gradient_visible(dW=grad_label_matrix, c_h=self.offset_h)
            self.hidden_layer.standardize_gradient_hidden(
                dW=grad_weight_matrix,
                dL=grad_label_matrix,
                c_v=self.offset_v,
                c_l=self.offset_l,
                c_h=self.offset_h,
            )

        self.weight_matrix.grad = grad_weight_matrix
        self.label_matrix.grad = grad_label_matrix
        # Regularization
        self.regularize(l1_strength, l2_strength, l1l2_strength)
        
        
    def apply_gradient(
        self,
        data: Dict[str, torch.Tensor],
        chains: Dict[str, torch.Tensor],
        pseudo_count: float = 0.0,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        l1l2_strength: float = 0.0,
        standardize: bool = False,
    ) -> None:
        """Applies the computed gradient to the model parameters.

        Args:
            data (Dict[str, torch.Tensor]): Data batch.
            chains (Dict[str, torch.Tensor]): Chains.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
            l1_strength (float, optional): L1 regularization weight. Defaults to 0.0.
            l2_strength (float, optional): L2 regularization weight. Defaults to 0.0.
            l1l2_strength (float, optional): L1L2 regularization weight. Defaults to 0.0.
            standardize (bool, optional): Whether to apply gradient standardization. Defaults to False.
        """
        # Normalize the weights
        data["weight"] = data["weight"] / data["weight"].sum()
        nchains = len(chains["visible"])
        
        # compute offsets
        offset_v, scale_v = get_meanvar(data["visible"], weights=data["weight"], pseudo_count=pseudo_count)
        offset_l, scale_l = get_meanvar(data["label"], weights=data["weight"], pseudo_count=pseudo_count)

        if standardize:
            v_pos = (data["visible"] - offset_v) / scale_v
            l_pos = (data["label"] - offset_l) / scale_l
            v_neg = (chains["visible"] - offset_v) / scale_v
            l_neg = (chains["label"] - offset_l) / scale_l

        else:
            v_pos, l_pos = data["visible"], data["label"]
            v_neg, l_neg = chains["visible"], chains["label"]
            
        # Hidden distribution means and variances
        h_pos, var_h_pos = self.meanvar_hidden_activation(v_pos, l_pos)
        h_neg, var_h_neg = self.meanvar_hidden_activation(v_neg, l_neg)
        
        offset_h, scale_h2 = get_meanvar(h_pos, weights=data["weight"], pseudo_count=pseudo_count)
        # Apply law of total variance
        scale_h = torch.sqrt(scale_h2 + var_h_pos.mean(dim=0))
        
        if standardize:
            h_pos = (h_pos - offset_h) / scale_h
            h_neg = (h_neg - offset_h) / scale_h

        self.visible_layer.apply_gradient_visible(
            x_pos=v_pos,
            x_neg=v_neg,
            weights=data["weight"],
            pseudo_count=pseudo_count,
        )
        
        self.label_layer.apply_gradient_visible(
            x_pos=l_pos,
            x_neg=l_neg,
            weights=data["weight"],
            pseudo_count=pseudo_count,
        )

        self.hidden_layer.apply_gradient_hidden(
            mean_h_pos=h_pos,
            mean_h_neg=h_neg,
            var_h_pos=var_h_pos,
            var_h_neg=var_h_neg,
            weights=data["weight"],
            pseudo_count=pseudo_count,
        )

        grad_weight_matrix = batched_outer(multiply(v_pos, data["weight"]), h_pos) - batched_outer(v_neg, h_neg) / nchains
        grad_label_matrix = batched_outer(multiply(l_pos, data["weight"]), h_pos) - batched_outer(l_neg, h_neg) / nchains

        self.weight_matrix.grad = grad_weight_matrix
        self.label_matrix.grad = grad_label_matrix

        # standardize params before applying gradients
        if standardize:
            self.unstandardize_rbm(
                offset_v=offset_v,
                scale_v=scale_v,
                offset_h=offset_h,
                scale_h=scale_h,
                offset_l=offset_l,
                scale_l=scale_l,
            )
        # Regularization
        self.regularize(l1_strength, l2_strength, l1l2_strength)
        
        
    def standardize_rbm(
        self,
        offset_v: torch.Tensor,
        scale_v: torch.Tensor,
        offset_h: torch.Tensor,
        scale_h: torch.Tensor,
        offset_l: torch.Tensor,
        scale_l: torch.Tensor,
    ):
        """Standarizes the RBM parameters, mapping the classical RBM to the standardized one.
        
        Args:
            offset_v (torch.Tensor): Visible layer offset.
            scale_v (torch.Tensor): Visible layer scale.
            offset_h (torch.Tensor): Hidden layer offset.
            scale_h (torch.Tensor): Hidden layer scale.
            offset_l (torch.Tensor): Label layer offset.
            scale_l (torch.Tensor): Label layer scale.
        """
        self.visible_layer.standardize_params_visible(scale_v=self.scale_v, scale_h=self.scale_h, offset_h=self.offset_h, W=self.weight_matrix)
        self.hidden_layer.standardize_params_hidden(offset_h=self.offset_h, scale_h=self.scale_h, offset_v=self.offset_v, scale_v=self.scale_v, offset_l=self.offset_l, scale_l=self.scale_l, W=self.weight_matrix, L=self.label_matrix)
        self.label_layer.standardize_params_visible(scale_v=self.scale_l, scale_h=self.scale_h, offset_h=self.offset_h, W=self.label_matrix)
        self.weight_matrix.copy_(self.weight_matrix / outer(self.scale_v, self.scale_h))
        self.label_matrix.copy_(self.label_matrix / outer(self.scale_l, self.scale_h))
        # Update offsets and scales
        self.offset_v.copy_(offset_v)
        self.scale_v.copy_(scale_v + 1e-10)
        self.offset_h.copy_(offset_h)
        self.scale_h.copy_(scale_h + 1e-10)
        self.offset_l.copy_(offset_l)
        self.scale_l.copy_(scale_l + 1e-10)

    def unstandardize_rbm(
        self,
        
    ):
        """Unstandarizes the RBM parameters, mapping the standardized RBM to the classical one.
        """
        self.weight_matrix.copy_(self.weight_matrix * outer(self.scale_v, self.scale_h))
        self.label_matrix.copy_(self.label_matrix * outer(self.scale_l, self.scale_h))
        self.visible_layer.unstandardize_params_visible(scale_v=self.scale_v, scale_h=self.scale_h, offset_h=self.offset_h, W=self.weight_matrix)
        self.hidden_layer.unstandardize_params_hidden(offset_h=self.offset_h, scale_h=self.scale_h, offset_v=self.offset_v, scale_v=self.scale_v, offset_l=self.offset_l, scale_l=self.scale_l, W=self.weight_matrix, L=self.label_matrix)
        self.label_layer.unstandardize_params_visible(scale_v=self.scale_l, scale_h=self.scale_h, offset_h=self.offset_h, W=self.label_matrix)



    def forward(
        self,
        data_batch: Dict[str, torch.Tensor],
        chains: Dict[str, torch.Tensor],
        gibbs_steps: int,
        pseudo_count: float = 0.0,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        l1l2_strength: float = 0.0,
        standardize: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Computes the gradient of the parameters of the model and the Markov chains using the Persistent Contrastive Divergence algorithm.
    
        Args:
            data_batch (Dict[str, torch.Tensor]): Batch of data.
            chains (Dict[str, torch.Tensor]): Persistent chains.
            gibbs_steps (int): Number of Alternating Gibbs Sampling steps.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
            l1_strength (float, optional): L1 regularization weight. Defaults to 0.0.
            l2_strength (float, optional): L2 regularization weight. Defaults to 0.0.
            l1l2_strength (float, optional): L1L2 regularization weight. Defaults to 0.0.
            standardize (bool, optional): Whether to standardize the gradients. Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: Updated chains.
        """
        
        # Compute the gradient of the Log-Likelihood
        self.apply_gradient(
            data=data_batch,
            chains=chains,
            pseudo_count=pseudo_count,
            l1_strength=l1_strength,
            l2_strength=l2_strength,
            l1l2_strength=l1l2_strength,
            standardize=standardize,
        )
        
        # THE GRADIENT IS APPLIED OUTSIDE! NEED TO 
        # 1) APPLY GRADIENT
        # 2) UNSTANDARDIZE RBM
        # 3) SAMPLE NEW CHAINS
        # 4) STANDARDIZE RBM AGAIN

        # Update the persistent chains
        chains = self.sample(
            gibbs_steps=gibbs_steps,
            visible=chains["visible"],
            label=chains["label"],
            beta=1.0)
        return chains
    
    
    def get_biased_model(
        self,
        gen_strength: float,
        label_strength: float,
        wt_strength: float,
        wt: torch.Tensor,
    ) -> "AnnaRBM":
        wt = wt.to(device=self.weight_matrix.device, dtype=self.weight_matrix.dtype)
        biased_rbm = AnnaRBM(
            visible_layer=copy.deepcopy(self.visible_layer),
            hidden_layer=copy.deepcopy(self.hidden_layer),
            label_layer=copy.deepcopy(self.label_layer),
            num_classes=self.num_classes[0],
        )
        biased_rbm.to(device=self.weight_matrix.device, dtype=self.weight_matrix.dtype)
        biased_rbm.weight_matrix.data.copy_(self.weight_matrix * gen_strength)
        biased_rbm.label_matrix.data.copy_(self.label_matrix * label_strength)
        for k, v in biased_rbm.visible_layer.named_parameters():
            v.data.copy_(self.visible_layer.state_dict()[k] * gen_strength)
            if "bias" in k:
                v.data.copy_(v.data + wt_strength * wt)
        for k, v in biased_rbm.hidden_layer.named_parameters():
            v.data.copy_(self.hidden_layer.state_dict()[k] * gen_strength)
        for k, v in biased_rbm.label_layer.named_parameters():
            v.data.copy_(self.label_layer.state_dict()[k] * label_strength)
        return biased_rbm


    def __repr__(self) -> str:
        return f"AnnaRBM(visible_layer={self.visible_layer}, hidden_layer={self.hidden_layer}, num_classes={self.num_classes}, shape={self.shape})"
