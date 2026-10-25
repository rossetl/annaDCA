import os
from typing import Optional, Tuple, Dict, Any
import torch
import copy
from torch.nn import Parameter
from annadca.layers.layer import Layer, LAYER_REGISTRY, build_layer
from annadca.layers.gaussian import GaussianLayer
from annadca.layers.label import CategoricalLabelLayer, LabelNullLayer
from annadca.utils.functions import mm_left, batched_mm_left, batched_mm_right, batched_outer, outer
from annadca.utils.stats import get_meanvar, get_mean


VISIBLE_TYPES = [name for name, cls in LAYER_REGISTRY.items() if cls.can_be_visible and name not in ("categorical", "null")]
HIDDEN_TYPES = [name for name, cls in LAYER_REGISTRY.items() if cls.can_be_hidden]
LABEL_TYPES = {"categorical": CategoricalLabelLayer, "gaussian": GaussianLayer, "null": LabelNullLayer}


def get_rbm(
    visible_type: str,
    hidden_type: str,
    visible_shape: int | Tuple[int, ...] | torch.Size,
    hidden_shape: int | torch.Size,
    num_classes: int,
    continuous_label: bool = False,
    label_type: Optional[str] = None,
) -> "AnnaRBM":
    """Builds an AnnaRBM model from the names of the layers.

    Args:
        visible_type (str): Type of the visible layer. One of 'potts', 'bernoulli', 'gaussian', 'relu'.
        hidden_type (str): Type of the hidden layer. One of 'bernoulli', 'gaussian', 'relu'.
        visible_shape (int | Tuple[int, ...] | torch.Size): Shape of the visible layer, e.g. (L, q) for a Potts layer.
        hidden_shape (int | torch.Size): Number of hidden units.
        num_classes (int): Number of label classes (1 for continuous labels).
        continuous_label (bool, optional): If True, the labels are modeled with a Gaussian layer. Defaults to False.
        label_type (str, optional): Type of the label layer, one of 'categorical', 'gaussian', 'null'. Overrides `continuous_label`.

    Returns:
        AnnaRBM: The model.
    """
    assert visible_type in VISIBLE_TYPES, f"Unknown visible layer type: {visible_type}. Available types are: {VISIBLE_TYPES}"
    assert hidden_type in HIDDEN_TYPES, f"Unknown hidden layer type: {hidden_type}. Available types are: {HIDDEN_TYPES}"
    if label_type is None:
        label_type = "gaussian" if continuous_label else "categorical"
    assert label_type in LABEL_TYPES, f"Unknown label layer type: {label_type}. Available types are: {list(LABEL_TYPES)}"
    return AnnaRBM(
        visible_layer=LAYER_REGISTRY[visible_type](shape=visible_shape),
        hidden_layer=LAYER_REGISTRY[hidden_type](shape=hidden_shape),
        label_layer=LABEL_TYPES[label_type](shape=(num_classes,)),
    )


def save_checkpoint(
    model: "AnnaRBM",
    chains: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    update: int,
    save_dir: str = "checkpoints",
    scheduler: Optional[Any] = None,
    metadata: Optional[dict] = None,
):
    """Saves a self-describing checkpoint, that can be loaded with `load_rbm`.

    Args:
        model (AnnaRBM): The model.
        chains (Dict[str, torch.Tensor]): Persistent chains, stored in compressed form.
        optimizer (torch.optim.Optimizer): Optimizer.
        update (int): Number of gradient updates performed.
        save_dir (str, optional): Directory of the checkpoints. Defaults to "checkpoints".
        scheduler (optional): Learning rate scheduler.
        metadata (dict, optional): Additional information, e.g. the alphabet and the label names.
    """
    os.makedirs(save_dir, exist_ok=True)
    # Handles prefixes when the model gets compiled
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()}
    checkpoint = {
        "update": update,
        "config": model.config(),
        "model_state_dict": state_dict,
        "chains": model.compress_chains(chains),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "metadata": metadata or {},
        "rng_state": torch.get_rng_state(),
    }
    torch.save(checkpoint, os.path.join(save_dir, f"model_update_{update:03d}.pt"))


def load_rbm(
    filepath: str,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple["AnnaRBM", Dict[str, Any]]:
    """Loads a model saved with `save_checkpoint`.

    Args:
        filepath (str): Path to the checkpoint.
        device (torch.device | str, optional): Device of the model. Defaults to "cpu".
        dtype (torch.dtype, optional): Data type of the model. Defaults to torch.float32.

    Returns:
        Tuple[AnnaRBM, Dict[str, Any]]: The model and the checkpoint dictionary, with the chains decompressed.
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    if "config" not in checkpoint:
        raise ValueError(
            f"The checkpoint {filepath} does not contain the model configuration (it was produced by an older version). "
            "Build the model with `get_rbm` and load `checkpoint['model_state_dict']` with `load_state_dict`."
        )
    rbm = AnnaRBM.from_config(checkpoint["config"])
    rbm.load_state_dict(checkpoint["model_state_dict"])
    rbm.to(device=device, dtype=dtype)
    checkpoint["chains"] = rbm.decompress_chains(checkpoint["chains"])
    return rbm, checkpoint


def _sum_per_sample(x: torch.Tensor) -> torch.Tensor:
    """Sums all the dimensions of x except the first one (batch dimension)."""
    return x.reshape(len(x), -1).sum(dim=1)


class AnnaRBM(torch.nn.Module):
    """RBM with an additional label layer coupled to the hidden layer. The energy is

        E(v, h, l) = U_v(v) + U_h(h) + U_l(l) - v^T W h - l^T L h

    where U_v, U_h, U_l are the potentials defined by the visible, hidden and label layers,
    W is the weight matrix and L is the label matrix.

    The parameters are always stored in the original (non-standardized) space. Centering and standardization only
    act on the gradient, see `apply_gradient`.
    """
    def __init__(
        self,
        visible_layer: Layer,
        hidden_layer: Layer,
        label_layer: Layer,
        num_classes: Optional[int] = None,
    ):
        super(AnnaRBM, self).__init__()
        assert isinstance(visible_layer, Layer), "visible_layer must be an instance of Layer."
        assert isinstance(hidden_layer, Layer), "hidden_layer must be an instance of Layer."
        assert isinstance(label_layer, Layer), "label_layer must be an instance of Layer."
        assert visible_layer.can_be_visible, f"{visible_layer.__class__.__name__} cannot be used as visible layer."
        assert hidden_layer.can_be_hidden, f"{hidden_layer.__class__.__name__} cannot be used as hidden layer."
        assert label_layer.can_be_visible, f"{label_layer.__class__.__name__} cannot be used as label layer."
        assert len(hidden_layer.shape) == 1, "The hidden layer must be one-dimensional."
        assert len(label_layer.shape) == 1, "The label layer must be one-dimensional."
        if num_classes is not None:
            assert num_classes == label_layer.shape[0], f"num_classes ({num_classes}) does not match the label layer shape ({label_layer.shape})."

        self.visible_layer = visible_layer
        self.hidden_layer = hidden_layer
        self.label_layer = label_layer
        self.shape = self.visible_layer.shape + self.hidden_layer.shape # (num_visible, [num_states,] num_hidden)
        self.num_classes = self.label_layer.shape[0]

        self.weight_matrix = Parameter(torch.randn(self.shape) * 1e-4, requires_grad=False)
        self.label_matrix = Parameter(torch.randn(self.label_layer.shape + self.hidden_layer.shape) * 1e-4, requires_grad=False)

        # Offsets and scales used in the last standardized gradient update. They are not trained.
        self.register_buffer("offset_v", torch.zeros(self.visible_layer.shape))
        self.register_buffer("offset_h", torch.zeros(self.hidden_layer.shape))
        self.register_buffer("offset_l", torch.zeros(self.label_layer.shape))
        self.register_buffer("scale_v", torch.ones(self.visible_layer.shape))
        self.register_buffer("scale_h", torch.ones(self.hidden_layer.shape))
        self.register_buffer("scale_l", torch.ones(self.label_layer.shape))


    def config(self) -> dict:
        """Configuration of the model: the configurations of its layers."""
        return {
            "visible_layer": self.visible_layer.config(),
            "hidden_layer": self.hidden_layer.config(),
            "label_layer": self.label_layer.config(),
        }


    @classmethod
    def from_config(cls, config: dict) -> "AnnaRBM":
        """Builds a model (with freshly initialized parameters) from its configuration."""
        return cls(
            visible_layer=build_layer(config["visible_layer"]),
            hidden_layer=build_layer(config["hidden_layer"]),
            label_layer=build_layer(config["label_layer"]),
        )


    def compress_chains(self, chains: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compact representation of the chains (e.g. state indices instead of one-hot vectors), used for saving."""
        layers = {"visible": self.visible_layer, "hidden": self.hidden_layer, "label": self.label_layer}
        return {k: layers[k].compress(v) if k in layers else v for k, v in chains.items()}


    def decompress_chains(self, chains: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inverse of `compress_chains`. Also accepts uncompressed chains."""
        layers = {"visible": self.visible_layer, "hidden": self.hidden_layer, "label": self.label_layer}
        return {k: layers[k].decompress(v) if k in layers else v for k, v in chains.items()}


    def compile_sampling(self, mode: str = "default"):
        """Compiles the sampling functions with torch.compile to speed up Gibbs sampling."""
        self.gibbs_step = torch.compile(self.gibbs_step, mode=mode)
        self.sample_visibles = torch.compile(self.sample_visibles, mode=mode)
        self.sample_hiddens = torch.compile(self.sample_hiddens, mode=mode)
        self.sample_labels = torch.compile(self.sample_labels, mode=mode)
        return self


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
        label: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        pseudo_count: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Initializes the Markov chains for Gibbs sampling.

        Args:
            num_samples (int): Number of Markov chains to initialize.
            data (torch.Tensor, optional): Empirical visible data tensor to sample the visible chains from.
            label (torch.Tensor, optional): Empirical label data tensor to sample the label chains from.
            weights (torch.Tensor, optional): Weights for the data samples. If None, uniform weights are assumed.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the chain type and the initialized chains tensor.
        """
        return {
            "visible": self.visible_layer.init_chains(num_samples=num_samples, data=data, weights=weights, pseudo_count=pseudo_count),
            "hidden": self.hidden_layer.init_chains(num_samples=num_samples),
            "label": self.label_layer.init_chains(num_samples=num_samples, data=label, weights=weights, pseudo_count=pseudo_count),
        }


    def input_hiddens(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """Input field to the hidden layer: W^T v + L^T l. Shape (batch_size, num_hiddens)."""
        return batched_mm_left(self.weight_matrix, visible) + label @ self.label_matrix


    def input_visibles(self, hidden: torch.Tensor) -> torch.Tensor:
        """Input field to the visible layer: W h. Shape (batch_size, num_visibles, ...)."""
        return batched_mm_right(self.weight_matrix, hidden)


    def input_labels(self, hidden: torch.Tensor) -> torch.Tensor:
        """Input field to the label layer: L h. Shape (batch_size, num_classes)."""
        return hidden @ self.label_matrix.T


    def sample_visibles(
        self,
        hidden: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Samples the visible units given the hidden units.

        Args:
            hidden (torch.Tensor): Hidden units tensor of shape (batch_size, num_hiddens).
            beta (float, optional): Inverse temperature parameter for sampling. Defaults to 1.0.

        Returns:
            torch.Tensor: Sampled visible units given the hidden units.
        """
        return self.visible_layer.sample(self.input_visibles(hidden), beta)


    def sample_hiddens(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Samples the hidden units given the visible and label units.

        Args:
            visible (torch.Tensor): Visible units tensor of shape (batch_size, num_visibles, ...).
            label (torch.Tensor): Label units tensor of shape (batch_size, num_classes).
            beta (float, optional): Inverse temperature parameter for sampling. Defaults to 1.0.

        Returns:
            torch.Tensor: Sampled hidden units.
        """
        return self.hidden_layer.sample(self.input_hiddens(visible, label), beta)


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
        return self.label_layer.sample(self.input_labels(hidden), beta)


    def gibbs_step(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs one step of alternating Gibbs sampling: h ~ p(h | v, l), then v ~ p(v | h) and l ~ p(l | h).

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

        Returns:
            Dict[str, torch.Tensor]: Visible, hidden and label configurations.
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
            targets (torch.Tensor): Target label units tensor of shape (batch_size, num_classes), one-hot for categorical labels.
            visible (torch.Tensor | None, optional): Initial visible units tensor. If None, initializes randomly. Defaults to None.
            beta (float, optional): Inverse temperature parameter for sampling. Defaults to 1.0.

        Returns:
            Dict[str, torch.Tensor]: Visible and hidden configurations.
        """
        targets = targets.to(self.weight_matrix)
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


    def _energy_visibles_all_labels(self, visible: torch.Tensor) -> torch.Tensor:
        """Computes the energy E(v, l) for all the categorical labels l, marginalizing over the hidden units.

        Args:
            visible (torch.Tensor): Visible units tensor of shape (batch_size, num_visibles, ...).

        Returns:
            torch.Tensor: Energy tensor of shape (batch_size, num_classes).
        """
        assert isinstance(self.label_layer, CategoricalLabelLayer), "This function requires categorical labels."
        I_v = batched_mm_left(self.weight_matrix, visible) # (batch_size, num_hiddens)
        I_h = I_v.unsqueeze(1) + self.label_matrix.unsqueeze(0) # (batch_size, num_classes, num_hiddens)
        log_term = self.hidden_layer.nonlinearity(I_h).sum(dim=-1) # (batch_size, num_classes)
        # The energy of the one-hot label e_c is -bias_c
        return self.visible_layer.layer_energy(visible).unsqueeze(1) - self.label_layer.bias.unsqueeze(0) - log_term


    def predict_labels(
        self,
        visible: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Predicts the labels given the visible units by computing p(l|v).

        Args:
            visible (torch.Tensor): Visible units tensor of shape (batch_size, num_visibles, ...).

        Returns:
            torch.Tensor: Predicted label probabilities tensor of shape (batch_size, num_classes).
        """
        visible = visible.to(self.weight_matrix)
        return torch.softmax(- self._energy_visibles_all_labels(visible), dim=-1)


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
        energy_fields = self.visible_layer.layer_energy(visible) + self.hidden_layer.layer_energy(hidden) + self.label_layer.layer_energy(label)
        energy_interaction = - torch.sum(self.input_hiddens(visible, label) * hidden, dim=-1)
        return energy_fields + energy_interaction


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
        log_term = self.hidden_layer.nonlinearity(self.input_hiddens(visible, label))
        return energy_fields - log_term.sum(dim=-1)


    def compute_energy_visibles(
        self,
        visible: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the energy of the visible configurations by marginalizing over hidden and label units.
        Requires categorical labels.

        Args:
            visible (torch.Tensor): Visible units tensor of shape (batch_size, num_visibles, ...).

        Returns:
            torch.Tensor: Energy tensor of shape (batch_size,).
        """
        return - torch.logsumexp(- self._energy_visibles_all_labels(visible), dim=-1)


    def compute_energy_hiddens(
        self,
        hidden: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Computes the energy of the hidden configurations by marginalizing over visible and label units.

        Args:
            hidden (torch.Tensor): Hidden units tensor of shape (batch_size, num_hiddens).

        Returns:
            torch.Tensor: Energy tensor of shape (batch_size,).
        """
        energy_hidden = self.hidden_layer.layer_energy(hidden)
        energy_visibles = - _sum_per_sample(self.visible_layer.nonlinearity(self.input_visibles(hidden)))
        energy_labels = - _sum_per_sample(self.label_layer.nonlinearity(self.input_labels(hidden)))
        return energy_hidden + energy_visibles + energy_labels


    def mean_hidden_activation(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the mean activity of the hidden layer given the visible and label units: <h | v, l>."""
        return self.hidden_layer.meanvar(self.input_hiddens(visible, label))[0]


    def var_hidden_activation(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the variance of the hidden layer activity given the visible and label units: <(h - <h | v, l>)^2 | v, l>."""
        return self.hidden_layer.meanvar(self.input_hiddens(visible, label))[1]


    def meanvar_hidden_activation(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the mean and variance of the hidden layer activity given the visible and label units."""
        return self.hidden_layer.meanvar(self.input_hiddens(visible, label))


    def meanvar_hidden_activation_missing_labels(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
        missing: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Positive-phase statistics when some categorical labels are missing. A missing label is summed over with its
        posterior p(l | v), which gives the exact gradient of the log-likelihood log p(v) of the unannotated sequences.

        For the missing rows, the label is replaced by its posterior, and the hidden mean and variance are those of the
        mixture sum_c p(c | v) p(h | v, c). Since E[l h] differs from E[l] E[h] under the posterior, the label-hidden
        covariance of each sample is also returned, to be added to the gradient of the label matrix.

        Args:
            visible (torch.Tensor): Visible units tensor of shape (batch_size, num_visibles, ...).
            label (torch.Tensor): Label units tensor of shape (batch_size, num_classes).
            missing (torch.Tensor): Boolean mask of shape (batch_size,) of the samples with a missing label.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Hidden means and variances (batch_size, num_hiddens),
                labels with the posterior in place of the missing ones (batch_size, num_classes), and label-hidden covariance
                of the missing samples (num_missing, num_classes, num_hiddens).
        """
        assert isinstance(self.label_layer, CategoricalLabelLayer), "Marginalizing missing labels requires categorical labels."
        mean_h, var_h = self.meanvar_hidden_activation(visible, label)
        label = label.clone()
        v_missing = visible[missing]
        posterior = self.predict_labels(v_missing) # (num_missing, num_classes)
        I_h = batched_mm_left(self.weight_matrix, v_missing).unsqueeze(1) + self.label_matrix.unsqueeze(0)
        mean_hc, var_hc = self.hidden_layer.meanvar(I_h) # (num_missing, num_classes, num_hiddens)
        p = posterior.unsqueeze(-1)
        mixture_mean = (p * mean_hc).sum(dim=1)
        mixture_var = (p * (var_hc + mean_hc**2)).sum(dim=1) - mixture_mean**2
        mean_h[missing] = mixture_mean
        var_h[missing] = mixture_var.clamp_min(0.0)
        label[missing] = posterior
        # sum_c p_c (e_c - p) (<h|v,c> - <h|v>)^T = p_c (<h|v,c> - <h|v>) for the row c
        covariance = p * (mean_hc - mixture_mean.unsqueeze(1))
        return mean_h, var_h, label, covariance


    def regularize(
        self,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        l1l2_strength: float = 0.0,
        l1_label_strength: float = 0.0,
        l2_label_strength: float = 0.0,
    ):
        """Adds the gradient of the regularization terms on the weight and label matrices. Must be called after the gradient has been computed.

        Args:
            l1_strength (float, optional): L1 penalty: l1 * sum |W|. Defaults to 0.0.
            l2_strength (float, optional): L2 penalty: l2/2 * sum W^2. Defaults to 0.0.
            l1l2_strength (float, optional): L1^2 penalty: l1l2/(2N) * sum_mu (sum_{i(,a)} |W_{i(a)mu}|)^2, with N the number of
                visible entries, which favors sparse hidden units. Defaults to 0.0.
            l1_label_strength (float, optional): L1 penalty on the label matrix. Defaults to 0.0.
            l2_label_strength (float, optional): L2 penalty on the label matrix. Defaults to 0.0.
        """
        if self.label_matrix.grad is not None:
            if l1_label_strength > 0:
                self.label_matrix.grad -= l1_label_strength * self.label_matrix.sign()
            if l2_label_strength > 0:
                self.label_matrix.grad -= l2_label_strength * self.label_matrix
        if self.weight_matrix.grad is None:
            return
        if l1_strength > 0:
            self.weight_matrix.grad -= l1_strength * self.weight_matrix.sign()
        if l2_strength > 0:
            self.weight_matrix.grad -= l2_strength * self.weight_matrix
        if l1l2_strength > 0:
            visible_dims = tuple(range(self.weight_matrix.dim() - 1))
            self.weight_matrix.grad -= l1l2_strength * self.weight_matrix.abs().mean(dim=visible_dims, keepdim=True) * self.weight_matrix.sign()


    def apply_gradient(
        self,
        data: Dict[str, torch.Tensor],
        chains: Dict[str, torch.Tensor],
        pseudo_count: float = 0.0,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        l1l2_strength: float = 0.0,
        standardize: bool = True,
        centered: bool = True,
        missing_labels: str = "posterior",
        l1_label_strength: float = 0.0,
        l2_label_strength: float = 0.0,
    ) -> None:
        """Computes the gradient of the log-likelihood and stores it in the `.grad` attribute of the parameters.

        The centered/standardized gradient is the gradient of the log-likelihood with respect to the reparametrized model

            E = U_v(v) + U_h(h) + U_l(l) - ((v - o_v)/s_v)^T W' ((h - o_h)/s_h) - ((l - o_l)/s_l)^T L' ((h - o_h)/s_h),

        where the offsets o and scales s are the data means and standard deviations of the units, mapped back to the
        original parameters W = W' / (s_v s_h), L = L' / (s_l s_h), and to the biases. Only the hidden and label units are
        rescaled (s_v = 1), since the standard deviation of rarely observed visible states is close to zero. The parameters
        stay in the original space, so the model can be sampled and evaluated without any transformation.

        Missing categorical labels are the all-zero rows of `data["label"]`. With `missing_labels="posterior"`, they are
        summed over with the model posterior p(l | v) (see `meanvar_hidden_activation_missing_labels`); with
        `missing_labels="zero"`, the all-zero vector is used as label.

        For discrete layers, the pseudo count regularizes the data statistics as the mixture (1 - pseudo_count) * data
        + pseudo_count * uniform distribution, in which the hidden units of the uniform component are distributed as in
        the data. This is applied consistently to the bias and to the weight gradients.

        Args:
            data (Dict[str, torch.Tensor]): Data batch.
            chains (Dict[str, torch.Tensor]): Chains.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
            l1_strength (float, optional): L1 regularization weight. Defaults to 0.0.
            l2_strength (float, optional): L2 regularization weight. Defaults to 0.0.
            l1l2_strength (float, optional): L1L2 regularization weight. Defaults to 0.0.
            standardize (bool, optional): Whether to rescale the hidden and label units by their standard deviation (implies centering). Defaults to True.
            centered (bool, optional): Whether to center the units by their mean. Defaults to True.
            missing_labels (str, optional): How to treat the missing categorical labels: 'posterior' or 'zero'. Defaults to 'posterior'.
            l1_label_strength (float, optional): L1 regularization weight of the label matrix. Defaults to 0.0.
            l2_label_strength (float, optional): L2 regularization weight of the label matrix. Defaults to 0.0.
        """
        assert missing_labels in ("posterior", "zero"), f"Unknown missing_labels option: {missing_labels}. Choose among 'posterior' and 'zero'."
        centered = centered or standardize
        v_pos, l_pos = data["visible"], data["label"]
        v_neg, l_neg = chains["visible"], chains["label"]
        weights = data["weight"].reshape(-1)
        weights = weights / weights.sum()
        num_chains = len(v_neg)

        # Hidden distribution means and variances
        missing = None
        if missing_labels == "posterior" and isinstance(self.label_layer, CategoricalLabelLayer):
            missing = l_pos.sum(dim=-1) == 0
            if not missing.any():
                missing = None
        if missing is None:
            h_pos, var_h_pos = self.meanvar_hidden_activation(v_pos, l_pos)
        else:
            h_pos, var_h_pos, l_pos, label_hidden_cov = self.meanvar_hidden_activation_missing_labels(v_pos, l_pos, missing)
        h_neg, var_h_neg = self.meanvar_hidden_activation(v_neg, l_neg)

        # Gradient of the parameters of the layers
        self.visible_layer.apply_gradient_visible(x_pos=v_pos, x_neg=v_neg, weights=weights, pseudo_count=pseudo_count)
        self.label_layer.apply_gradient_visible(x_pos=l_pos, x_neg=l_neg, weights=weights, pseudo_count=pseudo_count)
        self.hidden_layer.apply_gradient_hidden(
            mean_h_pos=h_pos,
            mean_h_neg=h_neg,
            var_h_pos=var_h_pos,
            var_h_neg=var_h_neg,
            weights=weights,
            pseudo_count=pseudo_count,
        )

        # Offsets and scales of the units
        if centered:
            offset_v, var_v = self.visible_layer.data_meanvar(v_pos, weights=weights, pseudo_count=pseudo_count)
            offset_l, var_l = self.label_layer.data_meanvar(l_pos, weights=weights, pseudo_count=pseudo_count)
            offset_h, var_h = get_meanvar(h_pos, weights=weights)
            # Law of total variance: Var[h] = Var[<h|v,l>] + <Var[h|v,l]>
            var_h = var_h + get_mean(var_h_pos, weights=weights)
            self.offset_v.copy_(offset_v)
            self.offset_l.copy_(offset_l)
            self.offset_h.copy_(offset_h)
        else:
            self.offset_v.zero_()
            self.offset_l.zero_()
            self.offset_h.zero_()
        # The visible units are only centered: rescaling them by their standard deviation makes the gradient diverge
        # for the rarely observed states (or never active binary units), whose standard deviation is close to zero.
        self.scale_v.fill_(1.0)
        if standardize:
            eps = 1e-6
            self.scale_l.copy_(torch.sqrt(var_l + eps))
            self.scale_h.copy_(torch.sqrt(var_h + eps))
        else:
            self.scale_l.fill_(1.0)
            self.scale_h.fill_(1.0)

        # Gradient with respect to the weights of the reparametrized model
        v_pos = (v_pos - self.offset_v) / self.scale_v
        v_neg = (v_neg - self.offset_v) / self.scale_v
        l_pos = (l_pos - self.offset_l) / self.scale_l
        l_neg = (l_neg - self.offset_l) / self.scale_l
        h_pos = (h_pos - self.offset_h) / self.scale_h
        h_neg = (h_neg - self.offset_h) / self.scale_h
        mean_h_pos = get_mean(h_pos, weights=weights)

        def positive_term(x: torch.Tensor, layer: Layer, offset: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            """<x h> over the data, regularized with the pseudo count for discrete layers."""
            term = batched_outer(x * weights.view((-1,) + (1,) * (x.dim() - 1)), h_pos)
            uniform = layer.uniform_mean()
            if pseudo_count > 0 and uniform is not None:
                term = (1 - pseudo_count) * term + pseudo_count * outer((uniform - offset) / scale, mean_h_pos)
            return term

        grad_weight_matrix = positive_term(v_pos, self.visible_layer, self.offset_v, self.scale_v) - batched_outer(v_neg, h_neg) / num_chains
        grad_label_matrix = positive_term(l_pos, self.label_layer, self.offset_l, self.scale_l) - batched_outer(l_neg, h_neg) / num_chains
        if missing is not None:
            # The covariance is invariant under centering and scales like l h
            label_hidden_cov = torch.einsum("n,nch->ch", weights[missing], label_hidden_cov) / outer(self.scale_l, self.scale_h)
            if pseudo_count > 0:
                label_hidden_cov = (1 - pseudo_count) * label_hidden_cov
            grad_label_matrix += label_hidden_cov

        # Map the gradient back to the original parameters
        grad_weight_matrix = grad_weight_matrix / outer(self.scale_v, self.scale_h)
        grad_label_matrix = grad_label_matrix / outer(self.scale_l, self.scale_h)
        if centered:
            self.visible_layer.bias.grad -= grad_weight_matrix @ self.offset_h
            self.label_layer.bias.grad -= grad_label_matrix @ self.offset_h
            self.hidden_layer.bias.grad -= mm_left(grad_weight_matrix, self.offset_v) + self.offset_l @ grad_label_matrix

        self.weight_matrix.grad = grad_weight_matrix
        self.label_matrix.grad = grad_label_matrix
        # Regularization
        self.regularize(l1_strength, l2_strength, l1l2_strength, l1_label_strength, l2_label_strength)


    def forward(
        self,
        data_batch: Dict[str, torch.Tensor],
        chains: Dict[str, torch.Tensor],
        gibbs_steps: int,
        **gradient_kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Computes the gradient of the parameters of the model and updates the Markov chains using the Persistent Contrastive Divergence algorithm.
        The parameters are not updated: the optimizer step has to be performed afterwards.

        Args:
            data_batch (Dict[str, torch.Tensor]): Batch of data.
            chains (Dict[str, torch.Tensor]): Persistent chains.
            gibbs_steps (int): Number of Alternating Gibbs Sampling steps.
            **gradient_kwargs: Options of the gradient, see `apply_gradient`.

        Returns:
            Dict[str, torch.Tensor]: Updated chains.
        """
        self.apply_gradient(data=data_batch, chains=chains, **gradient_kwargs)
        return self.sample(
            gibbs_steps=gibbs_steps,
            visible=chains["visible"],
            label=chains["label"],
            beta=1.0,
        )


    def get_biased_model(
        self,
        gen_strength: float,
        label_strength: float,
        wt_strength: float,
        wt: torch.Tensor,
    ) -> "AnnaRBM":
        """Returns a copy of the model where the generative part (visible and hidden layers, weight matrix) is multiplied by
        `gen_strength`, the label part (label layer and label matrix) by `label_strength`, and the visible bias is shifted by
        `wt_strength * wt`.

        Since multiplying the parameters by a factor is equivalent to changing the inverse temperature, the biased model
        samples sequences at different effective temperatures for the sequence and the annotation parts of the energy,
        optionally biased towards a reference sequence (e.g. a wild type), as in sequence design applications.

        Args:
            gen_strength (float): Multiplicative factor of the generative parameters.
            label_strength (float): Multiplicative factor of the label parameters.
            wt_strength (float): Strength of the additional visible bias.
            wt (torch.Tensor): Additional visible bias, e.g. the one-hot encoding of a wild-type sequence.

        Returns:
            AnnaRBM: The biased model.
        """
        biased_rbm = AnnaRBM(
            visible_layer=copy.deepcopy(self.visible_layer),
            hidden_layer=copy.deepcopy(self.hidden_layer),
            label_layer=copy.deepcopy(self.label_layer),
        ).to(self.weight_matrix)
        biased_rbm.load_state_dict(self.state_dict())
        biased_rbm.weight_matrix.mul_(gen_strength)
        biased_rbm.label_matrix.mul_(label_strength)
        for p in biased_rbm.visible_layer.parameters():
            p.mul_(gen_strength)
        for p in biased_rbm.hidden_layer.parameters():
            p.mul_(gen_strength)
        for p in biased_rbm.label_layer.parameters():
            p.mul_(label_strength)
        biased_rbm.visible_layer.bias.add_(wt_strength * wt.to(self.weight_matrix))
        return biased_rbm


    def __repr__(self) -> str:
        return f"AnnaRBM(visible_layer={self.visible_layer}, hidden_layer={self.hidden_layer}, label_layer={self.label_layer}, shape={tuple(self.shape)})"
