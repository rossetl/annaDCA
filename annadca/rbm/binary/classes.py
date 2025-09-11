from typing import Optional, Dict
import numpy as np

import torch
from adabmDCA.fasta import import_from_fasta

from annadca.classes import annaRBM
from annadca.io import _save_chains
from annadca.rbm.binary.statmech import (
    _compute_energy,
    _compute_energy_visibles,
    _compute_energy_hiddens,
    _update_weights_AIS,
    _compute_log_likelihood,
)
from annadca.rbm.binary.sampling import (
    _sample,
    _sample_hiddens,
    _sample_visibles,
    _sample_labels,
    _sample_conditioned,
    _predict_labels,
)
from annadca.rbm.binary.init import _init_parameters, _init_chains
from annadca.rbm.binary.grad import _compute_gradient


class annaRBMbin(annaRBM):
    def __init__(
        self,
        params: Dict[str, torch.Tensor] | None = None,
        device: Optional[torch.device] = torch.device("cpu"),
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        super().__init__(params, device, dtype)
        
        
    def to(
        self,
        device: Optional[torch.device] | None = None,
        dtype: Optional[torch.dtype] | None = None,
    ):
        """Moves the parameters to the specified device and/or dtype.

        Args:
            device (Optional[torch.device], optional): Device. Defaults to None.
            dtype (Optional[torch.dtype], optional): Dtype. Defaults to None.

        Returns:
            annaRBM: annaRBM instance with the parameters moved to the specified device and/or dtype.
        """
        return super().to(device, dtype)
    
    
    def clone(
        self,
        device: Optional[torch.device] | None = None,
        dtype: Optional[torch.dtype] | None = None,
    ):
        """Clone the annaRBMbin instance.

        Args:
            device (Optional[torch.device], optional): Device. Defaults to None.
            dtype (Optional[torch.dtype], optional): Dtype. Defaults to None.

        Returns:
            annaRBM: annaRBM instance cloned.
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
            
        cloned_params = {k: v.clone().to(device=device, dtype=dtype) for k, v in self.params.items()}
            
        return annaRBMbin(
            params=cloned_params,
            device=device,
            dtype=dtype,
        )
    
    
    def save(
        self,
        filename: str,
        num_updates: int,
    ) -> None:
        """Save the parameters of the annaRBM.

        Args:
            filename (str): Path to the h5 archive where to store the model.
            num_updates (int): Number of updates performed so far.
        """
        return super().save(filename, num_updates)
    
    
    def load(
        self,
        filename: str,
        index: int | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        set_rng_state: bool = False,
    ) -> int:
        """Loads the parameters of the annaRBM.

        Args:
            filename (str): Path to the h5 archive.
            index (int): Index of the machine to load.
            device (torch.device): PyTorch device on which to load the parameters.
            dtype (torch.dtype): Dtype for the parameters.
            set_rng_state (bool): Restore the random state at the given epoch (useful to restore training). Defaults to False.

        Returns:
            int: Number of model updates
        """
        return super().load(
            filename=filename,
            index=index,
            device=device,
            dtype=dtype,
            set_rng_state=set_rng_state,
            )
        
        
    def compute_energy(
        self,
        visible: torch.Tensor,
        hidden: torch.Tensor,
        label: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Computes the energy of the model on the given configuration.

        Args:
            visible (torch.Tensor): Visible units.
            hidden (torch.Tensor): Hidden units.
            label (torch.Tensor): Labels.

        Returns:
            torch.Tensor: Energy of the model on the given configuration.
        """
    
        return _compute_energy(visible, hidden, label, self.params)
    
    
    def compute_energy_visibles(
        self,
        visible: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Returns the energy of the model computed on the input visibles.

        Args:
            visible (torch.Tensor): Visible units.

        Returns:
            torch.Tensor: Energy of the data points.
        """
        return _compute_energy_visibles(visible, self.params)
    
    
    def compute_energy_hiddens(
        self,
        hidden: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Computes the energy of the model on the hidden layer.

        Args:
            hidden (torch.Tensor): Hidden units.

        Returns:
            torch.Tensor: Energy of the model on the hidden layer.
        """
        return _compute_energy_hiddens(hidden, self.params)
    
    
    def sample_hiddens(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
        beta: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Sample the hidden layer conditionally to the visible one and the labels.

        Args:
            visible (torch.Tensor): Visible units.
            label (torch.Tensor): Labels.
            beta (float, optional): Inverse temperature. Defaults to 1.0.

        Returns:
            Dict[str, torch.Tensor]: Sampled hidden units and hidden magnetization.
        """
        hidden, hidden_mag = _sample_hiddens(visible, label, self.params, beta)
        return {"hidden": hidden, "hidden_mag": hidden_mag}
    
    
    def sample_visibles(
        self,
        hidden: torch.Tensor,
        beta: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Sample the visible layer conditionally to the hidden one.

        Args:
            hidden (torch.Tensor): Hidden units.
            beta (float, optional): Inverse temperature. Defaults to 1.0.

        Returns:
            Dict[str, torch.Tensor]: Sampled visible units and visible magnetization.
        """
        visible, visible_mag = _sample_visibles(hidden, self.params, beta)
        return {"visible": visible, "visible_mag": visible_mag}
    
    
    def sample_labels(
        self,
        hidden: torch.Tensor,
        beta: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Sample the labels conditionally to the hidden layer.

        Args:
            hidden (torch.Tensor): Hidden units.
            beta (float, optional): Inverse temperature. Defaults to 1.0.

        Returns:
            Dict[str, torch.Tensor]: Sampled labels and label magnetization.
        """
        label, label_mag = _sample_labels(hidden, self.params, beta)
        return {"label": label, "label_mag": label_mag}
    
    
    def sample(
        self,
        gibbs_steps: int,
        num_samples: int | None = None,
        visible: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        beta: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Samples from the binary annaRBM.

        Args:
            gibbs_steps (int): Number of Gibbs steps.
            num_samples (int | None, optional): Number of samples to generate. If None, it will be inferred from the visible or label tensors. Defaults to None.
            visible (torch.Tensor): Visible units. If None, it will be initialized randomly.
            label (torch.Tensor): Labels. If None, it will be initialized randomly. If both visible and label are None, num_samples must be provided.
            beta (float, optional): Inverse temperature. Defaults to 1.0.

        Returns:
            Dict[str, torch.Tensor]: Sampled visible, hidden units and labels.
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

        visible, hidden, label, visible_mag, hidden_mag, label_mag = _sample(gibbs_steps, visible, label, self.params, beta)
        return {"visible": visible, "hidden": hidden, "label": label, "visible_mag": visible_mag, "hidden_mag": hidden_mag, "label_mag": label_mag}


    def sample_conditioned(
        self,
        gibbs_steps: int,
        targets: torch.Tensor | np.ndarray,
        visible: torch.Tensor | None = None,
        beta: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Samples from the annaRBM conditioned on the target labels. During the sampling, the labels are kept fixed
        and the visible and hidden units are sampled alternatively. The visible's conditional probability distribution
        is returned.

        Args:
            gibbs_steps (int): Number of Alternate Gibbs Sampling steps.
            targets (torch.Tensor | np.ndarray): Target labels.
            visible (torch.Tensor | None, optional): Visible units. If None, it will be initialized randomly. Defaults to None.
            beta (float, optional): Inverse temperature. Defaults to 1.0.

        Returns:
            Dict[str, torch.Tensor]: Sampled visible and hidden units.
        """
        if isinstance(targets, np.ndarray):
            targets = torch.tensor(targets, device=self.device, dtype=self.dtype)
        elif isinstance(targets, torch.Tensor):
            targets = targets.to(self.device, dtype=self.dtype)
        num_samples = targets.shape[0]
        if visible is None:
            visible = self.init_chains(num_samples=num_samples)["visible"]
        else:
            if visible.shape[0] != num_samples:
                raise ValueError(f"The number of visible units ({visible.shape[0]}) and targets ({num_samples}) must be the same.")

        visible, hidden, visible_mag, hidden_mag = _sample_conditioned(
            gibbs_steps=gibbs_steps,
            label=targets,
            visible=visible,
            params=self.params,
            beta=beta,
        )

        return {"visible": visible, "hidden": hidden, "visible_mag": visible_mag, "hidden_mag": hidden_mag}


    def predict_labels(
        self,
        visible: torch.Tensor | np.ndarray,
        beta: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """Returns the probability distribution of each label given the visible units: p(l|v).

        Args:
            visible (torch.Tensor | np.ndarray): Visible units.
            beta (float, optional): Inverse temperature. Defaults to 1.0.

        Returns:
            torch.Tensor: Labels's probability distribution: p(l|v).
        """
        if isinstance(visible, np.ndarray):
            visible = torch.tensor(visible, device=self.device, dtype=self.dtype)
        
        p_labels = _predict_labels(
            visible=visible,
            params=self.params,
            beta=beta,            
        )
        
        return p_labels
    
    
    def get_pattern(self, label_idx: int) -> torch.Tensor:
        """Returns the visible pattern associated to the given label index.

        Args:
            label_idx (int): Index of the label.

        Returns:
            torch.Tensor: Visible pattern of dimension (num_visibles,) associated to the label.
        """
        mag_prototype = self.params["label_matrix"][label_idx]
        pattern = (self.params["weight_matrix"] @ mag_prototype)
        
        return pattern
    
    
    def update_weights_AIS(
        self,
        prev_model: annaRBM,
        chains: Dict[str, torch.Tensor],
        log_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Update the weights used during the trajectory Annealed Importance Sampling (AIS) algorithm.

        Args:
            prev_model (annRBM): Model at time t-1.
            chains (torch.Tensor): Chains at time t-1.
            log_weights (torch.Tensor): Log-weights at time t-1.

        Returns:
            torch.Tensor: Log-weights at time t.
        """
        if prev_model.params is None:
            raise ValueError("Previous model parameters are not initialized.")
        return _update_weights_AIS(prev_model.params, self.params, chains, log_weights)
    
    
    def compute_log_likelihood(
        self,
        visible: torch.Tensor,
        weight: torch.Tensor,
        logZ: float,
        **kwargs,
    ) -> float:
        """Computes the log-likelihood of the model on the given configuration.

        Args:
            visible (torch.Tensor): Visible units.
            weight (torch.Tensor): weights of the sequences.
            logZ (float): Log partition function.

        Returns:
            float: Log-likelihood of the model on the given configuration.
        """
        return _compute_log_likelihood(visible, weight, self.params, logZ)
    
    
    def init_chains(
        self,
        num_samples: int,
        use_profile: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Initialize the Markov chains for the RBM by sampling a uniform distribution on the visible layer and the labels
        and sampling the hidden layer according to the visible one. If use_profile is True, the visible units and the label
        are sampled from the profile model using the local fields.

        Args:
            num_samples (int): Number of parallel chains.
            use_profile (bool, optional): Whether to use the profile model. Defaults to False.
    
        Returns:
            Dict[str, torch.Tensor]: Initial Markov chain.
        """
        return _init_chains(num_samples, self.params, use_profile)
    
    
    @staticmethod
    def save_chains(
        filename: str,
        visible: torch.Tensor,
        label: torch.Tensor,
        **kwargs,
    ) -> None:
        """Save the persistent chains on a fasta file.

        Args:
            filename (str): Path to the fasta file.
            visible (torch.Tensor): Visible units of the chains.
            label (torch.Tensor): Labels of the chains.
        """
        _save_chains(
            filename=filename,
            visible=visible,
            label=label,
            alphabet="01",
        )
        
    
    def load_chains(
        self,
        filename: str,
        device: torch.device,
        dtype: torch.dtype,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Load the persistent chains from a fasta file.

        Args:
            filename (str): Path to the fasta file.
            device (torch.device): Device on which to load the chains.
            dtype (torch.dtype): Data type for the chains.

        Returns:
            Dict[str, torch.Tensor]: Visible and hidden units and labels of the chains.
        """
        headers, sequences = import_from_fasta(filename)
        
        label = np.vectorize(lambda x: np.array([int(i) for i in x]), signature="() -> (l)")(headers)
        visible = np.vectorize(lambda x: np.array([int(i) for i in x]), signature="() -> (l)")(sequences)
        label = torch.tensor(label, device=device, dtype=dtype)
        visible = torch.tensor(visible, device=device, dtype=dtype)
        if self.params is not None:
            hidden = self.sample_hiddens(visible, label)["hidden"]
        else:
            hidden = torch.zeros((visible.shape[0],), device=device, dtype=dtype)

        return {"visible": visible, "hidden": hidden, "label": label}
    
    
    def compute_gradient(
        self,
        data: Dict[str, torch.Tensor],
        chains: Dict[str, torch.Tensor],
        pseudo_count: float = 0.0,
        centered: bool = True,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        eta: float = 1.0,
    ) -> None:
        """Computes the gradient of the log-likelihood and stores it.

        Args:
            data (Dict[str, torch.Tensor]): Data batch.
            chains (Dict[str, torch.Tensor]): Chains.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
            centered (bool, optional): Centered gradient. Defaults to True.
            lambda_l1 (float, optional): L1 regularization weight. Defaults to 0.0.
            lambda_l2 (float, optional): L2 regularization weight. Defaults to 0.0.
            eta (float, optional): Relative contribution of the label term. Defaults to 1.0.
        """
        _compute_gradient(
            data=data,
            chains=chains,
            params=self.params,
            pseudo_count=pseudo_count,
            centered=centered,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            eta=eta,
        )
        
    
    def zerosum_gauge(self) -> None:
        """Uneffective method for the binary annaRBM."""
        # The zerosum gauge is not applicable to the binary RBM, so we do nothing here.
        pass
        
    
    def init_parameters(
        self,
        num_visibles: int,
        num_hiddens: int,
        num_labels: int,
        frequencies_visibles: torch.Tensor | None = None,
        frequencies_labels: torch.Tensor | None = None,
        std_init: float = 1e-4,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> None:
        """
        Initialize the parameters of the model. Hidden biases are set to 0,
        visible and label biases are set to the independent-site model value using the frequencies of the dataset (if provided),
        and the weight and label matrices are initialized with a Gaussian distribution of sigma std_init.

        Args:
            num_visibles (int): Number of visible units.
            num_hiddens (int): Number of hidden units.
            num_labels (int): Number of labels.
            frequencies_visibles (torch.Tensor): Empirical frequencies of the visible units. Defaults to None.
            frequencies_labels (torch.Tensor): Empirical frequencies of the labels. Defaults to None.
            std_init (float, optional): Standard deviation of the weight matrix. Defaults to 1e-4.
            device (torch.device): Device for the parameters.
            dtype (torch.dtype): Data type for the parameters.
        """
        self.device = device
        self.dtype = dtype
        self.params = _init_parameters(
            num_visibles=num_visibles,
            num_hiddens=num_hiddens,
            num_labels=num_labels,
            frequencies_visibles=frequencies_visibles,
            frequencies_labels=frequencies_labels,
            std_init=std_init,
            device=self.device,
            dtype=self.dtype
        )
        
    
    def num_visibles(self) -> int:
        """Returns the number of visible units.

        Returns:
            int: Number of visible units.
        """
        return super().num_visibles()
    
    
    def num_hiddens(self) -> int:
        """Returns the number of hidden units.

        Returns:
            int: Number of hidden units.
        """
        return super().num_hiddens()
    
    
    def num_classes(self) -> int:
        """Returns the number of label classes.

        Returns:
            int: Number of label classes.
        """
        return super().num_classes()
    
    
    def logZ0(self) -> float:
        """Computes the initial log partition function for the annaRBM.

        Returns:
            float: Initial log partition function.
        """
        logZ_visibles = torch.log(1.0 + torch.exp(self.params["vbias"])).sum().item()
        logZ_labels = torch.log(1.0 + torch.exp(self.params["lbias"])).sum().item()
        logZ_hiddens = torch.log(1.0 + torch.exp(self.params["hbias"])).sum().item()
        return logZ_visibles + logZ_labels + logZ_hiddens
        