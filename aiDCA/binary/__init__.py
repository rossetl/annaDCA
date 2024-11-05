from typing import Optional, Dict, Self
import numpy as np

import torch
from adabmDCA.fasta_utils import write_fasta, import_from_fasta

from aiDCA.classes import aiRBM
from aiDCA.io import _save_chains
from aiDCA.binary.statmech import _compute_energy, _compute_energy_visibles, _compute_energy_hiddens
from aiDCA.binary.sampling import _sample, _sample_hiddens, _sample_visibles, _sample_labels
from aiDCA.binary.init import _init_parameters, _init_chains
from aiDCA.binary.grad import _compute_gradient


class aiRBMbin(aiRBM):
    def __init__(
        self,
        params: Dict[str, torch.Tensor] = None,
        device: Optional[torch.device] = "cpu",
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        super().__init__(params, device, dtype)
        
        
    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Self:
        """Moves the parameters to the specified device and/or dtype.

        Args:
            device (Optional[torch.device], optional): Device. Defaults to None.
            dtype (Optional[torch.dtype], optional): Dtype. Defaults to None.

        Returns:
            aiRBM: aiRBM instance with the parameters moved to the specified device and/or dtype.
        """
        return super().to(device, dtype)
    
    
    def clone(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Self:
        """Clone the aiRBM instance.

        Args:
            device (Optional[torch.device], optional): Device. Defaults to None.
            dtype (Optional[torch.dtype], optional): Dtype. Defaults to None.

        Returns:
            aiRBM: aiRBM instance cloned.
        """
        return super().clone(device, dtype)
    
    
    def save(
        self,
        filename: str,
        num_updates: int,
    ) -> None:
        """Save the parameters of the aiRBM.

        Args:
            filename (str): Path to the h5 archive where to store the model.
            num_updates (int): Number of updates performed so far.
        """
        return super().save(filename, num_updates)
    
    
    def load(
        self,
        filename: str,
        index: int,
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        set_rng_state: bool = False,
    ) -> None:
        """Loads the parameters of the aiRBM.

        Args:
            filename (str): Path to the h5 archive.
            index (int): Index of the machine to load.
            device (torch.device): PyTorch device on which to load the parameters.
            dtype (torch.dtype): Dtype for the parameters.
            set_rng_state (bool): Restore the random state at the given epoch (useful to restore training). Defaults to False.
        """
        return super().load(filename, index, device, dtype, set_rng_state)
        
        
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
        label: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Returns the energy of the model computed on the input visibles and labels.

        Args:
            visible (torch.Tensor): Visible units.
            label (torch.Tensor): Labels.

        Returns:
            torch.Tensor: Energy of the data points.
        """
        return _compute_energy_visibles(visible, label, self.params)
    
    
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
        h, mh = _sample_hiddens(visible, label, self.params, beta)
        return {"hidden": h, "hidden_mag": mh}
    
    
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
        v, mv = _sample_visibles(hidden, self.params, beta)
        return {"visible": v, "visible_mag": mv}
    
    
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
        l, ml = _sample_labels(hidden, self.params, beta)
        return {"label": l, "label_mag": ml}
    
    
    def sample(
        self,
        gibbs_steps: int,
        visible: torch.Tensor,
        label: torch.Tensor,
        beta: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Samples from the binary aiRBM.

        Args:
            gibbs_steps (int): Number of Gibbs steps.
            visible (torch.Tensor): Visible units.
            label (torch.Tensor): Labels.
            beta (float, optional): Inverse temperature. Defaults to 1.0.

        Returns:
            Dict[str, torch.Tensor]: Sampled visible, hidden units and labels.
        """
        v, h, l = _sample(gibbs_steps, visible, label, self.params, beta)
        return {"visible": v, "hidden": h, "label": l}
    
    
    def init_chains(self, num_samples: int) -> Dict[str, torch.Tensor]:
        """Initialize a Markov chain for the RBM by sampling a uniform distribution on the visible layer and the labels
        and sampling the hidden layer according to the visible one.

        Args:
            num_samples (int): Number of parallel chains.
    
        Returns:
            Dict[str, torch.Tensor]: Initial Markov chain.
        """
        return _init_chains(num_samples, self.params)
    
    
    @staticmethod
    def save_chains(
        filename: str,
        visible: torch.Tensor,
        label: torch.Tensor,
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
        label = np.vectorize(lambda x: np.array([int(i) for i in x]))(headers)
        visible = np.vectorize(lambda x: np.array([int(i) for i in x]))(sequences)
        label = torch.tensor(label, device=device, dtype=dtype)
        visible = torch.tensor(visible, device=device, dtype=dtype)
        hidden, _ = self.sample_hiddens(visible, label)

        return {"visible": visible, "hidden": hidden, "label": label}
    
    
    def compute_gradient(
        self,
        data: Dict[str, torch.Tensor],
        chains: Dict[str, torch.Tensor],
        pseudo_count: float = 0.0,
        centered: bool = True,
    ) -> None:
        """Computes the gradient of the log-likelihood and stores it.

        Args:
            data (Dict[str, torch.Tensor]): Data batch.
            chains (Dict[str, torch.Tensor]): Chains.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
            centered (bool, optional): Centered gradient. Defaults to True.
        """
        _compute_gradient(
            data=data,
            chains=chains,
            params=self.params,
            pseudo_count=pseudo_count,
            centered=centered,
        )
        
    
    def init_parameters(
        self,
        num_visibles: int,
        num_hiddens: int,
        num_labels: int,
        frequencies_visibles: torch.Tensor = None,
        frequencies_labels: torch.Tensor = None,
        std_init: float = 1e-4,
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
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

        Returns:
            Dict[str, torch.Tensor]: Parameters of the model.
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
            dtype=self.dtype)
        
    
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
        """Computes the initial log partition function for the aiRBM.

        Returns:
            float: Initial log partition function.
        """
        logZ_visibles = torch.log(1.0 + torch.exp(self.params["vbias"])).sum()
        logZ_labels = torch.log(1.0 + torch.exp(self.params["lbias"])).sum()
        logZ_hiddens = torch.log(1.0 + torch.exp(self.params["hbias"])).sum()
        return logZ_visibles + logZ_labels + logZ_hiddens
        