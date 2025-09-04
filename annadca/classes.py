from typing import Optional, Dict
from abc import ABC, abstractmethod
import numpy as np
import torch

from annadca.dataset import annaDataset
from annadca.io import _save_model, _load_model
        
        
class annaRBM(ABC):
    """Abstract class for the annotation-assisted Restricted Boltzmann Machine (annaRBM)"""

    def __init__(
        self,
        params: Dict[str, torch.Tensor] | None = None,
        device: Optional[torch.device] = torch.device("cpu"),
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        if params is not None:
            self.params = {k : params[k].to(device=device, dtype=dtype) for k in params.keys()}
        else:
            self.params = None

        self.device = device
        self.dtype = dtype


    @torch.jit.export
    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Moves the parameters to the specified device and/or dtype.

        Args:
            device (Optional[torch.device], optional): Device. Defaults to None.
            dtype (Optional[torch.dtype], optional): Dtype. Defaults to None.

        Returns:
            annaRBM: annaRBM instance with the parameters moved to the specified device and/or dtype.
        """
        if device is not None:
            if self.params is not None:
                self.params = {key: self.params[key].to(device) for key in self.params}
            self.device = device

        if dtype is not None:
            if self.params is not None:
                self.params = {key: self.params[key].to(dtype) for key in self.params}
            self.dtype = dtype
        return self


    @torch.jit.export
    def clone(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Clone the annaRBM instance.

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
            
        # Cannot instantiate abstract class annaRBM, so use type(self) to instantiate the concrete subclass
        return type(self)(
            params=self.params,
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
        if self.params is None:
            raise ValueError("Model parameters are not initialized.")
        _save_model(
            params=self.params,
            filename=filename,
            num_updates=num_updates,
        )
    
    
    def load(
        self,
        filename: str,
        device: torch.device,
        dtype: torch.dtype,
        index: int | None = None,
        set_rng_state: bool = False,
    ) -> int:
        """Loads the parameters of the annaRBM.

        Args:
            filename (str): Path to the h5 archive.
            device (torch.device): PyTorch device on which to load the parameters.
            dtype (torch.dtype): Dtype for the parameters.
            index (int | None): Index of the machine to load. If None, the last machine is loaded. Defaults to None.
            set_rng_state (bool): Restore the random state at the given epoch (useful to restore training). Defaults to False.

        Returns:
            int: Number of model updates.
        """
        num_updates, self.params = _load_model(
            filename=filename,
            device=device,
            dtype=dtype,
            index=index,
            set_rng_state=set_rng_state,
        )
        self.device = device
        self.dtype = dtype
        
        return num_updates
    
    
    @abstractmethod
    def compute_energy(
        self,
        visible: torch.Tensor,
        hidden: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Computes the energy of the visible and hidden units of the input data.

        Args:
            visible (torch.Tensor): Visible units.
            hidden (torch.Tensor): Hidden units.

        Returns:
            torch.Tensor: Energy of the data.
        """
        pass
    
    
    @abstractmethod
    def compute_energy_visibles(
        self,
        visible: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Computes the energy of the visible units.

        Args:
            visible (torch.Tensor): Visible units.

        Returns:
            torch.Tensor: Energy of the visible units.
        """
        pass
    
    
    @abstractmethod
    def compute_energy_hiddens(
        self,
        hidden: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Computes the energy of the hidden units.

        Args:
            hidden (torch.Tensor): Hidden units.

        Returns:
            torch.Tensor: Energy of the hidden units.
        """
        pass
    
    
    @abstractmethod
    def sample_hiddens(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
        beta: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Samples the hidden units given the visible units and labels.

        Args:
            visible (torch.Tensor): Visible units.
            label (torch.Tensor): Labels.
            beta (float, optional): Inverse temperature. Defaults to 1.0.

        Returns:
            Dict[str, torch.Tensor]: Sampled hidden units and hidden magnetization.
        """
        pass
    
    
    @abstractmethod
    def sample_visibles(
        self,
        hidden: torch.Tensor,
        label: torch.Tensor,
        beta: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Samples the visible units given the hidden units and labels.

        Args:
            hidden (torch.Tensor): Hidden units.
            label (torch.Tensor): Labels.
            beta (float, optional): Inverse temperature. Defaults to 1.0.

        Returns:
            Dict[str, torch.Tensor]: Sampled visible units and visible magnetization.
        """
        pass
        
        
    @abstractmethod
    def sample_labels(
        self,
        hidden: torch.Tensor,
        beta: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Samples the labels given the hidden units.

        Args:
            hidden (torch.Tensor): Hidden units.
            beta (float, optional): Inverse temperature. Defaults to 1.0.

        Returns:
            Dict[str, torch.Tensor] Sampled labels and label magnetization.
        """
        pass
        
        
    @abstractmethod
    def sample(
        self,
        gibbs_steps: int,
        visible: torch.Tensor,
        label: torch.Tensor,
        beta: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Samples from the annaRBM.

        Args:
            gibbs_steps (int): Number of Alternate Gibbs Sampling steps.
            visible (torch.Tensor): Visible units initial configuration.
            label (torch.Tensor): Labels initial configuration.
            beta (float, optional): Inverse temperature. Defaults to 1.0.

        Returns:
            Dict[str, torch.Tensor]: Sampled visible, hidden and label units.
        """
        pass
    
    
    @abstractmethod
    def sample_conditioned(
        self,
        gibbs_steps: int,
        chains: Dict[str, torch.Tensor] | torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        beta: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """Samples from the annaRBM conditioned on the target labels. During the sampling, the labels are kept fixed
        and the visible and hidden units are sampled alternatively. The visible's conditional probability distribution
        is returned.

        Args:
            gibbs_steps (int): Number of Alternate Gibbs Sampling steps.
            chains (Dict[str, torch.Tensor] | torch.Tensor | np.ndarray): Chains initialization. It can be either a chain dictionary
                instance or a torch.Tensor (np.ndarray) representing the visible units.
            targets (torch.Tensor | np.ndarray): Target labels.
            beta (float, optional): Inverse temperature. Defaults to 1.0.

        Returns:
            torch.Tensor: Conditional probability distribution of the visible units.
        """
        pass
        
        
    def predict_labels(
        self,
        gibbs_steps: int,
        chains: Dict[str, torch.Tensor] | torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        beta: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Samples from the annaRBM conditioned on the target visible units. During the sampling, the visibles are kept fixed
        and the labels and hidden units are sampled alternatively. The label's conditional probability distribution is returned.

        Args:
            gibbs_steps (int): Number of Alternate Gibbs Sampling steps.
            chains (Dict[str, torch.Tensor] | torch.Tensor | np.ndarray): Chains initialization. It can be either a chain dictionary
                instance or a torch.Tensor (np.ndarray) representing the labels.
            targets (torch.Tensor | np.ndarray): Target visible units.
            beta (float, optional): Inverse temperature. Defaults to 1.0.

        Returns:
            Dict[str, torch.Tensor]: Labels's probability distribution.
        """
        pass


    @abstractmethod
    def update_weights_AIS(
        self,
        prev_model: "annaRBM",
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
        pass
    
    
    @abstractmethod
    def compute_log_likelihood(
        self,
        visible: torch.Tensor,
        label: torch.Tensor,
        logZ: float,
        **kwargs,
    ) -> float:
        """Computes the log-likelihood of the model on the given configuration.

        Args:
            visible (torch.Tensor): Visible units.
            label (torch.Tensor): Labels.
            logZ (float): Log partition function.

        Returns:
            float: Log-likelihood of the model on the given configuration.
        """
        pass


    @abstractmethod
    def init_chains(
        self,
        num_samples: int,
    ) -> Dict[str, torch.Tensor]:
        """Initializes the chains.

        Args:
            num_samples (int): Number of samples.

        Returns:
            Dict[str, torch.Tensor]: Initialized chains.
        """
        pass
    

    @staticmethod
    @abstractmethod
    def save_chains(
        filename: str,
        visible: torch.Tensor,
        label: torch.Tensor,
    ) -> None:
        """Save the chains to a file.

        Args:
            filename (str): Path to the file where to store the chains.
            visible (torch.Tensor): Visible units.
            label (torch.Tensor): Labels.
        """
        pass
    
    
    @abstractmethod
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
        pass        
        
        
    @abstractmethod
    def compute_gradient(
        self,
        data: Dict[str, torch.Tensor],
        chains: Dict[str, torch.Tensor],
        pseudo_count: float = 0.0,
        centered: bool = True,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        eta: float = 1.0
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
        pass
        

    @abstractmethod
    def init_parameters(
        self,
        dataset: annaDataset,
        num_hiddens: int,
        device: torch.device,
        dtype: torch.dtype,
        sigma: float = 1e-4,
    ):
        """Initializes the parameters of the annaRBM.

        Args:
            dataset (aiDataset): Dataset.
            num_hiddens (int): Number of hidden units.
            device (torch.device): Device.
            dtype (torch.dtype): Dtype.
            sigma (float, optional): Standard deviation of the weight matrix. Defaults to 1e-4.

        Returns:
            annaRBM: annaRBM instance.
        """
        pass
        
    
    @abstractmethod
    def num_visibles(self) -> int:
        """Returns the number of visible units.

        Returns:
            int: Number of visible units.
        """
        if self.params is None:
            raise ValueError("Model parameters are not initialized.")
        return self.params["vbias"].shape[0]
        
        
    @abstractmethod
    def num_hiddens(self) -> int:
        """Returns the number of hidden units.

        Returns:
            int: Number of hidden units.
        """
        if self.params is None:
            raise ValueError("Model parameters are not initialized.")
        return self.params["hbias"].shape[0]
        
        
    @abstractmethod
    def num_classes(self) -> int:
        """Returns the number of labels.

        Returns:
            int: Number of labels.
        """
        if self.params is None:
            raise ValueError("Model parameters are not initialized.")
        return self.params["lbias"].shape[0]
        
        
    @abstractmethod
    def logZ0(self) -> float:
        """Computes the initial log partition function for the annaRBM.

        Returns:
            float: Initial log partition function.
        """
        pass
    
