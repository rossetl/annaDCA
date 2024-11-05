from typing import Optional, Self, Dict
from abc import ABC, abstractmethod

from aiDCA.dataset import aiDataset
from aiDCA.io import _save_model, _load_model

import torch
        
        
class aiRBM(ABC):
    """Abstract class for the annotation-informed Restricted Boltzmann Machine (aiRBM)"""

    def __init__(
        self,
        params: Dict[str, torch.Tensor] = None,
        device: Optional[torch.device] = "cpu",
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        if params is not None:
            self.params = {params[k].to(device=device, dtype=dtype) for k in params}

        self.device = device
        self.dtype = dtype


    @torch.jit.export
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
        if device is not None:
            self.params = {self.params[key].to(device) for key in self.params}
            self.device = device

        if dtype is not None:
            self.params = {self.params[key].to(dtype) for key in self.params}
            self.dtype = dtype
        return self


    @torch.jit.export
    def clone(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Self:
        """Clone the BBParams instance.

        Args:
            device (Optional[torch.device], optional): Device. Defaults to None.
            dtype (Optional[torch.dtype], optional): Dtype. Defaults to None.

        Returns:
            aiRBM: aiRBM instance cloned.
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
            
        return aiRBM(
            params=self.params,
            device=device,
            dtype=dtype,
        )
        

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
        _save_model(
            params=self.params,
            filename=filename,
            num_updates=num_updates,
        )
    
    
    def load(
        self,
        filename: str,
        index: int,
        device: torch.device,
        dtype: torch.dtype,
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
        self.params = _load_model(
            filename=filename,
            index=index,
            device=device,
            dtype=dtype,
            set_rng_state=set_rng_state,
        )
    
    
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
        """Samples from the aiRBM.

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
    ) -> None:
        """Computes the gradient of the log-likelihood and stores it.

        Args:
            data (Dict[str, torch.Tensor]): Data batch.
            chains (Dict[str, torch.Tensor]): Chains.
            pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
            centered (bool, optional): Centered gradient. Defaults to True.
        """
        pass
        

    @abstractmethod
    def init_parameters(
        self,
        dataset: aiDataset,
        num_hiddens: int,
        device: torch.device,
        dtype: torch.dtype,
        sigma: float = 1e-4,
    ) -> Self:
        """Initializes the parameters of the aiRBM.

        Args:
            dataset (aiDataset): Dataset.
            num_hiddens (int): Number of hidden units.
            device (torch.device): Device.
            dtype (torch.dtype): Dtype.
            sigma (float, optional): Standard deviation of the weight matrix. Defaults to 1e-4.

        Returns:
            Self: aiRBM instance.
        """
        pass
        
    
    @abstractmethod
    def num_visibles(self) -> int:
        """Returns the number of visible units.

        Returns:
            int: Number of visible units.
        """
        return self.params["vbias"].shape[0]
        
        
    @abstractmethod
    def num_hiddens(self) -> int:
        """Returns the number of hidden units.

        Returns:
            int: Number of hidden units.
        """
        return self.params["hbias"].shape[0]
        
        
    @abstractmethod
    def num_classes(self) -> int:
        """Returns the number of labels.

        Returns:
            int: Number of labels.
        """
        return self.params["lbias"].shape[0]
        
        
    @abstractmethod
    def logZ0(self) -> float:
        """Computes the initial log partition function for the aiRBM.

        Returns:
            float: Initial log partition function.
        """
        pass
    
