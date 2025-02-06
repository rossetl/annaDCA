from typing import Any
from pathlib import Path
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from torch.utils.data import Dataset
import torch
from adabmDCA.dataset import DatasetDCA
from adabmDCA.fasta import get_tokens, compute_weights
from adabmDCA.functional import one_hot

from annadca.utils import _parse_labels, _complete_labels    

class annaDataset(ABC, Dataset):
    """Abstract class for the dataset that handles annotations.
    """

    @abstractmethod
    def __init__(
        self,
        path_ann: str | Path,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize the dataset.

        Args:
            path_ann (str | Path): Path to the file containing the labels of the sequences.
            device (torch.device, optional): Device to be used. Defaults to "cpu".
            dtype (torch.dtype, optional): Data type of the data. Defaults to torch.float32.
        """
        self.device = device
        self.dtype = dtype
        
        # Import the labels
        self.labels_dict_list = []
        ann_df = pd.read_csv(path_ann).astype(str)
        self.legend = [n for n in ann_df.columns if n != "Name"]
        for leg in self.legend:
                self.labels_dict_list.append({str(n) : str(l) for n, l in zip(ann_df["Name"], ann_df[leg])})
        
        
    def parse_labels(
        self,
        data_names: np.ndarray | list,
    ) -> None:
        """Goes through all the data names and assigns the corresponding one-hot encoded labels to them.
        It also creates the mapping (dictionaries) from the labels to their indexes in the one-hot encoding
        and vice versa.
        
        Args:
            data_names (np.ndarray | list): Names of the data samples.
        """
        # Ensure that the labels order follows the order of the data. Puts None labels for missing data.
        complete_labels_dict_list = [_complete_labels(labels_dict, data_names) for labels_dict in self.labels_dict_list]
            
        self.label_to_idx, self.labels_one_hot = _parse_labels(complete_labels_dict_list)
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.labels_one_hot = torch.tensor(self.labels_one_hot, dtype=self.dtype, device=self.device)


    @abstractmethod
    def __len__(self):
        pass


    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        pass

    
    def to_label(
        self,
        labels: torch.Tensor | np.ndarray | list,
    ) -> torch.Tensor:
        """Converts the one-hot encoded labels (or their probability) to the original labels.
        
        Args:
            labels (torch.Tensor | np.ndarray | list): One-hot encoded labels or their probability.
            
        Returns:
            torch.Tensor: Original labels.
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        elif isinstance(labels, list):
            labels = np.array(labels)
        if len(labels.shape) == 1:
            labels = np.expand_dims(labels, axis=0)
            
        return np.array([self.idx_to_label[np.argmax(l).item()] for l in labels])


    @abstractmethod
    def get_num_residues(self) -> int:
        pass


    @abstractmethod
    def get_num_states(self) -> int:
        pass

    
    @abstractmethod
    def get_num_classes(self) -> int:
        pass


    @abstractmethod
    def get_effective_size(self) -> int:
        pass


    @abstractmethod
    def shuffle(self) -> None:
        pass


class DatasetCat(DatasetDCA, annaDataset):
    """Dataset class for processing multi-sequence alignments of biological sequences.
    """
    def __init__(
        self,
        path_data: str | Path,
        path_ann: str | Path,
        path_weights: str | Path | None = None,
        alphabet: str = "protein",
        clustering_th: float = 0.8,
        no_reweighting: bool = True,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize the dataset.

        Args:
            path_data (str | Path): Path to multi sequence alignment in fasta format.
            path_ann (str | Path | None): Path to the file containing the labels of the sequences.
            path_weights (str | Path, optional): Path to the file containing the importance weights of the sequences. If None, the weights are computed automatically.
            alphabet (str, optional): Selects the type of encoding of the sequences. Default choices are ("protein", "rna", "dna"). Defaults to "protein".
            clustering_th (float, optional): Sequence identity threshold for clustering the sequences. Defaults to 0.8.
            no_reweighting (bool, optional): If True, the sequences are not reweighted. Defaults to True.
            device (torch.device, optional): Device to be used. Defaults to "cpu".
            dtype (torch.dtype, optional): Data type of the data. Defaults to torch.float32.
        """
        annaDataset.__init__(
            self,
            path_ann=path_ann,
            device=device,
            dtype=dtype,
        )
        DatasetDCA.__init__(
            self,
            path_data=path_data,
            path_weights=path_weights,
            alphabet=alphabet,
            clustering_th=clustering_th,
            no_reweighting=no_reweighting,
            device=device,
            dtype=dtype,
        )
        
        # Parse the labels
        annaDataset.parse_labels(self, self.names)
        
        # Move data to device
        self.num_states = self.get_num_states()
        self.data_one_hot = one_hot(
            self.data,
            num_classes=self.num_states,
        ).to(device=device, dtype=dtype)
        self.weights = self.weights.view(-1, 1)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx: int) -> Any:
        return {
            "visible": self.data_one_hot[idx],
            "label": self.labels_one_hot[idx],
            "weight": self.weights[idx],
        }
    
    
    def get_num_residues(self) -> int:
        """Returns the number of residues (L) in the multi-sequence alignment.

        Returns:
            int: Length of the MSA.
        """
        return self.data.shape[1]
    
    
    def get_num_states(self) -> int:
        """Returns the number of states (q) in the alphabet.

        Returns:
            int: Number of states.
        """
        return DatasetDCA.get_num_states(self)
    
    
    def get_num_classes(self) -> int:
        """Returns the number of categories in the labels.

        Returns:
            int: Number of categories.
        """
        return self.labels_one_hot.shape[1]
    
    
    def get_effective_size(self) -> int:
        """Returns the effective size (Meff) of the dataset.

        Returns:
            int: Effective size of the dataset.
        """
        return int(self.weights.sum())
    
    
    def shuffle(self) -> None:
        """Shuffles the dataset.
        """
        perm = torch.randperm(len(self.data))
        self.data = self.data[perm]
        self.data_one_hot = self.data_one_hot[perm]
        self.names = self.names[perm]
        self.weights = self.weights[perm]
        self.labels_one_hot = self.labels_one_hot[perm]
        

class DatasetBin(annaDataset):
    """Dataset class for processing binary data.
    """
    
    def __init__(
        self,
        path_data: str | Path,
        path_ann: str | Path,
        path_weights: str | Path | None = None,
        clustering_th: float = 0.8,
        no_reweighting: bool = True,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize the dataset.

        Args:
            path_data (str | Path): Path to the file containing the binary data.
            path_ann (str | Path): Path to the file containing the labels of the data.
            path_weights (str | Path | None, optional): Path to the file containing the importance weights of the sequences. If None, the weights are computed automatically.
            clustering_th (float, optional): Sequence identity threshold for clustering the sequences. Defaults to 0.8.
            no_reweighting (bool, optional): If True, the sequences are not reweighted. Defaults to True.
            device (torch.device, optional): Device to be used. Defaults to "cpu".
            dtype (torch.dtype, optional): Data type of the data. Defaults to torch.float32.
        """
        annaDataset.__init__(
            self,
            path_ann=path_ann,
            device=device,
            dtype=dtype,
        )
        
        self.data = torch.tensor(
            np.loadtxt(path_data),
            dtype=dtype,
            device=device,
        )
        self.names = np.arange(len(self.data)).astype(str)
        if no_reweighting:
            self.weights = torch.ones(len(self.data), 1, dtype=dtype, device=device).view(-1, 1)
        else:
            if path_weights is not None:
                self.weights = torch.tensor(
                    np.loadtxt(path_weights),
                    dtype=dtype,
                    device=device,
                ).view(-1, 1)
            else:
                self.weights = compute_weights(
                    data=self.data,
                    th=clustering_th,
                    device=device,
                    dtype=dtype,
                ).view(-1, 1)
        self.tokens = get_tokens("01")
        
        # parse the labels
        annaDataset.parse_labels(self, self.names)
        
        print(f"Dataset imported: M = {self.data.shape[0]}, L = {self.data.shape[1]}, M_eff = {int(self.weights.sum())}.")
   
        
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx: int) -> Any:
        return {
            "visible": self.data[idx],
            "label": self.labels_one_hot[idx],
            "weight": self.weights[idx],
        }
    
    
    def get_num_residues(self) -> int:
        """Returns the number of residues (L) in the multi-sequence alignment.

        Returns:
            int: Length of the MSA.
        """
        return self.data.shape[1]
    
    
    def get_num_states(self) -> int:
        """Returns the number of states (q) in the alphabet.

        Returns:
            int: Number of states.
        """
        return 2
    
    
    def get_num_classes(self) -> int:
        """Returns the number of categories in the labels.

        Returns:
            int: Number of categories.
        """
        return self.labels_one_hot.shape[1]
    
    
    def get_effective_size(self) -> int:
        """Returns the effective size (Meff) of the dataset.

        Returns:
            int: Effective size of the dataset.
        """
        return int(self.weights.sum())
    
    
    def shuffle(self) -> None:
        """Shuffles the dataset.
        """
        perm = torch.randperm(len(self.data))
        self.data = self.data[perm]
        self.names = self.names[perm]
        self.weights = self.weights[perm]
        self.labels_one_hot = self.labels_one_hot[perm]
        
        
def get_dataset(
    path_data: str | Path,
    path_ann: str | Path,
    path_weights: str | Path | None = None,
    alphabet: str = "protein",
    clustering_th: float = 0.8,
    no_reweighting: bool = True,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> annaDataset:
    """Returns the proper dataset object based on the input data format.

    Args:
        path_data (str | Path): Path to the data file.
        path_labels (str | Path): Path to the labels file.
        path_weights (str | Path | None, optional): Path to the weights file. Defaults to None.
        alphabet (str, optional): Alphabet for encoding the data. It is uneffective for binary data. Defaults to "protein".
        device (torch.device, optional): Device. Defaults to torch.device("cpu").
        dtype (torch.dtype, optional): Data type. Defaults to torch.float32.

    Returns:
        aiDataset: Initilized dataset object.
    """
    # Check if the data is in fasta format
    with open(path_data, "r") as f:
        first_line = f.readline()
        if first_line.startswith(">"):
            return DatasetCat(
                path_data=path_data,
                path_ann=path_ann,
                path_weights=path_weights,
                alphabet=alphabet,
                clustering_th=clustering_th,
                no_reweighting=no_reweighting,
                device=device,
                dtype=dtype,
            )
        else:
            return DatasetBin(
                path_data=path_data,
                path_ann=path_ann,
                path_weights=path_weights,
                clustering_th=clustering_th,
                no_reweighting=no_reweighting,
                device=device,
                dtype=dtype,
            )