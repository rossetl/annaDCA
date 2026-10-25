from typing import Any, Iterator, Dict, Optional
import warnings
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
import torch
from torch.nn.functional import one_hot
from annadca.utils.sequences import get_tokens, compute_weights, encode_sequence
from annadca.utils.fasta import read_fasta

def _merge_duplicates(
    df: pd.DataFrame,
    key: str,
    column_labels: str,
    continuous_labels: bool,
) -> pd.DataFrame:
    """Keeps one row for each value of `key`. Duplicates take the first non-missing label; if they carry different labels,
    the label is set to their mean for continuous labels, and to missing for categorical labels.

    Args:
        df (pd.DataFrame): Data frame.
        key (str): Column identifying the duplicates (names or sequences).
        column_labels (str): Column of the labels.
        continuous_labels (bool): Whether the labels are continuous.

    Returns:
        pd.DataFrame: Data frame without duplicates.
    """
    if not df[key].duplicated().any():
        return df
    df = df.copy()
    groups = df.groupby(key, sort=False)[column_labels]
    conflict = groups.transform("nunique") > 1
    merged = groups.transform("first")
    if continuous_labels:
        merged = merged.where(~conflict, groups.transform("mean"))
    else:
        merged = merged.where(~conflict, np.nan)
    df[column_labels] = merged
    num_conflicts = df.loc[conflict, key].nunique()
    if num_conflicts > 0:
        action = "averaged" if continuous_labels else "set to missing"
        warnings.warn(f"{num_conflicts} duplicated entries of '{key}' have different labels: their label is {action}.")
    return df.drop_duplicates(subset=[key]).reset_index(drop=True)


def _to_python(x):
    """Converts numpy scalars to python scalars."""
    return x.item() if isinstance(x, np.generic) else x


class AnnaDataset(Dataset):
    """Dataset of aligned sequences with their annotations.

    Attributes:
        data_one_hot (torch.Tensor): Sequences, binary (M, L) or one-hot encoded (M, L, q).
        labels (torch.Tensor): Labels, one-hot encoded (M, num_classes) for categorical labels (all-zero rows for the
            missing ones), or standardized (M, 1) for continuous labels.
        weights (torch.Tensor): Weights of the sequences (M, 1).
    """
    def __init__(
        self,
        path_data: str,
        path_ann: str | None = None,
        column_names: str = "name",
        column_sequences: str = "sequence",
        column_labels: str = "label",
        is_binary: bool = False,
        continuous_labels: bool = False,
        alphabet: str = "protein",
        clustering_th: float = 0.8,
        no_reweighting: bool = True,
        path_weights: str | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.dtype = dtype
        self.data = pd.DataFrame()
        self.data_one_hot = torch.tensor([], dtype=dtype, device=device)
        self.labels = torch.tensor([], dtype=dtype, device=device)
        self.weights = torch.tensor([], dtype=dtype, device=device)
        self.is_binary = is_binary
        self.column_names = column_names
        self.column_sequences = column_sequences
        self.column_labels = column_labels
        self.continuous_labels = continuous_labels
        self.idx_to_label = {}
        self.label_to_idx = {}
        self.L = 0
        self.q = 0
        self.labels_mean = torch.tensor(0.0, dtype=dtype, device=device)
        self.labels_std = torch.tensor(1.0, dtype=dtype, device=device)
        
        if is_binary:
            self.tokens = "01"
        else:
            self.tokens = get_tokens(alphabet)

        # check if path_data points to a text file
        if path_data.endswith(".txt") or path_data.endswith(".dat"):
            self.parse_text(path_data)
            if path_ann is not None:
                self.parse_csv(
                    path_ann=path_ann,
                    column_names=column_names,
                    column_sequences=column_sequences,
                    column_labels=column_labels,
                )
            else:
                raise ValueError("The annotations file must be provided if data are taken from a text file.")
        elif path_data.endswith(".csv"):
            self.parse_csv(
                path_ann=path_data,
                column_names=column_names,
                column_sequences=column_sequences,
                column_labels=column_labels,
            )
        else:
            # check that the input file is a valid fasta file
            with open(path_data, "r") as f:
                first_line = f.readline()
                if first_line.startswith(">"):
                    self.parse_fasta(path_data)
                else:
                    raise ValueError("Input data not recognized as a text file nor a .csv file. Invalid FASTA format.")
            if path_ann is not None:
                self.parse_csv(
                    path_ann=path_ann,
                    column_names=column_names,
                    column_sequences=column_sequences,
                    column_labels=column_labels,
                )
            else:
                raise ValueError("The annotations file must be provided if data are taken from a fasta file.")
            
        # Filter rows where sequences have unknown letters
        tokens_set = set(self.tokens)
        mask = [isinstance(seq, str) and set(seq) <= tokens_set for seq in self.data[column_sequences].values]
        self.data = self.data[mask].reset_index(drop=True)
        print(f"Filtered out {len(mask) - sum(mask)} sequences with unknown letters. Remaining sequences: {sum(mask)}.")
        if len(self.data) == 0:
            raise ValueError("No valid sequences found in the dataset. Check the alphabet and the input format.")
        lengths = self.data[column_sequences].str.len()
        if lengths.nunique() > 1:
            raise ValueError("The sequences must be aligned (all of the same length).")

        # take only unique sequences
        self.data = _merge_duplicates(self.data, column_sequences, column_labels, continuous_labels)

        # encode sequences
        self.data_one_hot = torch.tensor(
                encode_sequence(list(self.data[column_sequences].values), self.tokens),
                dtype=self.dtype,
                device=self.device,
            )
        # one-hot representation of sequences and labels
        self.parse_labels()
        
        self.L = self.data_one_hot.shape[1]
        self.q = len(self.tokens)
            
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
                # Here, data_one_hot is still a (M, L) tensor
                self.weights = compute_weights(
                    data=self.data_one_hot,
                    th=clustering_th,
                    device=device,
                    dtype=dtype,
                ).view(-1, 1)
                
        if not is_binary:
            self.data_one_hot = one_hot(self.data_one_hot.long(), num_classes=self.q).to(self.dtype)


    def parse_labels(self):
        labels = self.data[self.column_labels]
        if self.continuous_labels:
            self.labels = torch.tensor(
                labels.values.astype(np.float64),
                dtype=self.dtype,
                device=self.device,
            ).unsqueeze(1)
            # standardize labels, ignoring missing values
            missing = torch.isnan(self.labels)
            valid_labels = self.labels[~missing]
            self.labels_mean = valid_labels.mean().view(1)
            self.labels_std = valid_labels.std().view(1)
            self.labels = (self.labels - self.labels_mean) / self.labels_std
            if missing.any():
                warnings.warn(f"{missing.sum().item()} sequences have a missing label: it is set to the mean label value.")
                self.labels[missing] = 0.0
        else:
            unique_labels = np.unique(labels.dropna())
            self.idx_to_label = {i: _to_python(label) for i, label in enumerate(unique_labels)}
            self.label_to_idx = {label: i for i, label in self.idx_to_label.items()}
            label_indices = labels.map(self.label_to_idx).fillna(-1).astype(int)
            num_classes = len(unique_labels)
            one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
            valid = label_indices != -1
            one_hot[valid, label_indices[valid]] = 1.0
            if not valid.all():
                print(f"{(~valid).sum()} sequences have a missing label.")
            self.labels = torch.tensor(
                one_hot,
                dtype=self.dtype,
                device=self.device,
            )
    

    def parse_csv(
        self,
        path_ann: str,
        column_names: str = "name",
        column_sequences: str = "sequence",
        column_labels: str = "label",
    ):
        
        df = pd.read_csv(path_ann, na_values=[""], dtype={column_names: str, column_sequences: str})
        assert column_names in df.columns, f"Column {column_names} not found in the labels file."
        assert column_labels in df.columns, f"Column {column_labels} not found in the labels file."
        df = _merge_duplicates(df, column_names, column_labels, self.continuous_labels)

        # If all data comes from the csv file:
        if len(self.data) == 0:
            assert column_sequences in df.columns, f"Column {column_sequences} not found in the labels file."
            self.data = df.filter(items=[column_names, column_sequences, column_labels])
        # Else, names and data have been already parsed from a fasta file:
        else:
            # merge df with self.data on the names
            self.data = pd.merge(
                self.data,
                df.filter(items=[column_names, column_labels]),
                on=column_names,
                how="left",
            )
        
            
    def parse_fasta(
        self,
        path_data: str,
    ):
        names, sequences = read_fasta(path_data)
        self.data = pd.DataFrame.from_dict(
            {self.column_names: names, self.column_sequences: sequences}
        )


    def parse_text(
        self,
        path_data: str,
    ):
        data = np.atleast_2d(np.loadtxt(path_data)).astype(int)
        # Sequences are stored as strings of tokens, as for the categorical data
        sequences = ["".join(str(x) for x in row) for row in data]
        names = np.arange(len(data)).astype(str)
        self.data = pd.DataFrame.from_dict(
            {self.column_names: names, self.column_sequences: sequences}
        )
    
    def to_label(
        self,
        labels: torch.Tensor | np.ndarray | list,
    ) -> np.ndarray:
        """Converts the one-hot encoded labels (or their probability) to the original labels.
        For continuous labels, maps the standardized labels back to the original scale.
        
        Args:
            labels (torch.Tensor | np.ndarray | list): One-hot encoded labels or their probability.
            
        Returns:
            np.ndarray: Original labels.
        """
        if self.continuous_labels:
            labels = self.unstandardize_labels(torch.as_tensor(labels, dtype=self.dtype, device=self.device))
            return labels.view(-1).cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        elif isinstance(labels, list):
            labels = np.array(labels)
        if len(labels.shape) == 1:
            labels = np.expand_dims(labels, axis=0)
            
        return np.array([self.idx_to_label[np.argmax(l).item()] for l in labels])
    
    
    def standardize_labels(self, x):
        """Takes as input the original labels and returns the standardized labels.

        Args:
            x (torch.Tensor): Original labels.

        Returns:
            torch.Tensor: Standardized labels.
        """
        if self.continuous_labels:
            return (x - self.labels_mean) / self.labels_std
        else:
            return x


    def unstandardize_labels(self, x):
        """Takes as input the standardized labels and returns the original labels.

        Args:
            x (torch.Tensor): Standardized labels.

        Returns:
            torch.Tensor: Original labels.
        """
        if self.continuous_labels:
            return x * self.labels_std + self.labels_mean
        else:
            return x
    
    
    def to_one_hot(
        self,
        labels: torch.Tensor | np.ndarray | list,
    ) -> torch.Tensor:
        """Converts the original labels to one-hot encoded labels.
        
        Args:
            labels (torch.Tensor | np.ndarray | list): Original labels.
            
        Returns:
            torch.Tensor: One-hot encoded labels.
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy().tolist()
        elif isinstance(labels, np.ndarray):
            labels = labels.tolist()
        
        one_hot_labels = np.zeros((len(labels), len(self.idx_to_label)), dtype=np.float32)
        for i, label in enumerate(labels):
            if label in self.label_to_idx:
                one_hot_labels[i, self.label_to_idx[label]] = 1.0
        return torch.tensor(one_hot_labels, dtype=self.dtype, device=self.device)


    @property
    def labels_one_hot(self) -> torch.Tensor:
        """Alias of `labels`, kept for backward compatibility."""
        return self.labels


    @labels_one_hot.setter
    def labels_one_hot(self, value: torch.Tensor):
        self.labels = value


    def metadata(self) -> Dict[str, Any]:
        """Information needed to interpret the model outputs, made of plain python types (saved in the checkpoints)."""
        return {
            "tokens": self.tokens,
            "is_binary": self.is_binary,
            "continuous_labels": self.continuous_labels,
            "idx_to_label": dict(self.idx_to_label),
            "labels_mean": float(self.labels_mean.view(-1)[0]),
            "labels_std": float(self.labels_std.view(-1)[0]),
        }


    def iterate_batches(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Infinite iterator over random batches. The samples are visited in a new random order at each epoch,
        so that all of them are used; a batch at the boundary between two epochs takes samples from both.

        Args:
            batch_size (int): Batch size, at most the size of the dataset.
            generator (torch.Generator, optional): Random number generator.

        Yields:
            Dict[str, torch.Tensor]: Batch with keys 'visible', 'label' and 'weight'.
        """
        assert 0 < batch_size <= len(self), f"The batch size must be between 1 and the dataset size ({len(self)})."
        indices = torch.empty(0, dtype=torch.long)
        while True:
            if len(indices) < batch_size:
                indices = torch.cat([indices, torch.randperm(len(self), generator=generator)])
            batch, indices = indices[:batch_size].to(self.data_one_hot.device), indices[batch_size:]
            yield self[batch]


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx: int) -> Any:
        return {
            "visible": self.data_one_hot[idx],
            "label": self.labels[idx],
            "weight": self.weights[idx],
        }
    
    
    def get_num_residues(self) -> int:
        """Returns the number of residues (L) in the multi-sequence alignment.

        Returns:
            int: Length of the MSA.
        """
        return self.L
    
    
    def get_num_states(self) -> int:
        """Returns the number of states (q) in the alphabet.

        Returns:
            int: Number of states.
        """
        return self.q
    
    
    def get_num_classes(self) -> int:
        """Returns the number of categories in the labels.

        Returns:
            int: Number of categories.
        """
        if self.continuous_labels:
            return 1
        else:
            return self.labels.shape[1]
    
    
    def get_effective_size(self) -> float:
        """Returns the effective size (Meff) of the dataset.

        Returns:
            float: Effective size of the dataset.
        """
        return float(self.weights.sum())
    
    
    def shuffle(self) -> None:
        """Shuffles the dataset.
        """
        perm = torch.randperm(len(self.data))
        # shuffle the data, one-hot encoded sequences, names, weights, and labels
        self.data = self.data.iloc[perm.numpy()].reset_index(drop=True)
        self.data_one_hot = self.data_one_hot[perm]
        self.weights = self.weights[perm]
        self.labels = self.labels[perm]


# Backward compatible name
annaDataset = AnnaDataset
