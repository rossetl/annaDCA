from typing import Any

import numpy as np
import pandas as pd
import torch
from adabmDCA.fasta import compute_weights, encode_sequence, get_tokens, import_from_fasta
from torch.nn.functional import one_hot
from torch.utils.data import Dataset


class annaDataset(Dataset):
    def __init__(
        self,
        path_data: str,
        path_ann: str | None = None,
        column_names: str = "name",
        column_sequences: str = "sequence",
        column_labels: str = "label",
        is_binary: bool = False,
        alphabet: str = "protein",
        clustering_th: float = 0.8,
        no_reweighting: bool = True,
        path_weights: str | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        path_data = str(path_data)
        self.device = device
        self.dtype = dtype
        self.data = pd.DataFrame()
        self.data_one_hot = torch.tensor([], dtype=dtype, device=device)
        self.labels_one_hot = torch.tensor([], dtype=dtype, device=device)
        self.weights = torch.tensor([], dtype=dtype, device=device)
        self.is_binary = is_binary
        self.column_names = column_names
        self.column_sequences = column_sequences
        self.column_labels = column_labels
        self.L = 0
        self.q = 0

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
                raise ValueError(
                    "The annotations file must be provided if data are taken from a text file."
                )
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
                    raise ValueError(
                        "Input data not recognized as a text file nor a .csv file. Invalid FASTA format."
                    )
            if path_ann is not None:
                self.parse_csv(
                    path_ann=path_ann,
                    column_names=column_names,
                    column_sequences=column_sequences,
                    column_labels=column_labels,
                )
            else:
                raise ValueError(
                    "The annotations file must be provided if data are taken from a fasta file."
                )

        if is_binary:
            self.tokens = "01"
        else:
            self.tokens = get_tokens(alphabet)

        sequences = self.data[column_sequences]
        if sequences.empty or sequences.isna().any():
            raise ValueError("Sequences must be nonempty and cannot be missing.")
        lengths = sequences.str.len()
        if (lengths == 0).any() or lengths.nunique() != 1:
            raise ValueError("Sequences must be nonempty and aligned to the same length.")
        if self.data[column_labels].notna().sum() == 0:
            raise ValueError("At least one annotated sequence is required.")

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

        if (
            len(self.weights) != len(self.data)
            or not torch.isfinite(self.weights).all()
            or (self.weights < 0).any()
            or self.weights.sum() <= 0
        ):
            raise ValueError(
                "Weights must be finite, nonnegative, match the dataset, and have a positive sum."
            )

        if not is_binary:
            self.data_one_hot = one_hot(self.data_one_hot.long(), num_classes=self.q).to(self.dtype)

    def parse_labels(self):
        labels = self.data[self.column_labels]
        unique_labels = np.unique(labels.dropna())
        self.idx_to_label = {i: label for i, label in enumerate(unique_labels)}
        self.label_to_idx = {label: i for i, label in self.idx_to_label.items()}
        label_indices = labels.map(self.label_to_idx).fillna(-1).astype(int)
        num_classes = len(unique_labels)
        one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
        valid = label_indices != -1
        one_hot[valid, label_indices[valid]] = 1.0
        self.labels_one_hot = torch.tensor(
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

        df = pd.read_csv(
            path_ann,
            dtype={column_names: str, column_sequences: str, column_labels: str},
            keep_default_na=False,
            na_values=[""],
        )
        required = [column_names, column_labels]
        if len(self.data) == 0:
            required.append(column_sequences)
        for column in required:
            if column not in df.columns:
                raise ValueError(f"Column {column!r} not found in {path_ann}.")
        if df[column_names].isna().any() or df[column_names].duplicated().any():
            raise ValueError("Sequence identifiers must be present and unique.")

        # If all data comes from the csv file:
        if len(self.data) == 0:
            self.data = df.filter(items=[column_names, column_sequences, column_labels])
        # Else, names and data have been already parsed from a fasta file:
        else:
            # merge df with self.data on the names
            self.data = pd.merge(
                self.data,
                df.filter(items=[column_names, column_labels]),
                on=column_names,
                how="left",
                validate="one_to_one",
            )

    def parse_fasta(
        self,
        path_data: str,
    ):
        names, sequences = import_from_fasta(path_data)
        self.data = pd.DataFrame.from_dict(
            {self.column_names: names, self.column_sequences: sequences}
        )

    def parse_text(
        self,
        path_data: str,
    ):
        if not self.is_binary:
            raise ValueError("Text datasets require is_binary=True (--is_binary).")
        data = np.loadtxt(path_data, ndmin=2)
        if not np.isin(data, [0, 1]).all():
            raise ValueError("Binary data must contain only 0 and 1.")
        names = np.arange(len(data)).astype(str)
        sequences = ["".join(str(int(value)) for value in row) for row in data]
        self.data = pd.DataFrame({self.column_names: names, self.column_sequences: sequences})

    def to_label(
        self,
        labels: torch.Tensor | np.ndarray | list,
    ) -> np.ndarray:
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
        return self.labels_one_hot.shape[1]

    def get_effective_size(self) -> float:
        """Returns the effective size (Meff) of the dataset.

        Returns:
            int: Effective size of the dataset.
        """
        return float(self.weights.sum())

    def shuffle(self) -> None:
        """Shuffles the dataset."""
        perm = torch.randperm(len(self.data))
        # shuffle the data, one-hot encoded sequences, names, weights, and labels
        self.data = self.data.iloc[perm.numpy()].reset_index(drop=True)
        self.data_one_hot = self.data_one_hot[perm]
        self.weights = self.weights[perm]
        self.labels_one_hot = self.labels_one_hot[perm]
