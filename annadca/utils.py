import numpy as np
from typing import List, Tuple
import h5py


def _encode_labels(
    dict_labels: dict,
    start_idx: int,
) -> Tuple[dict, np.array]:
    """Take a dictionary of labels and return a dictionary of label to index and a one-hot encoded array of labels.

    Args:
        dict_labels (dict): Labels dictionary.
        start_idx (int): Starting index for the one-hot encoding. Used if multiple categorizations are needed.

    Returns:
        Tuple[dict, np.array]: Dictionary of label to index in the one-hot encoding and one-hot encoded array of labels.
    """
    unique_labels = list(set(dict_labels.values()))
    num_categ = len(unique_labels)
    label_to_index = {lab : i for i, lab in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_index[n] for n in dict_labels.values()])
    one_hot_labels = np.eye(num_categ)[numeric_labels]
    # Shift index to start_idx
    label_to_index = {k: v + start_idx for k, v in label_to_index.items()}
    return label_to_index, one_hot_labels


def _parse_labels(labels_dict_list: List[dict]) -> Tuple[dict, np.array]:
    """Parse a list of dictionaries of labels and return a dictionary of label to index and a one-hot encoded array of labels.

    Args:
        labels_dict_list (List[dict]): List of dictionaries of labels.

    Returns:
        Tuple[dict, np.array]: Dictionary of label to index in the one-hot encoding and one-hot encoded array of labels.
    """
    start_idx = 0
    label_to_idx = {}
    one_hot_labels = []
    
    for labels_dict in labels_dict_list:
        label_to_idx_, one_hot_labels_ = _encode_labels(labels_dict, start_idx)
        label_to_idx.update(label_to_idx_)
        one_hot_labels.append(one_hot_labels_)
        start_idx += len(label_to_idx_)
    
    one_hot_labels = np.concatenate(one_hot_labels, axis=1)
    
    return label_to_idx, one_hot_labels


def get_saved_updates(filename: str) -> np.ndarray:
    """Returns the list of indices of the saved updates in the h5 archive.

    Args:
        filename (str): Path to the h5 archive.

    Returns:
        np.ndarray: Array of the indices of the saved updates.
    """
    updates = []
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            if "update" in key:
                update = int(key.replace("update_", ""))
                updates.append(update)
    return np.sort(np.array(updates))