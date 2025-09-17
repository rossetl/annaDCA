import numpy as np
from typing import List, Tuple
import h5py
import torch
import os


def _complete_labels(
    dict_labels: dict,
    names: list | np.ndarray,
) -> dict:
    """Goes through 'names' and returns a dictionary of 'name -> label' with 'None' for missing labels or labels
    marked as 'None' in 'dict_labels'.
    
    Args:
        dict_labels (dict): Dictionary of 'name -> label'.
        names (list | np.ndarray): List of names to complete.
        
    Returns:
        dict: Dictionary of 'name -> label' with 'None' for missing labels or labels marked as 'None' in 'dict_labels'.
    """
    name_to_label = {}
    for n in names:
        if n not in dict_labels:
            name_to_label[n] = None
        elif dict_labels[n] == "None":
            name_to_label[n] = None
        else:
            name_to_label[n] = dict_labels[n]
    return name_to_label


def _encode_labels(
    dict_labels: dict,
    start_idx: int,
) -> Tuple[dict, np.ndarray]:
    """Take a dictionary of 'name -> label' and return a dictionary of 'label -> one-hot index' and a one-hot encoded array of labels.
    When the label is marked as 'None', it is represented with an array of zeros.

    Args:
        dict_labels (dict): name to label dictionary.
        start_idx (int): Starting index for the one-hot encoding. Used if multiple categorizations are needed.

    Returns:
        Tuple[dict, np.array]: Dictionary of label to index in the one-hot encoding and one-hot encoded array of labels.
    """
    # Get all unique labels that are not None
    unique_labels = np.unique([lab for lab in dict_labels.values() if lab != None])
    num_categ = len(unique_labels)
    # None label is represented as an array of zeros
    one_hot_transform = np.vstack([np.zeros(num_categ), np.eye(num_categ)])
    label_to_index = {lab.item() : (i + 1) for i, lab in enumerate(unique_labels)}
    label_to_index[None] = 0
    numeric_labels = np.array([label_to_index[n] for n in dict_labels.values()])
    one_hot_labels = one_hot_transform[numeric_labels]
    # Shift index to start_idx and subtract 1 to account for the None label
    label_to_index = {k: (v - 1 + start_idx) for k, v in label_to_index.items() if k != None}
    return label_to_index, one_hot_labels


def _parse_labels(labels_dict_list: List[dict]) -> Tuple[dict, np.ndarray]:
    """Parse a list of dictionaries of labels and return a dictionary of label to index and a one-hot encoded array of labels.

    Args:
        labels_dict_list (List[dict]): List of dictionaries of labels.

    Returns:
        Tuple[dict, np.ndarray]: Dictionary of label to index in the one-hot encoding and one-hot encoded array of labels.
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


def get_eigenvalues_history(
    checkpoints_repo: str,
    target_matrix: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each update in the file, return the eigenvalues of the target matrix.
    
    Args:
        checkpoints_repo (str): The repository where the checkpoints are stored.
        target_matrix (str): The matrix for which to compute the eigenvalues. Must be either 'weight_matrix' or 'label_matrix'.
        device (torch.device): The device on which to compute the eigenvalues.
        dtype (torch.dtype): The dtype of the eigenvalues.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: The updates indices and the eigenvalues.
    
    """
    if target_matrix not in ["weight_matrix", "label_matrix"]:
        raise ValueError("target_matrix must be either 'weight_matrix' or 'label_matrix'")
    updates = []
    eigenvalues = []
    for checkpoint_file in os.listdir(checkpoints_repo):
        checkpoint = torch.load(os.path.join(checkpoints_repo, checkpoint_file), map_location=device)
        model_params = checkpoint['model_state_dict']
        update = checkpoint['update']
        matrix = model_params[target_matrix].to(device=device, dtype=dtype)
        matrix = matrix.reshape(-1, matrix.shape[-1])
        eig = torch.linalg.svdvals(matrix).cpu().numpy()
        eigenvalues.append(eig.reshape(*eig.shape, 1))
        updates.append(update)
    
    # Sort the results
    sorting = np.argsort(updates)
    updates = np.array(updates)[sorting]
    eigenvalues = np.array(np.hstack(eigenvalues).T)[sorting]     
    return updates, eigenvalues


def mutual_information(
    visible: torch.Tensor,
    label: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Estimates the mutual information between the visible and the label units.
    
    Args:
        visible (torch.Tensor): Visible units.
        label (torch.Tensor): Label units.
        
    Returns:
        torch.Tensor: The mutual information between the visible and the label units
    """
    nchains = len(visible)
    fl = label.mean(0)
    fi = visible.mean(0)
    fil = label.T @ visible / nchains
    Iil = fil * (torch.log(1e-8 + fil) - torch.log(1e-8 + (fl.unsqueeze(1) @ fi.unsqueeze(0))))
    
    return Iil
