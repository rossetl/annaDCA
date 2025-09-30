import numpy as np
from typing import Tuple
import torch
import os

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