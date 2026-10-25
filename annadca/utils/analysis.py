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
        if not checkpoint_file.endswith(".pt"):
            continue
        checkpoint = torch.load(os.path.join(checkpoints_repo, checkpoint_file), map_location=device, weights_only=True)
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
        visible (torch.Tensor): Visible units of shape (N, L) or (N, L, q).
        label (torch.Tensor): Label units of shape (N, num_classes).
        
    Returns:
        torch.Tensor: The contributions to the mutual information between each label class and each visible unit (and state),
            of shape (num_classes, L) or (num_classes, L, q).
    """
    nchains = len(visible)
    fl = label.mean(0)
    fi = visible.mean(0)
    fil = torch.einsum("nc,n...->c...", label, visible) / nchains
    fl_fi = torch.einsum("c,...->c...", fl, fi)
    Iil = fil * (torch.log(1e-8 + fil) - torch.log(1e-8 + fl_fi))

    return Iil

def _connected_correlations(x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Connected correlation matrix of the columns of x (n, d)."""
    w = torch.full((len(x),), 1.0 / len(x), device=x.device, dtype=x.dtype) if weights is None else weights.view(-1) / weights.sum()
    mean = w @ x
    return (x * w.unsqueeze(1)).T @ x - torch.outer(mean, mean)


@torch.no_grad()
def two_point_pearson(
    data: torch.Tensor,
    chains: torch.Tensor,
    weights: torch.Tensor | None = None,
    max_features: int = 2000,
    seed: int = 0,
) -> float:
    """Pearson correlation coefficient between the connected two-point correlations of the data and of the chains,
    computed on the pairs of different sites. For long sequences, a random subset of `max_features` features is used.

    Args:
        data (torch.Tensor): Data of shape (M, L) or one-hot (M, L, q).
        chains (torch.Tensor): Chains with the same shape as the data (apart from the first dimension).
        weights (torch.Tensor, optional): Weights of the data. Defaults to uniform.
        max_features (int, optional): Maximum number of features (sites times states) used. Defaults to 2000.
        seed (int, optional): Seed used to select the features, so that the metric is comparable along the training.

    Returns:
        float: Pearson correlation coefficient.
    """
    num_states = data.shape[2] if data.dim() == 3 else 1
    data, chains = data.flatten(1), chains.flatten(1).to(data)
    site = torch.arange(data.shape[1], device=data.device) // num_states
    if data.shape[1] > max_features:
        generator = torch.Generator().manual_seed(seed)
        features = torch.randperm(data.shape[1], generator=generator)[:max_features].to(data.device)
        data, chains, site = data[:, features], chains[:, features], site[features]
    mask = site.unsqueeze(0) < site.unsqueeze(1)
    c_data = _connected_correlations(data, weights)[mask]
    c_chains = _connected_correlations(chains)[mask]
    return torch.corrcoef(torch.stack([c_data, c_chains]))[0, 1].item()


@torch.no_grad()
def label_accuracy(
    rbm,
    visible: torch.Tensor,
    label: torch.Tensor,
) -> float:
    """Fraction of the annotated sequences whose most probable label p(l | v) is the annotated one (categorical labels).

    Args:
        rbm (AnnaRBM): The model.
        visible (torch.Tensor): Sequences.
        label (torch.Tensor): One-hot labels; the all-zero rows (missing labels) are ignored.

    Returns:
        float: Accuracy (nan if there are no annotated sequences).
    """
    annotated = label.sum(dim=-1) > 0
    if not annotated.any():
        return float("nan")
    predictions = rbm.predict_labels(visible[annotated]).argmax(dim=-1)
    return (predictions == label[annotated].argmax(dim=-1)).float().mean().item()
