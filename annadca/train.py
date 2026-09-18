from typing import Dict

import torch

from annadca.classes import annaRBM
from annadca.rbm.missing_labels import marginalize_missing_labels


def pcd(
    rbm: annaRBM,
    data_batch: Dict[str, torch.Tensor],
    chains: Dict[str, torch.Tensor],
    gibbs_steps: int,
    pseudo_count: float = 0.0,
    centered: bool = True,
    lambda_l1: float = 0.0,
    lambda_l2: float = 0.0,
    eta: float = 1.0,
    marginalize_labels: bool = False,
    backend: str = "torch",
) -> Dict[str, torch.Tensor]:
    """Computes the gradient of the parameters of the model and the Markov chains using the Persistent Contrastive Divergence algorithm.

    Args:
        rbm (annaRBM): RBM model to be trained.
        data_batch (Dict[str, torch.Tensor]): Batch of data.
        chains (Dict[str, torch.Tensor]): Persistent chains.
        gibbs_steps (int): Number of Alternating Gibbs Sampling steps.
        pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
        centered (bool, optional): Whether to use centered gradients. Defaults to True.
        lambda_l1 (float, optional): L1 regularization weight. Defaults to 0.0.
        lambda_l2 (float, optional): L2 regularization weight. Defaults to 0.0.
        eta (float, optional): Relative contribution of the label term. Defaults to 1.0.
        marginalize_labels (bool): Exactly marginalize missing labels in the positive phase.
        backend (str): Sampling backend (torch or triton). Defaults to torch.

    Returns:
        Dict[str, torch.Tensor]: Updated chains.
    """
    # Compute the hidden magnetization of the data
    if marginalize_labels:
        data_batch = marginalize_missing_labels(data_batch, rbm.params)
    else:
        data_batch["hidden"] = rbm.sample_hiddens(**data_batch, beta=1.0, backend=backend)["hidden_mag"]

    # Compute the gradient of the Log-Likelihood
    rbm.compute_gradient(
        data=data_batch,
        chains=chains,
        pseudo_count=pseudo_count,
        centered=centered,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        eta=eta,
    )

    # Update the persistent chains
    chains = rbm.sample(gibbs_steps=gibbs_steps, **chains, beta=1.0, backend=backend)

    return chains
