from typing import Dict
import torch

from annadca.classes import annaRBM


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

    Returns:
        Dict[str, torch.Tensor]: Updated chains.
    """
    # Compute the hidden magnetization of the data
    data_batch["hidden"] = rbm.sample_hiddens(**data_batch, beta=1.0)["hidden_mag"]

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
    chains = rbm.sample(gibbs_steps=gibbs_steps, **chains, beta=1.0)

    return chains