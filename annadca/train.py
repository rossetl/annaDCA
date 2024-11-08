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
) -> Dict[str, torch.Tensor]:
    """Computes the gradient of the parameters of the model and the Markov chains using the Persistent Contrastive Divergence algorithm.

    Args:
        rbm (annaRBM): RBM model to be trained.
        data_batch (Dict[str, torch.Tensor]): Batch of data.
        chains (Dict[str, torch.Tensor]): Persistent chains.
        gibbs_steps (int): Number of Alternating Gibbs Sampling steps.
        pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
        centered (bool, optional): Whether to use centered gradients. Defaults to True.

    Returns:
        Dict[str, torch.Tensor]: Updated chains.
    """
    # Compute the hidden magnetization of the data
    data_batch["hidden"] = rbm.sample_hiddens(**data_batch)["hidden_mag"]
    
    # Compute the gradient of the Log-Likelihood
    rbm.compute_gradient(
        data=data_batch,
        chains=chains,
        pseudo_count=pseudo_count,
        centered=centered,
    )
    
    # Update the persistent chains
    chains = rbm.sample(gibbs_steps=gibbs_steps, **chains)
    
    return chains