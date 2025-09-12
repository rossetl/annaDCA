from typing import Dict
import torch

from annadca.rbm.binary.stats import get_freq_single_point


@torch.jit.script
def _compute_gradient(
    data: Dict[str, torch.Tensor],
    chains: Dict[str, torch.Tensor],
    params: Dict[str, torch.Tensor],
    pseudo_count: float = 0.0,
    centered: bool = True,
    lambda_l1: float = 0.0,
    lambda_l2: float = 0.0,
    eta: float = 1.0,
) -> None:
    """Computes the gradient of the Log-Likelihood for the binary annaRBM.

    Args:
        data (Dict[str, torch.Tensor]): Data batch.
        chains (Dict[str, torch.Tensor]): Chains.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
        centered (bool, optional): Whether to use centered gradients. Defaults to True.
        lambda_l1 (float, optional): L1 regularization weight. Defaults to 0.0.
        lambda_l2 (float, optional): L2 regularization weight. Defaults to 0.0.
        eta (float, optional): Relative contribution of the label term. Defaults to 1.0.
    """
    # Normalize the weights
    data["weight"] = data["weight"] / data["weight"].sum()
    nchains = len(chains["visible"])

    # Averages over data and generated samples
    v_data_mean = get_freq_single_point(data["visible"], weights=data["weight"], pseudo_count=pseudo_count)
    h_data_mean = get_freq_single_point(data["hidden"], weights=data["weight"], pseudo_count=pseudo_count)
    l_data_mean = get_freq_single_point(data["label"], weights=data["weight"], pseudo_count=pseudo_count)
    v_gen_mean = chains["visible"].mean(0)
    h_gen_mean = chains["hidden"].mean(0)
    l_gen_mean = chains["label"].mean(0)

    if centered:
        # Centered variables
        v_data_centered = data["visible"] - v_data_mean
        h_data_centered = data["hidden"] - h_data_mean
        l_data_centered = data["label"] - l_data_mean
        v_gen_centered = chains["visible"] - v_data_mean
        h_gen_centered = chains["hidden"] - h_data_mean
        l_gen_centered = chains["label"] - l_data_mean

        # Gradient
        grad_weight_matrix = (
            (v_data_centered * data["weight"]).T @ h_data_centered
        ) - (v_gen_centered.T @ h_gen_centered) / nchains
        grad_label_matrix = (
            (l_data_centered * data["weight"]).T @ h_data_centered
        ) - (l_gen_centered.T @ h_gen_centered) / nchains - lambda_l2 * params["label_matrix"]
        grad_vbias = v_data_mean - v_gen_mean - (grad_weight_matrix @ h_data_mean)
        grad_hbias = h_data_mean - h_gen_mean - (v_data_mean @ grad_weight_matrix)
        grad_lbias = l_data_mean - l_gen_mean - (grad_label_matrix @ h_data_mean)
        
        # regularization
        grad_weight_matrix -= lambda_l1 * params["weight_matrix"].sign() + lambda_l2 * params["weight_matrix"]
        #grad_label_matrix -= lambda_l1 * params["label_matrix"].sign() + lambda_l2 * params["label_matrix"]

    else:
        # Gradient
        grad_weight_matrix = (
            (data["visible"] * data["weight"]).T @ data["hidden"]
        ) - (chains["visible"].T @ chains["hidden"]) / nchains
        grad_label_matrix = (
            (data["label"] * data["weight"]).T @ data["hidden"]
        ) - (chains["label"].T @ chains["hidden"]) / nchains
        grad_vbias = v_data_mean - v_gen_mean
        grad_hbias = h_data_mean - h_gen_mean
        grad_lbias = l_data_mean - l_gen_mean
        
        # regularization
        grad_weight_matrix -= lambda_l1 * params["weight_matrix"].sign() + lambda_l2 * params["weight_matrix"]
        #grad_label_matrix -= lambda_l1 * params["label_matrix"].sign() + lambda_l2 * params["label_matrix"]

    # Attach the gradients to the parameters
    params["weight_matrix"].grad.set_(grad_weight_matrix)
    params["label_matrix"].grad.set_(eta * grad_label_matrix)
    params["vbias"].grad.set_(grad_vbias)
    params["hbias"].grad.set_(grad_hbias)
    params["lbias"].grad.set_(eta * grad_lbias)
