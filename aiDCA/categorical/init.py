from typing import Dict
import numpy as np
import torch
from adabmDCA.functional import one_hot


def _init_parameters(
    num_visibles: int,
    num_hiddens: int,
    num_labels: int,
    num_states: int,
    frequencies_visibles: torch.Tensor = None,
    frequencies_labels: torch.Tensor = None,
    std_init: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Hidden biases are set to 0, visible and label biases are set to the independent-site model value using the frequencies of the dataset,
    and the weight and label matrices are initialized with a Gaussian distribution of sigma std_init.
    
    Args:
        num_visibles (int): Number of visible units.
        num_hiddens (int): Number of hidden units.
        num_labels (int): Number of labels.
        num_states (int): Number of states of the categorical variables.
        frequencies_visibles (torch.Tensor): Empirical frequencies of the visible units. Defaults to None.
        frequencies_labels (torch.Tensor): Empirical frequencies of the labels. Defaults to None.
        std_init (float, optional): Standard deviation of the weight matrix. Defaults to 1e-4.
        device (torch.device): Device for the parameters.
        dtype (torch.dtype): Data type for the parameters.
    
    Returns:
        Dict[str, torch.Tensor]: Parameters of the model.
    """
    if isinstance(frequencies_visibles, np.ndarray):
        frequencies_visibles = torch.from_numpy(frequencies_visibles).to(device=device, dtype=dtype)
    if isinstance(frequencies_labels, np.ndarray):
        frequencies_labels = torch.from_numpy(frequencies_labels).to(device=device, dtype=dtype)
        
    weight_matrix = torch.randn(size=(num_visibles, num_states, num_hiddens), device=device, dtype=dtype) * std_init
    label_matrix = torch.randn(size=(num_labels, num_hiddens), device=device, dtype=dtype) * std_init
    hbias = torch.zeros(num_hiddens, device=device, dtype=dtype)
    
    if frequencies_visibles is None:
        vbias = torch.zeros((num_visibles, num_states), device=device, dtype=dtype)
    else:
        vbias = (
            torch.log(frequencies_visibles) - 1.0 / num_states * torch.sum(torch.log(frequencies_visibles), 0)
        ).to(device=device, dtype=dtype)
    
    if frequencies_labels is None:
        lbias = torch.zeros(num_labels, device=device, dtype=dtype)
    else:
        lbias = (torch.log(frequencies_labels) - torch.log(1.0 - frequencies_labels)).to(device)
    
    return {"vbias": vbias, "hbias": hbias, "lbias": lbias, "weight_matrix": weight_matrix, "label_matrix": label_matrix}


def _init_chains(
    num_samples: int,
    num_states: int,
    params: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Initialize a Markov chain for the RBM by sampling a uniform distribution on the visible layer and the labels
    and sampling the hidden layer according to the visible one.

    Args:
        num_samples (int): Number of parallel chains.
        num_states (int): Number of states of the categorical variables.
        params (Dict[str, torch.Tensor]): Parameters of the model.

    Returns:
        Dict[str, torch.Tensor]: Initial Markov chain.
    """
    num_visibles = len(params["vbias"])
    num_labels = len(params["lbias"])
    mv = torch.ones(
        size=(num_samples, num_visibles, num_states),
        device=params["vbias"].device,
        dtype=params["vbias"].dtype) / 2
    ml = torch.ones(
        size=(num_samples, num_labels),
        device=params["lbias"].device,
        dtype=params["lbias"].dtype) / 2
    visible = one_hot(
        torch.multinomial(mv.view(-1, num_states), 1).view(-1, num_visibles),
        num_classes=num_states,
    ).to(params["weight_matrix"].dtype)
    label = torch.bernoulli(ml)
    visible_flat = visible.view(num_samples, -1)
    weight_matrix_flat = params["weight_matrix"].view(num_visibles * num_states, -1)
    
    mh = torch.sigmoid((params["hbias"] + (visible_flat @ weight_matrix_flat) + (label @ params["label_matrix"])))
    hidden = torch.bernoulli(mh)
    
    return {"visible": visible, "hidden": hidden, "label": label}
