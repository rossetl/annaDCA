from typing import Dict

import numpy as np
import torch


def _init_parameters(
    num_visibles: int,
    num_hiddens: int,
    num_labels: int,
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
        
    weight_matrix = torch.randn(size=(num_visibles, num_hiddens), device=device, dtype=dtype) * std_init
    label_matrix = torch.randn(size=(num_labels, num_hiddens), device=device, dtype=dtype) * std_init
    hbias = torch.zeros(num_hiddens, device=device, dtype=dtype)
    
    if frequencies_visibles is None:
        vbias = torch.zeros(num_visibles, device=device, dtype=dtype)
    else:
        vbias = (torch.log(frequencies_visibles) - torch.log(1.0 - frequencies_visibles)).to(device)
    
    if frequencies_labels is None:
        lbias = torch.zeros(num_labels, device=device, dtype=dtype)
    else:
        lbias = (torch.log(frequencies_labels) - torch.log(1.0 - frequencies_labels)).to(device)
    
    return {"vbias": vbias, "hbias": hbias, "lbias": lbias, "weight_matrix": weight_matrix, "label_matrix": label_matrix}


def _init_chains(
    num_samples: int,
    params: Dict[str, torch.Tensor],
    use_profile: bool = False,
) -> Dict[str, torch.Tensor]:
    """Initialize the Markov chains for the RBM by sampling a uniform distribution on the visible layer and the labels
    and sampling the hidden layer according to the visible one. If use_profile is True, the visible units and the label
    are sampled from the profile model using the local fields.

    Args:
        num_samples (int): Number of parallel chains.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        use_profile (bool, optional): Whether to use the profile model. Defaults to False.

    Returns:
        Dict[str, torch.Tensor]: Initial Markov chain.
    """
    num_visibles = len(params["vbias"])
    num_labels = len(params["lbias"])
    
    if use_profile:
        mv = torch.sigmoid(params["vbias"]).repeat(num_samples, 1)
        ml = torch.sigmoid(params["lbias"]).repeat(num_samples, 1)
        
    else:
        mv = torch.ones(
            size=(num_samples, num_visibles),
            device=params["vbias"].device,
            dtype=params["vbias"].dtype) / 2
        ml = torch.ones(
            size=(num_samples, num_labels),
            device=params["lbias"].device,
            dtype=params["lbias"].dtype) / 2
        
    visible = torch.bernoulli(mv)
    label = torch.bernoulli(ml)
    
    mh = torch.sigmoid((params["hbias"] + (visible @ params["weight_matrix"]) + (label @ params["label_matrix"])))
    hidden = torch.bernoulli(mh)
    
    return {"visible": visible, "hidden": hidden, "label": label}
