from typing import Dict
import torch


def _compute_energy(
    visible: torch.Tensor,
    hidden: torch.Tensor,
    label: torch.Tensor,
    params: Dict[str, torch.Tensor],
    **kwargs,
) -> torch.Tensor:
    """Computes the energy of the model on the given configuration.

    Args:
        visible (torch.Tensor): Visible units.
        hidden (torch.Tensor): Hidden units.
        label (torch.Tensor): Labels.
        params (Dict[str, torch.Tensor]): Parameters of the model.

    Returns:
        torch.Tensor: Energy of the model on the given configuration.
    """
    def _compute_energy_chain(
        visible: torch.Tensor,
        hidden: torch.Tensor,
        label: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Computes the energy of a single chain.
        """
        fields = (visible @ params["vbias"]) + (hidden @ params["hbias"]) + (label @ params["lbias"])
        interaction = (visible @ params["weight_matrix"] @ hidden) + (label @ params["label_matrix"] @ hidden)
        return - fields - interaction
    
    return torch.vmap(_compute_energy_chain, in_dims=(0, 0, 0, None))(visible, hidden, label, params)


def _compute_energy_visibles(
    visible: torch.Tensor,
    label: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Returns the energy of the model computed on the input visibles and labels.

    Args:
        visible (torch.Tensor): Visible units.
        label (torch.Tensor): Labels.
        params (Dict[str, torch.Tensor]): Parameters of the RBM.

    Returns:
        torch.Tensor: Energy of the data points.
    """
    @torch.jit.script
    def _compute_energy_chain(
        visible: torch.Tensor,
        label: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Computes the energy of a single chain.
        """
        fields = (visible @ params["vbias"]) + (label @ params["lbias"])
        exponent = params["hbias"] + (visible @ params["weight_matrix"]) + (label @ params["label_matrix"])
        log_term = torch.where(exponent < 10, torch.log(1.0 + torch.exp(exponent)), exponent)
        return - fields - log_term.sum()
    
    return torch.vmap(_compute_energy_chain, in_dims=(0, 0, None))(visible, label, params)


def _compute_energy_hiddens(
    hidden: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Computes the energy of the model on the hidden layer.

    Args:
        hidden (torch.Tensor): Hidden units.
        params (Dict[str, torch.Tensor]): Parameters of the model.

    Returns:
        torch.Tensor: Energy of the model on the hidden layer.
    """
    @torch.jit.script
    def _compute_energy_chain(
        hidden: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Computes the energy of a single chain.
        """
        # Effective parameters as if the labels were visible units
        eff_bias = torch.hstack([params["vbias"], params["lbias"]])
        eff_weight_matrix = torch.vstack([params["weight_matrix"], params["label_matrix"]])
        field = hidden @ params["hbias"]
        exponent = eff_bias + (hidden @ eff_weight_matrix.T)
        log_term = torch.where(exponent < 10, torch.log(1.0 + torch.exp(exponent)), exponent)
        return -field - log_term.sum()
    
    return torch.vmap(_compute_energy_chain, in_dims=(0, None))(hidden, params)