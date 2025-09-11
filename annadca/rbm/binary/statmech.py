from typing import Dict
import torch


def _compute_energy(
    visible: torch.Tensor,
    hidden: torch.Tensor,
    label: torch.Tensor,
    params: Dict[str, torch.Tensor],
    **kwargs,
) -> torch.Tensor:
    def _compute_energy_chain(
        visible: torch.Tensor,
        hidden: torch.Tensor,
        label: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        fields = (visible @ params["vbias"]) + (hidden @ params["hbias"]) + (label @ params["lbias"])
        interaction = (visible @ params["weight_matrix"] @ hidden) + (label @ params["label_matrix"] @ hidden)
        return - fields - interaction
    
    return torch.vmap(_compute_energy_chain, in_dims=(0, 0, 0, None))(visible, hidden, label, params)


def _compute_energy_visibles_labels(
    visible: torch.Tensor,
    label: torch.Tensor,
    params: Dict[str, torch.Tensor],
    **kwargs,
) -> torch.Tensor:
    @torch.jit.script
    def _compute_energy_chain(
        visible: torch.Tensor,
        label: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        fields = (visible @ params["vbias"]) + (label @ params["lbias"])
        exponent = params["hbias"] + (visible @ params["weight_matrix"]) + (label @ params["label_matrix"])
        log_term = torch.where(exponent < 10, torch.log(1.0 + torch.exp(exponent)), exponent)
        return - fields - log_term.sum()
    
    return torch.vmap(_compute_energy_chain, in_dims=(0, 0, None))(visible, label, params)


def _compute_energy_visibles(
    visible: torch.Tensor,
    params: Dict[str, torch.Tensor],
    **kwargs,
) -> torch.Tensor:

    def _compute_energy_chain(
        visible: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        fields = visible @ params["vbias"]
        exponent = params["hbias"].unsqueeze(0) + \
            + (visible @ params["weight_matrix"]).unsqueeze(0) + params["label_matrix"] # (num_classes, num_hiddens)
        log_term = torch.where(exponent < 10, torch.log(1.0 + torch.exp(exponent)), exponent)
        exponent_label_term = log_term.sum(1) + params["lbias"] # (num_classes,)
        return - fields - torch.logsumexp(exponent_label_term, dim=0)

    return torch.vmap(_compute_energy_chain, in_dims=(0, None))(visible, params)


def _compute_energy_hiddens(
    hidden: torch.Tensor,
    params: Dict[str, torch.Tensor],
    **kwargs,
) -> torch.Tensor:

    @torch.jit.script
    def _compute_energy_chain(
        hidden: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # Effective parameters as if the labels were visible units
        eff_bias = torch.hstack([params["vbias"], params["lbias"]])
        eff_weight_matrix = torch.vstack([params["weight_matrix"], params["label_matrix"]])
        field = hidden @ params["hbias"]
        exponent = eff_bias + (hidden @ eff_weight_matrix.T)
        log_term = torch.where(exponent < 10, torch.log(1.0 + torch.exp(exponent)), exponent)
        return - field - log_term.sum()
    
    return torch.vmap(_compute_energy_chain, in_dims=(0, None))(hidden, params)


def _update_weights_AIS(
    prev_params: Dict[str, torch.Tensor],
    curr_params: Dict[str, torch.Tensor],
    chains: Dict[str, torch.Tensor],
    log_weights: torch.Tensor,
) -> torch.Tensor:

    energy_prev = _compute_energy(**chains, params=prev_params)
    energy_curr = _compute_energy(**chains, params=curr_params)
    log_weights += energy_prev - energy_curr
    
    return log_weights


def _compute_log_likelihood(
    visible: torch.Tensor,
    weight: torch.Tensor,
    params: Dict[str, torch.Tensor],
    logZ: float,
) -> float:
    
    energy_data = _compute_energy_visibles(
        visible=visible,
        params=params,
    )
    mean_energy_data = (energy_data * weight.view(-1)).sum() / weight.sum()

    return - mean_energy_data.item() - logZ