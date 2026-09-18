"""Exact positive-phase expectations for sequences with missing categorical labels."""

import torch
from torch.nn.functional import softplus


@torch.no_grad()
def marginalize_missing_labels(
    data: dict[str, torch.Tensor], params: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Return batch statistics at beta=1 without modifying the observed annotations.

    All-zero label rows denote missing annotations. The returned correction is
    the weighted conditional label-hidden covariance, needed in both centered
    and uncentered gradients. The negative phase is unaffected.
    """
    result = dict(data)
    label = data["label"].clone()
    visible = data["visible"].reshape(len(label), -1)
    weight_matrix = params["weight_matrix"].reshape(-1, params["hbias"].numel())
    field = visible @ weight_matrix + params["hbias"]
    hidden = torch.sigmoid(field + label @ params["label_matrix"])
    missing = (label == 0).all(dim=-1)
    correction = torch.zeros_like(params["label_matrix"])
    if missing.any():
        # Only missing rows need the extra label axis.
        conditional_field = field[missing, None, :] + params["label_matrix"][None, :, :]
        posterior = torch.softmax(params["lbias"] + softplus(conditional_field).sum(dim=-1), dim=-1)
        conditional_hidden = torch.sigmoid(conditional_field)
        mean_hidden = (posterior[:, :, None] * conditional_hidden).sum(dim=1)
        weights = data["weight"].reshape(-1)
        weights = weights[missing] / weights.sum()
        correction = torch.einsum(
            "n,nk,nkh->kh",
            weights,
            posterior,
            conditional_hidden - mean_hidden[:, None, :],
        )
        label[missing] = posterior
        hidden[missing] = mean_hidden
    result["label"] = label
    result["hidden"] = hidden
    result["label_hidden_correction"] = correction
    return result
