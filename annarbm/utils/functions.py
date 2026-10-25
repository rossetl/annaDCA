"""Products between the weight matrix W of shape (*visible_shape, num_hiddens) and the units of the layers.
The visible dimensions are handled with an ellipsis, so the same functions work for (L, h) and (L, q, h) weights."""
import torch


def mm_left(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """x^T W for a single visible configuration x of shape visible_shape. Returns a tensor of shape (h,)."""
    return torch.einsum("...,...h->h", x, W)


def batched_mm_left(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """x^T W for a batch of visible configurations x of shape (n, *visible_shape). Returns a tensor of shape (n, h)."""
    return torch.einsum("n...,...h->nh", x, W)


def batched_mm_right(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """W x for a batch of hidden configurations x of shape (n, h). Returns a tensor of shape (n, *visible_shape)."""
    return torch.einsum("nh,...h->n...", x, W)


def outer(v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Outer product of v of shape visible_shape and h of shape (h,). Returns a tensor of shape (*visible_shape, h)."""
    return torch.einsum("...,h->...h", v, h)


def batched_outer(v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Sum over the batch of the outer products of v (n, *visible_shape) and h (n, h). Returns a tensor of shape (*visible_shape, h)."""
    return torch.einsum("n...,nh->...h", v, h)
