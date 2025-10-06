import torch
from typing import Optional

def mm_left(
    W: torch.Tensor,
    x: torch.Tensor
) -> torch.Tensor:
    """Generic matrix multiplication operation W @ x between weight tensor W and input tensor x.

    Args:
        W (torch.Tensor): Weight tensor of size (l, h) or (l, q, h).
        x (torch.Tensor): Input tensor of size (l,) or (l, q).
        
    Returns:
        torch.Tensor: Output tensor after matrix multiplication: W @ x.
    """
    if len(W.shape) == 2:
        return x @ W
    elif len(W.shape) == 3:
        return torch.einsum("lq,lqh->h", x, W)
    else:
        raise ValueError("Invalid weight tensor shape.")


def batched_mm_left(
    W: torch.Tensor,
    x: torch.Tensor
) -> torch.Tensor:
    """Generic batched matrix multiplication operation W @ x between weight tensor W and input tensor x.

    Args:
        W (torch.Tensor): Weight tensor of size (l, h) or (l, q, h).
        x (torch.Tensor): Input tensor of size (batch_size, l) or (batch_size, l, q).
        
    Returns:
        torch.Tensor: Output tensor after matrix multiplication: W @ x.
    """
    if len(W.shape) == 2:
        return x @ W
    elif len(W.shape) == 3:
        return torch.einsum("nlq,lqh->nh", x, W)
    else:
        raise ValueError("Invalid weight tensor shape.")
    
    
def mm_right(
    W: torch.Tensor,
    x: torch.Tensor
) -> torch.Tensor:
    """Generic matrix multiplication operation x @ W between input tensor x and weight tensor W.

    Args:
        W (torch.Tensor): Weight tensor of size (l, h) or (l, q, h).
        x (torch.Tensor): Input tensor of size (h,).

    Returns:
        torch.Tensor: Output tensor after matrix multiplication: W @ x.
    """
    return W @ x


def batched_mm_right(
    W: torch.Tensor,
    x: torch.Tensor
) -> torch.Tensor:
    """Generic batched matrix multiplication operation x @ W between input tensor x and weight tensor W.

    Args:
        W (torch.Tensor): Weight tensor of size (l, h) or (l, q, h).
        x (torch.Tensor): Input tensor of size (batch_size, h).

    Returns:
        torch.Tensor: Output tensor after matrix multiplication: W @ x.
    """
    if len(W.shape) == 2:
        return x @ W.t()
    elif len(W.shape) == 3:
        return torch.einsum("nh,lqh->nlq", x, W)
    else:
        raise ValueError("Invalid weight tensor shape.")
    
    
def outer(
    v: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """Compute the outer product of vector v and matrix h.

    Args:
        v (torch.Tensor): Input vector of size (l,) or (l, q).
        h (torch.Tensor): Input matrix of size (h,).

    Returns:
        torch.Tensor: Output tensor of size (l, h) or (l, q, h).
    """
    if len(v.shape) == 1:
        return torch.einsum("l,h->lh", v, h)
    elif len(v.shape) == 2:
        return torch.einsum("lq,h->lqh", v, h)
    else:
        raise ValueError("Invalid input tensor shape.")
    

def batched_outer(
    v: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """Compute the outer product of vector v and matrix h.

    Args:
        v (torch.Tensor): Input vector of size (batch_size, l) or (batch_size, l, q).
        h (torch.Tensor): Input matrix of size (batch_size, h).

    Returns:
        torch.Tensor: Output tensor of size (l, h) or (l, q, h).
    """
    if len(v.shape) == 2:
        return torch.einsum("nl,nh->lh", v, h)
    elif len(v.shape) == 3:
        return torch.einsum("nlq,nh->lqh", v, h)
    else:
        raise ValueError("Invalid input tensor shape.")
    
    
def multiply(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """Generic element-wise multiplication operation between input tensors x and y.

    Args:
        x (torch.Tensor): First input tensor of shape (batch_size, l) or (batch_size, l, q).
        y (torch.Tensor): Second input tensor of shape (batch_size, ...).

    Returns:
        torch.Tensor: Output tensor after element-wise multiplication.
    """
    if len(x.shape) == 2:
        return x * y.view(y.shape[0], 1)
    elif len(x.shape) == 3:
        return x * y.view(y.shape[0], 1, 1)
    else:
        raise ValueError("Invalid input tensor shape.")