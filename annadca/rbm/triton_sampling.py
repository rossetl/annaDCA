"""Optional fused sampling epilogues; matrix products remain in PyTorch/cuBLAS.

Float32 CUDA only. Random uniforms use PyTorch's device RNG, so manual_seed and
saved RNG states work; draws are distributionally, not bitwise, equivalent to
multinomial/bernoulli. No backward is provided for this sampling backend.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _bernoulli(
    X, B, U, S, P, N: tl.constexpr, C: tl.constexpr, BETA: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0) + tl.load(B + i % C)
    p = tl.sigmoid(BETA * x)
    u = tl.load(U + i, i < N, 0)
    tl.store(P + i, p, i < N)
    tl.store(S + i, (u < p).to(tl.float32), i < N)


@triton.jit
def _categorical(
    X,
    B,
    U,
    S,
    P,
    N: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
    BETA: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.arange(0, BLOCK)
    i = row[:, None] * C + c[None, :]
    mask = (row[:, None] < N) & (c[None, :] < C)
    x = tl.load(X + i, mask, 0) + tl.load(B + i % BC, mask, 0)
    x = tl.where(c[None, :] < C, BETA * x, -float("inf"))
    p = tl.exp(x - tl.max(x, 1)[:, None])
    p = p / tl.sum(p, 1)[:, None]
    cdf = tl.cumsum(p, 1)
    u = tl.load(U + row, row < N, 0)
    selected = tl.minimum(tl.sum((cdf <= u[:, None]).to(tl.int32), 1), C - 1)
    tl.store(P + i, p, mask)
    tl.store(S + i, (c[None, :] == selected[:, None]).to(tl.float32), mask)


def draw(logits, bias, beta=1.0, categorical=False):
    if logits.device.type != "cuda" or logits.dtype != torch.float32:
        raise ValueError("Triton sampling requires float32 CUDA tensors")
    if bias.device != logits.device or bias.dtype != logits.dtype:
        raise ValueError("Bias and logits must have matching device and dtype")
    logits, bias = logits.contiguous(), bias.contiguous()
    samples, probs = torch.empty_like(logits), torch.empty_like(logits)
    if logits.numel() == 0:
        return samples, probs
    with torch.cuda.device(logits.device):
        if categorical:
            c = logits.shape[-1]
            if c > 4096:
                raise ValueError("Triton categorical sampling supports at most 4096 states")
            rows = logits.numel() // c
            uniforms = torch.rand((rows,), device=logits.device)
            block = triton.next_power_of_2(c)
            group = min(16, max(1, 1024 // block))
            _categorical[(triton.cdiv(rows, group),)](
                logits, bias, uniforms, samples, probs, rows, c, bias.numel(), beta, block, group
            )
        else:
            uniforms = torch.rand_like(logits)
            _bernoulli[(triton.cdiv(logits.numel(), 256),)](
                logits, bias, uniforms, samples, probs, logits.numel(), bias.numel(), beta, 256
            )
    return samples, probs
