"""
Triton kernels for efficient sampling during inference using autoregressive-nanoTabPFN.

Key optimization involve sharing the context K/V across batch dimension, reducing memory bandwidth during inference.
Additionally, self-attention does not get used to reestimate values in the KV cache
"""

"""
I don't really want any **fallback** in my custom custom kernel... i either want or i don't want :)
"""

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"GROUP_B": 16, "BLOCK_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"GROUP_B": 16, "BLOCK_N": 128}, num_warps=4, num_stages=3),
            triton.Config({"GROUP_B": 32, "BLOCK_N": 64}, num_warps=8, num_stages=2),
            triton.Config({"GROUP_B": 32, "BLOCK_N": 128}, num_warps=8, num_stages=3),
            triton.Config({"GROUP_B": 32, "BLOCK_N": 256}, num_warps=8, num_stages=3),
        ],
        key=["B", "H", "Lq", "D", "Nctx"],
    )
    @triton.jit
    def cross_attention_shared_ctx_kernel(
        Q_ptr,  # [B, H, Lq, D] queries
        Kc_ptr,  # [H, Nctx, D] context keys (shared across batch)
        Vc_ptr,  # [H, Nctx, D] context values (shared)
        Out_ptr,  # [B, H, Lq, D] output
        B: tl.constexpr,
        H: tl.constexpr,
        Lq: tl.constexpr,
        D: tl.constexpr,
        Nctx: tl.constexpr,
        stride_q_b,
        stride_q_h,
        stride_q_l,
        stride_q_d,
        stride_kc_h,
        stride_kc_n,
        stride_kc_d,
        stride_vc_h,
        stride_vc_n,
        stride_vc_d,
        stride_o_b,
        stride_o_h,
        stride_o_l,
        stride_o_d,
        scale,
        GROUP_B: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Cross-attention where context K/V is shared across batch.
        Uses online softmax for numerical stability.
        """
        pid_b = tl.program_id(0)  # batch group
        pid_h = tl.program_id(1)  # head
        pid_l = tl.program_id(2)  # query position

        offs_b = pid_b * GROUP_B + tl.arange(0, GROUP_B)
        mask_b = offs_b < B

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        # Load Q [GROUP_B, D]
        q_ptrs = (
            Q_ptr
            + offs_b[:, None] * stride_q_b
            + pid_h * stride_q_h
            + pid_l * stride_q_l
            + offs_d[None, :] * stride_q_d
        )
        mask_q = mask_b[:, None] & mask_d[None, :]
        q = tl.load(q_ptrs, mask=mask_q, other=0.0).to(tl.float32)

        # Online softmax accumulators
        m = tl.full((GROUP_B,), -float("inf"), tl.float32)
        l = tl.zeros((GROUP_B,), tl.float32)
        acc = tl.zeros((GROUP_B, BLOCK_D), tl.float32)

        # Iterate over context tiles
        for n0 in range(0, Nctx, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < Nctx

            # Load K, V tiles from shared context
            kc_ptrs = (
                Kc_ptr
                + pid_h * stride_kc_h
                + offs_n[:, None] * stride_kc_n
                + offs_d[None, :] * stride_kc_d
            )
            vc_ptrs = (
                Vc_ptr
                + pid_h * stride_vc_h
                + offs_n[:, None] * stride_vc_n
                + offs_d[None, :] * stride_vc_d
            )

            k_tile = tl.load(
                kc_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0, cache_modifier=".cg"
            ).to(tl.float32)
            v_tile = tl.load(
                vc_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0, cache_modifier=".cg"
            ).to(tl.float32)

            # Compute attention scores
            s = tl.dot(q, tl.trans(k_tile)) * scale  # [GROUP_B, BLOCK_N]
            s = tl.where(mask_n[None, :], s, -float("inf"))

            # Online softmax update
            smax = tl.max(s, 1)
            m_new = tl.maximum(m, smax)
            alpha = tl.exp(m - m_new)
            s_exp = tl.exp(s - m_new[:, None])

            l = l * alpha + tl.sum(s_exp, 1)
            acc = acc * alpha[:, None] + tl.dot(s_exp, v_tile)
            m = m_new

        # Normalize and store
        out = acc / l[:, None]
        out_ptrs = (
            Out_ptr
            + offs_b[:, None] * stride_o_b
            + pid_h * stride_o_h
            + pid_l * stride_o_l
            + offs_d[None, :] * stride_o_d
        )
        tl.store(out_ptrs, out.to(out_ptrs.dtype.element_ty), mask=mask_q)


def triton_cross_attention(q: Tensor, k_ctx: Tensor, v_ctx: Tensor) -> Tensor:
    """
    Cross-attention with shared context K/V.

    Args:
        q: [B, H, Lq, D] queries
        k_ctx: [H, Nctx, D] context keys (shared across batch)
        v_ctx: [H, Nctx, D] context values

    Returns:
        [B, H, Lq, D] attention output
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available")

    B, H, Lq, D = q.shape
    Nctx = k_ctx.shape[1]

    out = torch.empty_like(q)
    scale = D**-0.5

    def grid(META):
        return (triton.cdiv(B, META["GROUP_B"]), H, Lq)

    cross_attention_shared_ctx_kernel[grid](
        q,
        k_ctx,
        v_ctx,
        out,
        B,
        H,
        Lq,
        D,
        Nctx,
        *q.stride(),
        *k_ctx.stride(),
        *v_ctx.stride(),
        *out.stride(),
        scale,
        BLOCK_D=D,
    )
    return out


def triton_available() -> bool:
    """Check if Triton is available."""
    return HAS_TRITON


# Fallback using PyTorch for non-Triton environments
def pytorch_cross_attention(q: Tensor, k_ctx: Tensor, v_ctx: Tensor) -> Tensor:
    """
    PyTorch fallback for cross-attention with shared context.

    Args:
        q: [B, H, Lq, D] queries
        k_ctx: [H, Nctx, D] context keys
        v_ctx: [H, Nctx, D] context values

    Returns:
        [B, H, Lq, D] attention output
    """
    B, H, Lq, D = q.shape

    # Expand context to batch dimension
    k = k_ctx.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, Nctx, D]
    v = v_ctx.unsqueeze(0).expand(B, -1, -1, -1)

    # Standard scaled dot-product attention
    scale = D**-0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


def cross_attention(q: Tensor, k_ctx: Tensor, v_ctx: Tensor, use_triton: bool = True) -> Tensor:
    """
    Cross-attention dispatcher.

    Uses Triton kernel when available and requested, otherwise falls back to PyTorch.
    """
    if use_triton and HAS_TRITON and q.is_cuda:
        return triton_cross_attention(q, k_ctx, v_ctx)
    return pytorch_cross_attention(q, k_ctx, v_ctx)
