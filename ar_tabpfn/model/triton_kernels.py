"""
Triton kernels for efficient inference in autoregressive-nanoTabPFN.

Key optimization: Context K/V is shared across batch dimension, reducing memory
bandwidth during inference. The kernel returns LogSumExp (LSE) statistics to
enable mathematical merging with buffer attention.

Architecture:
- Context K/V: [H, Nctx, D] - shared across all batch elements
- Query: [B, H, Lq, D] - batched queries
- Output: [B, H, Lq, D] + LSE stats for merging with buffer attention
"""

import torch
from torch import Tensor
from typing import Tuple, Optional

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE_B": 32, "BLOCK_SIZE_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE_B": 64, "BLOCK_SIZE_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE_B": 32, "BLOCK_SIZE_N": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE_B": 64, "BLOCK_SIZE_N": 128}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE_B": 128, "BLOCK_SIZE_N": 64}, num_warps=8, num_stages=2),
        ],
        key=["B", "H", "N_CTX", "HEAD_DIM"],
    )
    @triton.jit
    def _shared_context_attn_kernel(
        # Pointers
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        LSE_ptr,  # LogSumExp output for merging
        # Shapes
        B,
        H,
        N_CTX,
        HEAD_DIM: tl.constexpr,
        # Strides for Q [B, H, D] (squeezed Lq=1)
        stride_q_b,
        stride_q_h,
        stride_q_d,
        # Strides for K [H, N, D]
        stride_k_h,
        stride_k_n,
        stride_k_d,
        # Strides for V [H, N, D]
        stride_v_h,
        stride_v_n,
        stride_v_d,
        # Strides for Out [B, H, D]
        stride_o_b,
        stride_o_h,
        stride_o_d,
        # Meta-parameters
        sm_scale,
        BLOCK_SIZE_B: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        """
        Shared-context cross-attention kernel with LSE output.

        Computes: softmax(Q @ K_ctx.T / sqrt(d)) @ V_ctx
        Returns both the attention output and LSE statistics.

        Key optimization: K_ctx and V_ctx are loaded once per block and
        reused across BLOCK_SIZE_B queries, reducing memory bandwidth.
        """
        # 1. Program IDs
        pid_b = tl.program_id(0)  # Batch tile ID
        pid_h = tl.program_id(1)  # Head ID

        # 2. Block Pointers using modern tl.make_block_ptr

        # Q_block: [BLOCK_SIZE_B, HEAD_DIM] - tile of queries from different batch items
        Q_block_ptr = tl.make_block_ptr(
            base=Q_ptr + pid_h * stride_q_h,
            shape=(B, HEAD_DIM),
            strides=(stride_q_b, stride_q_d),
            offsets=(pid_b * BLOCK_SIZE_B, 0),
            block_shape=(BLOCK_SIZE_B, HEAD_DIM),
            order=(1, 0),
        )

        # K_block: [HEAD_DIM, BLOCK_SIZE_N] - transposed for dot product
        # Note: These do NOT depend on batch (shared context)
        K_block_ptr = tl.make_block_ptr(
            base=K_ptr + pid_h * stride_k_h,
            shape=(HEAD_DIM, N_CTX),
            strides=(stride_k_d, stride_k_n),
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_SIZE_N),
            order=(0, 1),
        )

        # V_block: [BLOCK_SIZE_N, HEAD_DIM]
        V_block_ptr = tl.make_block_ptr(
            base=V_ptr + pid_h * stride_v_h,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_v_n, stride_v_d),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_N, HEAD_DIM),
            order=(1, 0),
        )

        # 3. Load Query Tile (with boundary check for non-divisible B)
        q = tl.load(Q_block_ptr, boundary_check=(0, 1))
        q = (q * sm_scale).to(K_ptr.dtype.element_ty)

        # 4. Online Softmax Accumulators
        m_i = tl.zeros([BLOCK_SIZE_B], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_SIZE_B], dtype=tl.float32)
        acc = tl.zeros([BLOCK_SIZE_B, HEAD_DIM], dtype=tl.float32)

        # 5. Main Loop: Iterate over Shared Context
        # This is the key optimization - K/V tiles are loaded once and
        # reused for ALL BLOCK_SIZE_B queries
        for start_n in range(0, N_CTX, BLOCK_SIZE_N):
            start_n = tl.multiple_of(start_n, BLOCK_SIZE_N)

            # Load shared context K/V tiles (reused across batch)
            k = tl.load(K_block_ptr, boundary_check=(0, 1))
            v = tl.load(V_block_ptr, boundary_check=(0, 1))

            # Compute attention scores: Q[Batch, Dim] @ K[Dim, N] -> S[Batch, N]
            qk = tl.dot(q, k)

            # Mask out-of-bounds context positions (zero-padded K gives 0, not -inf)
            # This is critical for correctness when N_CTX is not divisible by BLOCK_SIZE_N
            offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
            qk = tl.where(offs_n[None, :] < N_CTX, qk, float("-inf"))

            # Online Softmax Update
            m_i_new = tl.max(qk, 1)
            m_i_new = tl.maximum(m_i, m_i_new)

            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new[:, None])

            # Update accumulators
            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new

            # Advance context pointers to next tile
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_N))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_N, 0))

        # 6. Normalize output
        acc = acc / l_i[:, None]

        # 7. Compute LSE for merging: lse = m + log(l)
        lse = m_i + tl.log(l_i)

        # 8. Store Output
        Out_block_ptr = tl.make_block_ptr(
            base=Out_ptr + pid_h * stride_o_h,
            shape=(B, HEAD_DIM),
            strides=(stride_o_b, stride_o_d),
            offsets=(pid_b * BLOCK_SIZE_B, 0),
            block_shape=(BLOCK_SIZE_B, HEAD_DIM),
            order=(1, 0),
        )
        tl.store(Out_block_ptr, acc.to(Out_ptr.dtype.element_ty), boundary_check=(0, 1))

        # 9. Store LSE (shape: [B, H], row-major means stride is H for batch dim)
        off_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        LSE_ptr_base = LSE_ptr + off_b * H + pid_h  # [B, H] layout: b * H + h
        tl.store(LSE_ptr_base, lse, mask=off_b < B)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE_BQ": 32, "BLOCK_SIZE_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE_BQ": 64, "BLOCK_SIZE_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE_BQ": 32, "BLOCK_SIZE_N": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE_BQ": 64, "BLOCK_SIZE_N": 128}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE_BQ": 128, "BLOCK_SIZE_N": 64}, num_warps=8, num_stages=2),
        ],
        key=["B", "H", "LQ", "N_CTX", "HEAD_DIM"],
    )
    @triton.jit
    def _shared_context_attn_kernel_multiq(
        # Pointers
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        LSE_ptr,  # LogSumExp output for merging
        # Shapes
        B,
        H,
        LQ,
        N_CTX,
        HEAD_DIM: tl.constexpr,
        # Strides for Q [B, H, Lq, D]
        stride_q_b,
        stride_q_h,
        stride_q_lq,
        stride_q_d,
        # Strides for K [H, N, D]
        stride_k_h,
        stride_k_n,
        stride_k_d,
        # Strides for V [H, N, D]
        stride_v_h,
        stride_v_n,
        stride_v_d,
        # Strides for Out [B, H, Lq, D]
        stride_o_b,
        stride_o_h,
        stride_o_lq,
        stride_o_d,
        # Strides for LSE [B, H, Lq]
        stride_lse_b,
        stride_lse_h,
        stride_lse_lq,
        # Meta-parameters
        sm_scale,
        BLOCK_SIZE_BQ: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        """
        Shared-context cross-attention kernel with LSE output for Lq > 1.

        Computes: softmax(Q @ K_ctx.T / sqrt(d)) @ V_ctx
        Returns both the attention output and LSE statistics.

        The kernel parallelizes over the flattened (batch, query) dimension.
        """
        pid_bq = tl.program_id(0)  # Batch-query tile ID
        pid_h = tl.program_id(1)  # Head ID

        offs_bq = pid_bq * BLOCK_SIZE_BQ + tl.arange(0, BLOCK_SIZE_BQ)
        offs_d = tl.arange(0, HEAD_DIM)

        # Map flattened index -> (batch, query)
        b_idx = offs_bq // LQ
        q_idx = offs_bq - b_idx * LQ
        valid_bq = b_idx < B

        # Load Q tile: [BLOCK_SIZE_BQ, HEAD_DIM]
        q_ptrs = (
            Q_ptr
            + b_idx[:, None] * stride_q_b
            + pid_h * stride_q_h
            + q_idx[:, None] * stride_q_lq
            + offs_d[None, :] * stride_q_d
        )
        q = tl.load(
            q_ptrs,
            mask=valid_bq[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        q = (q * sm_scale).to(K_ptr.dtype.element_ty)

        # K/V block pointers (shared context)
        K_block_ptr = tl.make_block_ptr(
            base=K_ptr + pid_h * stride_k_h,
            shape=(HEAD_DIM, N_CTX),
            strides=(stride_k_d, stride_k_n),
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_SIZE_N),
            order=(0, 1),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V_ptr + pid_h * stride_v_h,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_v_n, stride_v_d),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_N, HEAD_DIM),
            order=(1, 0),
        )

        # Online softmax accumulators
        m_i = tl.zeros([BLOCK_SIZE_BQ], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_SIZE_BQ], dtype=tl.float32)
        acc = tl.zeros([BLOCK_SIZE_BQ, HEAD_DIM], dtype=tl.float32)

        for start_n in range(0, N_CTX, BLOCK_SIZE_N):
            start_n = tl.multiple_of(start_n, BLOCK_SIZE_N)

            k = tl.load(K_block_ptr, boundary_check=(0, 1))
            v = tl.load(V_block_ptr, boundary_check=(0, 1))

            qk = tl.dot(q, k)

            offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
            qk = tl.where(offs_n[None, :] < N_CTX, qk, float("-inf"))

            m_i_new = tl.max(qk, 1)
            m_i_new = tl.maximum(m_i, m_i_new)

            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new[:, None])

            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new

            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_N))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_N, 0))

        acc = acc / l_i[:, None]
        lse = m_i + tl.log(l_i)

        # Store output
        out_ptrs = (
            Out_ptr
            + b_idx[:, None] * stride_o_b
            + pid_h * stride_o_h
            + q_idx[:, None] * stride_o_lq
            + offs_d[None, :] * stride_o_d
        )
        tl.store(
            out_ptrs,
            acc.to(Out_ptr.dtype.element_ty),
            mask=valid_bq[:, None] & (offs_d[None, :] < HEAD_DIM),
        )

        # Store LSE
        lse_ptrs = (
            LSE_ptr
            + b_idx * stride_lse_b
            + pid_h * stride_lse_h
            + q_idx * stride_lse_lq
        )
        tl.store(lse_ptrs, lse, mask=valid_bq)


def triton_context_attention(
    q: Tensor, k_ctx: Tensor, v_ctx: Tensor, scale: Optional[float] = None
) -> Tuple[Tensor, Tensor]:
    """
    Shared-context cross-attention using Triton kernel.

    Args:
        q: [B, H, Lq, D] queries
        k_ctx: [H, Nctx, D] context keys (SHARED across batch)
        v_ctx: [H, Nctx, D] context values (SHARED across batch)
        scale: Optional scaling factor (default: D^-0.5)

    Returns:
        out: [B, H, Lq, D] attention output
        lse: [B, H, Lq] LogSumExp statistics for merging with buffer attention
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available")

    B, H, Lq, D = q.shape
    N_CTX = k_ctx.shape[1]

    if scale is None:
        scale = D**-0.5

    if Lq > 1:
        return triton_context_attention_multiq(q, k_ctx, v_ctx, scale=scale)

    # Squeeze Lq dimension for kernel
    q_squeezed = q.squeeze(2)  # [B, H, D]

    # Allocate outputs
    out = torch.empty((B, H, D), device=q.device, dtype=q.dtype)
    lse = torch.empty((B, H), device=q.device, dtype=torch.float32)

    # Launch kernel
    grid = lambda META: (triton.cdiv(B, META["BLOCK_SIZE_B"]), H)

    _shared_context_attn_kernel[grid](
        q_squeezed,
        k_ctx,
        v_ctx,
        out,
        lse,
        B,
        H,
        N_CTX,
        D,
        q_squeezed.stride(0),
        q_squeezed.stride(1),
        q_squeezed.stride(2),
        k_ctx.stride(0),
        k_ctx.stride(1),
        k_ctx.stride(2),
        v_ctx.stride(0),
        v_ctx.stride(1),
        v_ctx.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        scale,
    )

    # Restore Lq dimension
    return out.unsqueeze(2), lse.unsqueeze(2)


def triton_context_attention_multiq(
    q: Tensor, k_ctx: Tensor, v_ctx: Tensor, scale: Optional[float] = None
) -> Tuple[Tensor, Tensor]:
    """
    Shared-context cross-attention using Triton kernel for Lq > 1.

    Args:
        q: [B, H, Lq, D] queries
        k_ctx: [H, Nctx, D] context keys (SHARED across batch)
        v_ctx: [H, Nctx, D] context values (SHARED across batch)
        scale: Optional scaling factor (default: D^-0.5)

    Returns:
        out: [B, H, Lq, D] attention output
        lse: [B, H, Lq] LogSumExp statistics for merging with buffer attention
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available")

    B, H, Lq, D = q.shape
    N_CTX = k_ctx.shape[1]

    if scale is None:
        scale = D**-0.5

    out = torch.empty((B, H, Lq, D), device=q.device, dtype=q.dtype)
    lse = torch.empty((B, H, Lq), device=q.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(B * Lq, META["BLOCK_SIZE_BQ"]), H)

    _shared_context_attn_kernel_multiq[grid](
        q,
        k_ctx,
        v_ctx,
        out,
        lse,
        B,
        H,
        Lq,
        N_CTX,
        D,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_ctx.stride(0),
        k_ctx.stride(1),
        k_ctx.stride(2),
        v_ctx.stride(0),
        v_ctx.stride(1),
        v_ctx.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        scale,
    )

    return out, lse


def merge_attention_outputs(
    out_ctx: Tensor,
    lse_ctx: Tensor,
    out_buf: Tensor,
    lse_buf: Tensor,
) -> Tensor:
    """
    Merge two attention outputs using their LogSumExp statistics.

    This is mathematically equivalent to computing attention over the
    concatenation of context and buffer K/V, but allows us to:
    1. Use the fast Triton kernel for shared context
    2. Use standard attention for per-batch buffer

    The formula is:
        out = (exp(lse_ctx - lse_max) * out_ctx + exp(lse_buf - lse_max) * out_buf)
              / (exp(lse_ctx - lse_max) + exp(lse_buf - lse_max))

    Which simplifies using the LogSumExp trick for numerical stability.

    Args:
        out_ctx: [B, H, Lq, D] context attention output
        lse_ctx: [B, H, Lq] context LogSumExp
        out_buf: [B, H, Lq, D] buffer attention output
        lse_buf: [B, H, Lq] buffer LogSumExp

    Returns:
        merged: [B, H, Lq, D] combined attention output
    """
    # Numerical stability: subtract max before exp
    lse_max = torch.maximum(lse_ctx, lse_buf)

    # Compute rescaling weights
    w_ctx = torch.exp(lse_ctx - lse_max)  # [B, H, Lq]
    w_buf = torch.exp(lse_buf - lse_max)  # [B, H, Lq]

    # Normalize weights
    w_total = w_ctx + w_buf
    w_ctx = w_ctx / w_total
    w_buf = w_buf / w_total

    # Weighted combination
    # Need to expand weights for broadcasting with D dimension
    merged = w_ctx.unsqueeze(-1) * out_ctx + w_buf.unsqueeze(-1) * out_buf

    # Return same dtype as input
    return merged.to(out_ctx.dtype)


def triton_available() -> bool:
    """Check if Triton is available."""
    return HAS_TRITON


def pytorch_context_attention(
    q: Tensor, k_ctx: Tensor, v_ctx: Tensor, scale: Optional[float] = None
) -> Tuple[Tensor, Tensor]:
    """
    PyTorch reference implementation of shared-context attention with LSE.

    Args:
        q: [B, H, Lq, D] queries
        k_ctx: [H, Nctx, D] context keys (shared)
        v_ctx: [H, Nctx, D] context values (shared)

    Returns:
        out: [B, H, Lq, D] attention output
        lse: [B, H, Lq] LogSumExp statistics
    """
    B, H, Lq, D = q.shape
    N_CTX = k_ctx.shape[1]

    if scale is None:
        scale = D**-0.5

    # Expand context to batch dimension (logically, not physically copied)
    k_ctx_exp = k_ctx.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, Nctx, D]
    v_ctx_exp = v_ctx.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, Nctx, D]

    # Compute attention scores
    scores = torch.matmul(q, k_ctx_exp.transpose(-2, -1)) * scale  # [B, H, Lq, Nctx]

    # Compute LSE for merging (in float32 for numerical stability)
    lse = torch.logsumexp(scores.float(), dim=-1)  # [B, H, Lq], float32

    # Softmax and weighted sum
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_ctx_exp)  # [B, H, Lq, D]

    return out, lse


def pytorch_buffer_attention(
    q: Tensor,
    k_buf: Tensor,
    v_buf: Tensor,
    mask: Optional[Tensor] = None,
    scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Standard attention over per-batch buffer K/V with LSE output.

    Args:
        q: [B, H, Lq, D] queries
        k_buf: [B, H, Nbuf, D] buffer keys (per-batch)
        v_buf: [B, H, Nbuf, D] buffer values (per-batch)
        mask: Optional [B, H, Lq, Nbuf] or broadcastable attention mask

    Returns:
        out: [B, H, Lq, D] attention output
        lse: [B, H, Lq] LogSumExp statistics
    """
    B, H, Lq, D = q.shape

    if scale is None:
        scale = D**-0.5

    # Compute attention scores
    scores = torch.matmul(q, k_buf.transpose(-2, -1)) * scale  # [B, H, Lq, Nbuf]

    no_kv = None

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
        no_kv = ~mask.any(dim=-1)

    # Compute LSE for merging (in float32 for numerical stability)
    lse = torch.logsumexp(scores.float(), dim=-1)  # [B, H, Lq], float32

    # Softmax and weighted sum
    attn = torch.softmax(scores, dim=-1)
    if no_kv is not None:
        attn = torch.where(no_kv.unsqueeze(-1), torch.zeros_like(attn), attn)
    out = torch.matmul(attn, v_buf)  # [B, H, Lq, D]

    return out, lse


_tf_buffer_mask_cache = {}


def _teacher_forcing_buffer_mask(num_targets: int, device: torch.device) -> Tensor:
    """Create teacher-forcing buffer mask for [2*Nt] queries attending to [Nt] buffers."""
    key = (num_targets, str(device))
    cached = _tf_buffer_mask_cache.get(key)
    if cached is not None:
        return cached

    q_idx = torch.arange(2 * num_targets, device=device)
    kv_idx = torch.arange(num_targets, device=device)

    is_buffer = q_idx < num_targets
    buffer_mask = kv_idx[None, :] <= q_idx[:, None]
    target_mask = kv_idx[None, :] < (q_idx - num_targets)[:, None]
    mask = torch.where(is_buffer[:, None], buffer_mask, target_mask)

    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 2*Nt, Nt]
    _tf_buffer_mask_cache[key] = mask
    return mask


def pytorch_teacher_forcing_buffer_attention(
    q: Tensor,
    k_buf: Tensor,
    v_buf: Tensor,
    num_targets: int,
    scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Buffer attention for teacher forcing with LSE output.

    Queries are ordered as [Buffer_0..Nt-1, Target_0..Nt-1].
    Buffers attend to [0..i], targets attend to [0..i-1].
    """
    if q.shape[2] != 2 * num_targets:
        raise ValueError("Expected 2*num_targets queries for teacher forcing.")
    if k_buf.shape[2] != num_targets:
        raise ValueError("Expected buffer length to match num_targets.")

    mask = _teacher_forcing_buffer_mask(num_targets, q.device)
    return pytorch_buffer_attention(q, k_buf, v_buf, mask=mask, scale=scale)


def hybrid_teacher_forcing_attention(
    q: Tensor,
    k_ctx: Tensor,
    v_ctx: Tensor,
    k_buf: Tensor,
    v_buf: Tensor,
    num_targets: int,
    use_triton: bool = True,
) -> Tensor:
    """
    Hybrid attention for teacher forcing: Triton context + PyTorch buffer + LSE merge.

    Args:
        q: [B, H, 2*Nt, D] queries (buffers then targets)
        k_ctx: [H, Nctx, D] context keys (SHARED)
        v_ctx: [H, Nctx, D] context values (SHARED)
        k_buf: [B, H, Nt, D] buffer keys (per-batch)
        v_buf: [B, H, Nt, D] buffer values (per-batch)
        num_targets: Nt
        use_triton: Use Triton kernel for context (default True)

    Returns:
        out: [B, H, 2*Nt, D] combined attention output
    """
    if k_buf.shape[2] == 0:
        if use_triton and HAS_TRITON and q.is_cuda:
            out_ctx, _ = triton_context_attention(q, k_ctx, v_ctx)
        else:
            out_ctx, _ = pytorch_context_attention(q, k_ctx, v_ctx)
        return out_ctx

    if use_triton and HAS_TRITON and q.is_cuda:
        out_ctx, lse_ctx = triton_context_attention(q, k_ctx, v_ctx)
    else:
        out_ctx, lse_ctx = pytorch_context_attention(q, k_ctx, v_ctx)

    out_buf, lse_buf = pytorch_teacher_forcing_buffer_attention(
        q, k_buf, v_buf, num_targets=num_targets
    )

    return merge_attention_outputs(out_ctx, lse_ctx, out_buf, lse_buf)


def hybrid_attention(
    q: Tensor,
    k_ctx: Tensor,
    v_ctx: Tensor,
    k_buf: Tensor,
    v_buf: Tensor,
    buf_mask: Optional[Tensor] = None,
    use_triton: bool = True,
) -> Tensor:
    """
    Hybrid attention: Triton kernel for shared context + PyTorch for buffer.

    This is the main entry point for inference. It:
    1. Computes context attention using Triton (fast, shared K/V)
    2. Computes buffer attention using PyTorch (per-batch K/V)
    3. Merges them using LSE statistics

    Args:
        q: [B, H, Lq, D] queries
        k_ctx: [H, Nctx, D] context keys (SHARED)
        v_ctx: [H, Nctx, D] context values (SHARED)
        k_buf: [B, H, Nbuf, D] buffer keys (per-batch)
        v_buf: [B, H, Nbuf, D] buffer values (per-batch)
        buf_mask: Optional mask for buffer attention
        use_triton: Use Triton kernel for context (default True)

    Returns:
        out: [B, H, Lq, D] combined attention output
    """
    # Handle empty buffer case
    if k_buf.shape[2] == 0:
        # No buffer, just return context attention
        if use_triton and HAS_TRITON and q.is_cuda:
            out_ctx, _ = triton_context_attention(q, k_ctx, v_ctx)
        else:
            out_ctx, _ = pytorch_context_attention(q, k_ctx, v_ctx)
        return out_ctx

    # Context attention (Triton or PyTorch)
    if use_triton and HAS_TRITON and q.is_cuda:
        out_ctx, lse_ctx = triton_context_attention(q, k_ctx, v_ctx)
    else:
        out_ctx, lse_ctx = pytorch_context_attention(q, k_ctx, v_ctx)

    # Buffer attention (always PyTorch for now)
    out_buf, lse_buf = pytorch_buffer_attention(q, k_buf, v_buf, mask=buf_mask)

    # Merge using LSE statistics
    out = merge_attention_outputs(out_ctx, lse_ctx, out_buf, lse_buf)

    return out


def cross_attention(
    q: Tensor, k_ctx: Tensor, v_ctx: Tensor, use_triton: bool = True
) -> Tensor:
    """Cross-attention without LSE output. Prefer hybrid_attention for new code."""
    if use_triton and HAS_TRITON and q.is_cuda:
        out, _ = triton_context_attention(q, k_ctx, v_ctx)
        return out
    else:
        out, _ = pytorch_context_attention(q, k_ctx, v_ctx)
        return out
