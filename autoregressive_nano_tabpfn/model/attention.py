"""
flex_attention-based attention for autoregressive-nanoTabPFN.

Mask patterns adapted for TabPFN's two-stage attention:
- Feature attention: dense self-attention across columns
- Row attention: Context/Buffer/Target pattern for efficient autoregressive inference

Sequence structure for row attention:
    [Context (train rows)] [Buffer (AR tokens)] [Target (test rows)]
"""

from typing import Tuple
import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    or_masks,
    BlockMask,
)

# Compile flex_attention only on CUDA (CPU doesn't support compiled flex_attention)
if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)


class MultiheadAttention(nn.Module):
    """Multi-head attention using flex_attention. Returns KV for caching."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, block_mask: BlockMask
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        B, Sq, D = q.shape
        _, Skv, _ = k.shape

        # Project and reshape
        qh = self.q_proj(q).view(B, Sq, self.n_heads, self.head_dim).transpose(1, 2)
        kh = self.k_proj(k).view(B, Skv, self.n_heads, self.head_dim).transpose(1, 2)
        vh = self.v_proj(v).view(B, Skv, self.n_heads, self.head_dim).transpose(1, 2)

        # Forward
        out = flex_attention(qh, kh, vh, block_mask=block_mask)

        # Reshape back
        out = out.transpose(1, 2).reshape(B, Sq, D)
        return self.out_proj(out), (kh, vh)


# Mask cache to avoid recompilation
_mask_cache = {}


def create_dense_mask(seq_len: int, device: str = "cuda") -> BlockMask:
    """Dense self-attention mask for feature dimension (all attend to all)."""
    key = ("dense", seq_len, device)
    if key not in _mask_cache:

        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= 0  # Always true

        _mask_cache[key] = create_block_mask(
            mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device
        )
    return _mask_cache[key]


def create_row_mask(
    num_rows: int,
    context_len: int,
    buffer_len: int,
    attending_chunks: int | None = None,
    device: str = "cuda",
) -> BlockMask:
    """
    Row attention mask with Context/Buffer/Target sections.

    Args:
        num_rows: Total number of rows (context + buffer + target)
        context_len: Number of context (training) rows
        buffer_len: Number of buffer (AR) tokens
        attending_chunks: Number of target chunks that attend to buffer.
            If None, defaults to half the target chunks (target_len // (2 * buffer_len)).
            Target length must be 2 * N * buffer_len for some integer N.
        device: Device for mask tensors

    Returns:
        BlockMask implementing the ACE attention pattern
    """
    target_len = num_rows - context_len - buffer_len

    if attending_chunks is None:
        attending_chunks = target_len // (2 * buffer_len)

    key = ("row", num_rows, context_len, buffer_len, attending_chunks, device)
    if key not in _mask_cache:
        target_start = context_len + buffer_len

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        def prefix_mask(b, h, q_idx, kv_idx):
            """All tokens can attend to context."""
            return kv_idx < context_len

        def localized_causal_ctx_buf(b, h, q_idx, kv_idx):
            """Causal attention within context+buffer region."""
            ctx_buf_end = context_len + buffer_len
            q_in_region = q_idx < ctx_buf_end
            kv_in_region = kv_idx < ctx_buf_end
            return q_in_region & kv_in_region & causal_mask(b, h, q_idx, kv_idx)

        def chunked_target_buffer(b, h, q_idx, kv_idx):
            """First N chunks of targets attend causally to buffer."""
            buffer_start = context_len

            q_in_target = q_idx >= target_start
            kv_in_buffer = (kv_idx >= buffer_start) & (kv_idx < target_start)

            base_condition = q_in_target & kv_in_buffer

            target_offset = q_idx - target_start
            buffer_offset = kv_idx - buffer_start

            # First attending_chunks * buffer_len positions attend
            in_attending_region = target_offset < (attending_chunks * buffer_len)

            # Causal within chunk
            chunk_position = target_offset % buffer_len
            causal = buffer_offset <= chunk_position

            return base_condition & in_attending_region & causal

        def diag_mask(b, h, q_idx, kv_idx):
            """Self-attention on diagonal."""
            return q_idx == kv_idx

        # Combine all masks
        final_mask_mod = or_masks(
            prefix_mask,
            localized_causal_ctx_buf,
            diag_mask,
            chunked_target_buffer,
        )
        final_mask_mod.__name__ = f"row_mask_{num_rows}_{context_len}_{buffer_len}"

        _mask_cache[key] = create_block_mask(
            final_mask_mod, B=None, H=None, Q_LEN=num_rows, KV_LEN=num_rows, device=device
        )
    return _mask_cache[key]


def create_context_self_attention_mask(context_len: int, device: str = "cuda") -> BlockMask:
    """Dense self-attention mask for context encoding (used in inference)."""
    key = ("context_self", context_len, device)
    if key not in _mask_cache:

        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= 0  # Always true (dense)

        _mask_cache[key] = create_block_mask(
            mask_mod, B=None, H=None, Q_LEN=context_len, KV_LEN=context_len, device=device
        )
    return _mask_cache[key]


def clear_mask_cache():
    """Clear cached masks."""
    _mask_cache.clear()
