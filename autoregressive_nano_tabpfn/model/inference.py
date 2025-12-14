"""
Autoregressive predictor for ARTabPFN with KV caching.

Uses flex_attention for sampling and log-likelihood evaluation.
"""

from typing import Optional

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    BlockMask,
)

from .attention import create_dense_mask


class ARTabPFNPredictor:
    """Methods for joint sampling and log-density evaluation."""

    def __init__(self, embedder, backbone, head, ar_tokens):
        """
        Args:
            embedder: Embedder module (embed_context, embed_buffer, embed_target)
            backbone: TwoStageTransformer module
            head: MixtureGaussianHead module
            ar_tokens: [buffer_size, d_model] AR position embeddings
        """
        self.embedder = embedder
        self.backbone = backbone
        self.head = head
        self.ar_tokens = ar_tokens

        # Cache state (set and updated during inference)
        self.seq_len = 0  # Current committed sequence length
        self._device = None
        self._dtype = None

    @classmethod
    def from_trained_model(cls, model) -> "ARTabPFNPredictor":
        """Create predictor from a trained ARTabPFN model."""
        return cls(
            embedder=model.embedder,
            backbone=model.backbone,
            head=model.head,
            ar_tokens=model.ar_tokens,
        )

    # Cache management
    def init_kv_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Allocate KV cache for each transformer layer.

        Args:
            batch_size: Batch size B
            max_seq_len: Maximum sequence length (context + all buffers)
            device: Device for cache tensors
            dtype: Data type for cache tensors
        """
        if device is None or dtype is None:
            param = next(self.backbone.parameters())
            device = device or param.device
            dtype = dtype or param.dtype

        self._device = device
        self._dtype = dtype

        for layer in self.backbone.layers:
            H = layer.attn_rows.n_heads
            Dh = layer.attn_rows.head_dim
            layer.k_cache = torch.zeros(
                batch_size, H, max_seq_len, Dh, device=device, dtype=dtype
            )
            layer.v_cache = torch.zeros_like(layer.k_cache)

        self.seq_len = 0

    def clear_cache(self) -> None:
        """Reset cache state (keeps allocated memory)."""
        self.seq_len = 0

    @torch.no_grad()
    def prefill_context(self, x_context: Tensor, y_context: Tensor) -> None:
        """
        Encode context and populate KV cache.

        After this call:
            - Cache positions [0, Nc) contain context KV
            - seq_len = Nc

        Args:
            x_context: [B, Nc, num_features] context features
            y_context: [B, Nc] context targets
        """
        # Embed context
        ctx_emb = self.embedder.embed_context(x_context, y_context)  # [B, Nc, D]
        B, Nc, D = ctx_emb.shape

        # Expand for two-stage attention: [B, Nc, 1, D]
        x = ctx_emb.unsqueeze(2)

        # Get masks (context uses dense self-attention, not causal)
        feature_mask = create_dense_mask(seq_len=1, device=x.device)
        row_mask = create_dense_mask(seq_len=Nc, device=x.device)

        # Run through transformer layers, caching KV from row attention
        for layer in self.backbone.layers:
            x = self._layer_forward_with_cache(
                layer, x, feature_mask, row_mask, cache_start=0
            )

        # Apply final norm (though we don't need the output, just the cached KV)
        self.seq_len = Nc

    # Autoregressive decoding
    @torch.no_grad()
    def transformer_decode(self, embedding: Tensor, commit: int) -> Tensor:
        """
        Process new embeddings using cached KV.

        Args:
            embedding: [B, N, D] new token embeddings
            commit: How many positions to commit to cache
                    - N-1 for [buffer, target]: commit buffer only
                    - 0 for [target] only: don't commit

        Returns:
            [B, N, D] transformed embeddings
        """
        B, N, D = embedding.shape

        # Expand for two-stage attention: [B, N, 1, D]
        x = embedding.unsqueeze(2)

        # Feature mask (trivial C=1)
        feature_mask = create_dense_mask(seq_len=1, device=x.device)

        row_mask = self._create_decode_mask(
            num_cached=self.seq_len, num_new=N, device=x.device
        )

        # Run through transformer layers
        for layer in self.backbone.layers:
            x = self._layer_decode_with_cache(
                layer, x, feature_mask, row_mask, cache_start=self.seq_len
            )

        # Commit only the first `commit` positions
        self.seq_len += commit

        # Apply final norm and squeeze
        return self.backbone.norm(x.squeeze(2))

    @torch.no_grad()
    def autoregressive_decode(
        self,
        x_target: Tensor,
        prev_x: Optional[Tensor] = None,
        prev_y: Optional[Tensor] = None,
        ar_idx: int = 0,
    ) -> Tensor:
        """
        Decode one target, optionally with previous prediction as buffer element.

        Args:
            x_target: [B, 1, num_features] current target features
            prev_x: [B, 1, num_features] previous target features (for buffer)
            prev_y: [B, 1] previous prediction (for buffer)
            ar_idx: AR token index (position in buffer)

        Returns:
            y_pred: [B, 1] sampled prediction
        """
        # Embed current target
        target_emb = self.embedder.embed_target(x_target)  # [B, 1, D]

        if prev_x is not None and prev_y is not None:
            # Create buffer embedding from previous prediction
            buffer_emb = self.embedder.embed_buffer(prev_x, prev_y)  # [B, 1, D]
            buffer_emb = buffer_emb + self.ar_tokens[ar_idx]

            # Batch [buffer, target] together
            embedding = torch.cat([buffer_emb, target_emb], dim=1)  # [B, 2, D]
            commit = 1  # Commit buffer, not target
        else:
            # First target, no buffer
            embedding = target_emb  # [B, 1, D]
            commit = 0  # Don't commit target

        # Decode
        z = self.transformer_decode(embedding, commit=commit)

        # Sample from last position (the target)
        y_pred = self.head.sample(z[:, -1:, :])  # [B, 1, num_samples, D]

        return y_pred[:, :, 0, 0]  # [B, 1]

    # User-level functions
    @torch.no_grad()
    def sample_sequence(
        self,
        x_context: Tensor,
        y_context: Tensor,
        x_target: Tensor,
    ) -> Tensor:
        """
        Predict all targets autoregressively.

        Args:
            x_context: [B, Nc, num_features] context features
            y_context: [B, Nc] context targets
            x_target: [B, Nt, num_features] target features

        Returns:
            y_pred: [B, Nt] predictions
        """
        B, Nc, num_features = x_context.shape
        Nt = x_target.shape[1]
        device, dtype = x_context.device, x_context.dtype

        # Setup cache
        max_seq = Nc + Nt  # Context + all buffers
        self.init_kv_cache(B, max_seq, device, dtype)
        self.prefill_context(x_context, y_context)

        # Autoregressive generation
        predictions = []
        prev_x, prev_y = None, None

        for t in range(Nt):
            x_t = x_target[:, t : t + 1, :]  # [B, 1, num_features]
            ar_idx = t % self.ar_tokens.shape[0]

            y_t = self.autoregressive_decode(x_t, prev_x, prev_y, ar_idx)
            predictions.append(y_t)

            prev_x, prev_y = x_t, y_t

        return torch.cat(predictions, dim=1)  # [B, Nt]

    @torch.no_grad()
    def evaluate_joint_density(
        self,
        x_context: Tensor,
        y_context: Tensor,
        x_target: Tensor,
        y_target: Tensor,
    ) -> Tensor:
        """
        Compute log-density of targets using teacher forcing (single forward pass).

        Returns:
            log_density: [B, Nt] log-density of each y_target under the model
        """
        B, Nc, _ = x_context.shape
        Nt = x_target.shape[1]
        device, dtype = x_context.device, x_context.dtype

        self.init_kv_cache(B, Nc + 2 * Nt, device, dtype)
        self.prefill_context(x_context, y_context)

        # Embed all buffers from (x_target, y_target) with AR position embeddings
        buffer_emb = self.embedder.embed_buffer(x_target, y_target)
        ar_positions = torch.arange(Nt, device=device) % self.ar_tokens.shape[0]
        buffer_emb = buffer_emb + self.ar_tokens[ar_positions]

        # Embed all targets
        target_emb = self.embedder.embed_target(x_target)

        # [Buffer_0..Nt-1, Target_0..Nt-1]
        embedding = torch.cat([buffer_emb, target_emb], dim=1)

        # Single forward pass with teacher forcing mask
        z = self._teacher_forcing_decode(embedding, Nt)

        # Extract target representations and compute log-density
        z_targets = z[:, Nt:, :]
        return self.head.log_likelihood(z_targets, y_target.unsqueeze(-1))

    @torch.no_grad()
    def _teacher_forcing_decode(self, embedding: Tensor, num_targets: int) -> Tensor:
        """Process [buffers, targets] with teacher forcing mask."""
        B, N, D = embedding.shape
        x = embedding.unsqueeze(2)

        feature_mask = create_dense_mask(seq_len=1, device=x.device)
        row_mask = self._create_teacher_forcing_mask(
            num_cached=self.seq_len, num_targets=num_targets, device=x.device
        )

        for layer in self.backbone.layers:
            x = self._layer_decode_with_cache(
                layer, x, feature_mask, row_mask, cache_start=self.seq_len
            )

        return self.backbone.norm(x.squeeze(2))

    # Internal helpers
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> BlockMask:
        """Create causal self-attention mask for prefill."""

        def causal_mod(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        return create_block_mask(
            causal_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device
        )

    def _create_decode_mask(
        self, num_cached: int, num_new: int, device: torch.device
    ) -> BlockMask:
        """
        Create mask for decode step.

        Structure: [cached context] [buffer_0..buffer_{n-2}] [target]
        - Buffers attend to: context + causal within buffers
        - Target (last position) attends to: context + all buffers (no self-attention)
        """
        total = num_cached + num_new

        def decode_mod(b, h, q_idx, kv_idx):
            is_target = q_idx == num_new - 1
            is_buffer = q_idx < num_new - 1

            target_pattern = is_target & (kv_idx < num_cached + q_idx)
            buffer_pattern = is_buffer & (kv_idx < num_cached + q_idx + 1)

            return target_pattern | buffer_pattern

        return create_block_mask(
            decode_mod, B=None, H=None, Q_LEN=num_new, KV_LEN=total, device=device
        )

    def _create_teacher_forcing_mask(
        self, num_cached: int, num_targets: int, device: torch.device
    ) -> BlockMask:
        """
        Create mask for batched teacher forcing evaluation.

        Sequence structure for new tokens: [Buffer_0, ..., Buffer_{Nt-1}, Target_0, ..., Target_{Nt-1}]
        - Buffers: positions [0, Nt) in new tokens
        - Targets: positions [Nt, 2*Nt) in new tokens

        Attention pattern:
        - Buffer_i attends to: context (all), buffers [0..i] (causal)
        - Target_i attends to: context (all), buffers [0..i-1] (strictly < i), NO other targets
        """
        Nt = num_targets
        num_new = 2 * Nt
        total = num_cached + num_new

        def teacher_forcing_mod(b, h, q_idx, kv_idx):
            # q_idx in [0, 2*Nt): [0, Nt) are buffers, [Nt, 2*Nt) are targets
            # kv_idx in [0, num_cached + 2*Nt)

            # Everyone attends to context
            attends_context = kv_idx < num_cached

            # Buffer queries (q_idx < Nt): causal within buffers
            is_buffer_query = q_idx < Nt
            buffer_start = num_cached
            kv_is_buffer = (kv_idx >= buffer_start) & (kv_idx < buffer_start + Nt)
            buffer_kv_idx = kv_idx - buffer_start  # Which buffer position
            buffer_causal = kv_is_buffer & (buffer_kv_idx <= q_idx)

            # Target queries (q_idx >= Nt): attend to buffers [0, target_idx)
            is_target_query = q_idx >= Nt
            target_idx = q_idx - Nt  # Which target (0..Nt-1)
            # Target_i attends to Buffer_j where j < i (strictly less than)
            target_to_buffer = kv_is_buffer & (buffer_kv_idx < target_idx)

            # Combine: buffers use buffer_causal, targets use target_to_buffer
            return (
                attends_context
                | (is_buffer_query & buffer_causal)
                | (is_target_query & target_to_buffer)
            )

        return create_block_mask(
            teacher_forcing_mod,
            B=None,
            H=None,
            Q_LEN=num_new,
            KV_LEN=total,
            device=device,
        )

    def _layer_forward_with_cache(
        self,
        layer,
        x: Tensor,
        feature_mask: BlockMask,
        row_mask: BlockMask,
        cache_start: int,
    ) -> Tensor:
        """
        Forward through one layer during prefill, caching row attention KV.

        Args:
            layer: TwoStageTransformerLayer
            x: [B, R, C, D] input (C=1)
            feature_mask: Mask for feature attention
            row_mask: Mask for row attention
            cache_start: Where to start writing in cache

        Returns:
            [B, R, C, D] output
        """
        B, R, C, D = x.shape

        # Feature attention
        x_feat = x.reshape(B * R, C, D)
        attn_out, _ = layer.attn_features(x_feat, x_feat, x_feat, feature_mask)
        x = layer.norm1((attn_out + x_feat).reshape(B, R, C, D))

        # Row attention - cache KV here
        x_row = x.squeeze(2)

        H = layer.attn_rows.n_heads
        Dh = layer.attn_rows.head_dim

        q = layer.attn_rows.q_proj(x_row).view(B, R, H, Dh).transpose(1, 2)
        k = layer.attn_rows.k_proj(x_row).view(B, R, H, Dh).transpose(1, 2)
        v = layer.attn_rows.v_proj(x_row).view(B, R, H, Dh).transpose(1, 2)

        # Cache K, V
        layer.k_cache[:, :, cache_start : cache_start + R, :] = k
        layer.v_cache[:, :, cache_start : cache_start + R, :] = v

        # Attention
        attn_out = flex_attention(q, k, v, block_mask=row_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, R, D)
        attn_out = layer.attn_rows.out_proj(attn_out)

        x_row = layer.norm2(x_row + attn_out)

        # FFN
        x_out = layer.norm3(x_row + layer.ff(x_row))

        return x_out.unsqueeze(2)  # [B, R, 1, D]

    def _layer_decode_with_cache(
        self,
        layer,
        x: Tensor,
        feature_mask: BlockMask,
        row_mask: BlockMask,
        cache_start: int,
    ) -> Tensor:
        """
        Decode through one layer using cached KV.

        Args:
            layer: TwoStageTransformerLayer
            x: [B, N, C, D] new tokens (C=1)
            feature_mask: Mask for feature attention
            row_mask: Mask for row attention (Q_LEN=N, KV_LEN=cached+N)
            cache_start: Current cache position (= self.seq_len)

        Returns:
            [B, N, C, D] output
        """
        B, N, C, D = x.shape

        # Feature attention
        x_feat = x.reshape(B * N, C, D)
        attn_out, _ = layer.attn_features(x_feat, x_feat, x_feat, feature_mask)
        x = layer.norm1((attn_out + x_feat).reshape(B, N, C, D))

        # Row attention with KV cache
        x_row = x.squeeze(2)  # [B, N, D]

        H = layer.attn_rows.n_heads
        Dh = layer.attn_rows.head_dim

        q = layer.attn_rows.q_proj(x_row).view(B, N, H, Dh).transpose(1, 2)
        k_new = layer.attn_rows.k_proj(x_row).view(B, N, H, Dh).transpose(1, 2)
        v_new = layer.attn_rows.v_proj(x_row).view(B, N, H, Dh).transpose(1, 2)

        # Write new K, V to cache (temporary until committed)
        layer.k_cache[:, :, cache_start : cache_start + N, :] = k_new
        layer.v_cache[:, :, cache_start : cache_start + N, :] = v_new

        # Get full K, V from cache
        total_len = cache_start + N
        k_full = layer.k_cache[:, :, :total_len, :]
        v_full = layer.v_cache[:, :, :total_len, :]

        # Attention
        attn_out = flex_attention(q, k_full, v_full, block_mask=row_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        attn_out = layer.attn_rows.out_proj(attn_out)

        x_row = layer.norm2(x_row + attn_out)

        # FFN
        x_out = layer.norm3(x_row + layer.ff(x_row))

        return x_out.unsqueeze(2)  # [B, N, 1, D]
