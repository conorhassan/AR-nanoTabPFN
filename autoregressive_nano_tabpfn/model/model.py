"""
Autoregressive nanoTabPFN model.

Architecture combining:
- Context/Buffer/Target structure for autoregressive inference
- TabPFN's two-stage attention (feature + row)
- MixtureGaussian head for regression

Training uses flex_attention, inference uses Triton kernels with KV caching.
"""

import math
from typing import Optional, Tuple, List

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

from .attention import MultiheadAttention, create_dense_mask, create_row_mask


class Embedder(nn.Module):
    """
    Embeds tabular data into D-dimensional space.

    Embeds (x, y) pairs, then adds marker embedding to distinguish
    context, buffer, and target sections.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.x_embed = nn.Linear(1, d_model)
        self.y_embed = nn.Linear(1, d_model)

        # Marker embeddings: 0=target, 1=context, 2=buffer
        self.marker_embed = nn.Embedding(3, d_model)
        self._marker_lookup = {"target": 0, "context": 1, "buffer": 2}

    def _get_marker(self, batch_size: int, marker_type: str, device: torch.device) -> Tensor:
        idx = torch.full((batch_size, 1), self._marker_lookup[marker_type],
                        dtype=torch.long, device=device)
        return self.marker_embed(idx)

    def embed_context(self, x: Tensor, y: Tensor) -> Tensor:
        """Embed context (training) data. x: [B, N, C], y: [B, N] or [B, N, 1]"""
        if y.dim() == 2:
            y = y.unsqueeze(-1)
        x_emb = self.x_embed(x.unsqueeze(-1))  # [B, N, C, D]
        y_emb = self.y_embed(y)  # [B, N, 1, D]
        emb = x_emb.mean(dim=2) + y_emb.squeeze(2)  # [B, N, D]
        marker = self._get_marker(x.size(0), "context", x.device)
        return emb + marker

    def embed_buffer(self, x: Tensor, y: Tensor) -> Tensor:
        """Embed buffer (AR) data. AR token needs added afterwards."""
        if y.dim() == 2:
            y = y.unsqueeze(-1)
        x_emb = self.x_embed(x.unsqueeze(-1))
        y_emb = self.y_embed(y)
        emb = x_emb.mean(dim=2) + y_emb.squeeze(2)
        marker = self._get_marker(x.size(0), "buffer", x.device)
        return emb + marker

    def embed_target(self, x: Tensor) -> Tensor:
        """Embed target (test) data. x: [B, T, C], no y values."""
        x_emb = self.x_embed(x.unsqueeze(-1))  # [B, T, C, D]
        emb = x_emb.mean(dim=2)  # [B, T, D]
        marker = self._get_marker(x.size(0), "target", x.device)
        return emb + marker


class TwoStageTransformerLayer(nn.Module):
    """
    Two-stage attention layer.

    Stage 1: Feature attention - dense self-attention across columns [B*R, C, D]
    Stage 2: Row attention - Context/Buffer/Target pattern [B*C, R, D]
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn_features = MultiheadAttention(d_model, n_heads)
        self.attn_rows = MultiheadAttention(d_model, n_heads)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        mask_features: BlockMask,
        mask_rows: BlockMask
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            x: [B, R, C, D] input tensor
            mask_features: Dense mask for feature attention
            mask_rows: Context/Buffer/Target mask for row attention

        Returns:
            output: [B, R, C, D]
            kv_cache: (K, V) from row attention for inference caching
        """
        B, R, C, D = x.shape

        # Feature attention [B*R, C, D]
        x_feat = x.reshape(B * R, C, D)
        attn_out, _ = self.attn_features(x_feat, x_feat, x_feat, mask_features)
        x = self.norm1((attn_out + x_feat).reshape(B, R, C, D))

        # Row attention [B*C, R, D]
        x_row = x.permute(0, 2, 1, 3).reshape(B * C, R, D)
        attn_out, kv = self.attn_rows(x_row, x_row, x_row, mask_rows)
        x = self.norm2((attn_out + x_row).reshape(B, C, R, D).permute(0, 2, 1, 3))

        # FFN
        return self.norm3(x + self.ff(x)), kv


class TwoStageTransformer(nn.Module):
    """Stack of two-stage transformer layers."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            TwoStageTransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        mask_features: BlockMask,
        mask_rows: BlockMask
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        kv_cache = []
        for layer in self.layers:
            x, kv = layer(x, mask_features, mask_rows)
            kv_cache.append(kv)
        return self.norm(x), kv_cache


class MixtureGaussianHead(nn.Module):
    """
    Mixture of Gaussians head for regression.

    Outputs K mixture components, each with mean, std, and weight.
    Assumes num_components >= 2.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dim_y: int = 1,
        num_components: int = 5,
        std_min: float = 1e-3,
    ):
        super().__init__()
        self.dim_y = dim_y
        self.num_components = num_components
        self.std_min = std_min

        # Head outputs: mean, std, weight for each component
        self.head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, num_components * dim_y * 3)
        )

        # Mixture initialization
        self.mean_bias = nn.Parameter(torch.linspace(-1.0, 1.0, num_components))
        delta = 1.0 / (num_components - 1)
        self.std_bias = nn.Parameter(torch.ones(num_components) * self._inv_softplus(delta))
        self.weight_bias = nn.Parameter(torch.zeros(num_components))

    @staticmethod
    def _inv_softplus(y: float) -> float:
        return math.log(math.exp(y) - 1)

    def _parameterize(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert network output to mixture parameters."""
        B, T, _ = z.shape
        K, D = self.num_components, self.dim_y

        raw = self.head(z).view(B, T, K, D, 3)
        raw_mean, raw_std, raw_weight = raw.unbind(dim=-1)

        mean = raw_mean + self.mean_bias[None, None, :, None]
        std = F.softplus(raw_std + self.std_bias[None, None, :, None]).clamp(max=2.0) + self.std_min
        weight = F.softmax(raw_weight + self.weight_bias[None, None, :, None], dim=2)

        return mean, std, weight

    def forward(
        self,
        z: Tensor,
        y: Optional[Tensor] = None,
        loss_mask: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], Tensor, Tensor, Tensor]:
        """
        Args:
            z: [B, T, D] embeddings
            y: [B, T, dim_y] targets (optional, for training)
            loss_mask: [B, T] mask (optional)

        Returns:
            loss: scalar loss (if y provided)
            mean, std, weight: mixture parameters
        """
        mean, std, weight = self._parameterize(z)

        loss = None
        if y is not None:
            ll = self._log_likelihood(y, mean, std, weight)
            if loss_mask is not None:
                ll = ll.mean(-1) * loss_mask
                denom = loss_mask.sum().clamp(min=1)
            else:
                denom = ll.numel()
            loss = -ll.sum() / denom

        return loss, mean, std, weight

    def _log_likelihood(
        self, y: Tensor, mean: Tensor, std: Tensor, weight: Tensor
    ) -> Tensor:
        """Compute log-likelihood under mixture."""
        y = y.unsqueeze(2)  # [B, T, 1, D] for broadcasting with [B, T, K, D]
        log_prob = -0.5 * (math.log(2 * math.pi) + 2 * std.log() + ((y - mean) / std) ** 2)
        log_prob = log_prob + weight.clamp(min=1e-12).log()
        return torch.logsumexp(log_prob, dim=2)  # [B, T, D]

    def sample(self, z: Tensor, num_samples: int = 1) -> Tensor:
        """Sample from the mixture distribution."""
        mean, std, weight = self._parameterize(z)
        B, T, K, D = mean.shape

        weight_flat = weight.permute(0, 1, 3, 2).reshape(B * T * D, K)
        indices = torch.multinomial(weight_flat, num_samples, replacement=True)

        mean_flat = mean.permute(0, 1, 3, 2).reshape(B * T * D, K)
        std_flat = std.permute(0, 1, 3, 2).reshape(B * T * D, K)

        sel_mean = torch.gather(mean_flat, 1, indices).view(B, T, D, num_samples)
        sel_std = torch.gather(std_flat, 1, indices).view(B, T, D, num_samples)

        # Sample from Gaussians
        samples = sel_mean + sel_std * torch.randn_like(sel_mean)
        return samples.permute(0, 1, 3, 2)  # [B, T, num_samples, D]

    def log_likelihood(self, z: Tensor, y: Tensor) -> Tensor:
        """Compute log-likelihood for evaluation."""
        mean, std, weight = self._parameterize(z)
        ll = self._log_likelihood(y, mean, std, weight)
        return ll.sum(dim=-1)  # [B, T]


class ARTabPFN(nn.Module):
    """
    Autoregressive TabPFN.

    - Context/Buffer/Target structure for AR inference
    - TabPFN's two-stage (feature + row) attention
    - MixtureGaussian head for regression

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        num_features: Number of input features (columns)
        buffer_size: Size of AR buffer
        num_components: Number of mixture components (>= 2)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 256,
        num_features: int = 10,
        buffer_size: int = 8,
        num_components: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_features = num_features
        self.buffer_size = buffer_size

        self.embedder = Embedder(d_model)
        self.backbone = TwoStageTransformer(d_model, n_heads, n_layers, d_ff)
        self.head = MixtureGaussianHead(d_model, d_ff, dim_y=1, num_components=num_components)

        # AR position tokens
        self.ar_tokens = nn.Parameter(torch.randn(buffer_size, d_model) * 0.02)

    def forward(
        self,
        x_context: Tensor,
        y_context: Tensor,
        x_buffer: Tensor,
        y_buffer: Tensor,
        x_target: Tensor,
        mask_features: BlockMask,
        mask_rows: BlockMask, 
        y_target: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], Tensor]:
        """
        Forward pass for training.

        Args:
            x_context: [B, Nc, C] context features
            y_context: [B, Nc] context labels
            x_buffer: [B, Nb, C] buffer features
            y_buffer: [B, Nb] buffer labels
            x_target: [B, Nt, C] target features
            y_target: [B, Nt] target labels (optional)
            mask_features: Pre-computed feature attention mask
            mask_rows: Pre-computed row attention mask

        Returns:
            loss: Training loss (if y_target provided)
            mean: Predicted means [B, Nt, K, 1]
        """

        B = x_context.size(0)
        C = x_context.size(2)  # num features

        # Embed context/buffer/targets
        ctx_emb = self.embedder.embed_context(x_context, y_context)   # [B, Nc, D]
        buf_emb = self.embedder.embed_buffer(x_buffer, y_buffer) + self.ar_tokens[:x_buffer.size(1)]
        tgt_emb = self.embedder.embed_target(x_target)                # [B, Nt, D]

        Nc, Nb, Nt = ctx_emb.size(1), buf_emb.size(1), tgt_emb.size(1)
        R = Nc + Nb + Nt  # total rows

        embeddings = torch.cat([ctx_emb, buf_emb, tgt_emb], dim=1)    # [B, R, D]
        embeddings = embeddings.unsqueeze(2)                          # [B, R, 1, D]

        # Forward through transformer
        z, _ = self.backbone(embeddings, mask_features, mask_rows)    # [B, R, 1, D]

        # Extract target embeddings
        z_target = z[:, Nc + Nb:, 0, :]  # [B, Nt, D]

        # Predict
        if y_target is not None and y_target.dim() == 2:
            y_target = y_target.unsqueeze(-1)
        loss, mean, std, weight = self.head(z_target, y_target)

        return loss
