"""
MLP-based Structural Causal Model (SCM) for synthetic tabular data generation.

Based on TabICL: https://github.com/soda-inria/tabicl
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
from torch import nn


class GaussianNoise(nn.Module):
    """Adds Gaussian noise to inputs."""

    def __init__(self, std: float | torch.Tensor = 0.01):
        super().__init__()
        if isinstance(std, torch.Tensor):
            self.register_buffer("std", std)
        else:
            self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


class XSampler:
    """Samples initial cause variables for the SCM."""

    def __init__(
        self,
        seq_len: int,
        num_causes: int,
        pre_stats: bool = False,
        sampling: str = "normal",
        device: str = "cpu",
    ):
        self.seq_len = seq_len
        self.num_causes = num_causes
        self.pre_stats = pre_stats
        self.sampling = sampling
        self.device = device

        if pre_stats:
            self._pre_stats()
        else:
            self.means = None
            self.stds = None

    def _pre_stats(self):
        """Pre-sample mean and std for normal distributions."""
        means = np.random.normal(0, 1, self.num_causes)
        stds = np.abs(np.random.normal(0, 1, self.num_causes) * means)
        self.means = (
            torch.tensor(means, dtype=torch.float, device=self.device)
            .unsqueeze(0)
            .repeat(self.seq_len, 1)
        )
        self.stds = (
            torch.tensor(stds, dtype=torch.float, device=self.device)
            .unsqueeze(0)
            .repeat(self.seq_len, 1)
        )

    def sample(self) -> torch.Tensor:
        """Sample cause variables. Returns (seq_len, num_causes)."""
        if self.sampling == "normal":
            return self._sample_normal_all()
        elif self.sampling == "uniform":
            return torch.rand(self.seq_len, self.num_causes, device=self.device)
        elif self.sampling == "mixed":
            return self._sample_mixed()
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling}")

    def _sample_normal_all(self) -> torch.Tensor:
        """Sample all features from normal distribution."""
        if self.means is not None:
            return torch.normal(self.means, self.stds.abs()).float()
        else:
            return torch.normal(
                0.0, 1.0, (self.seq_len, self.num_causes), device=self.device
            ).float()

    def _sample_normal(self, n: int) -> torch.Tensor:
        """Sample single feature from normal distribution."""
        if self.means is not None:
            return torch.normal(self.means[:, n], self.stds[:, n].abs()).float()
        else:
            return torch.normal(0.0, 1.0, (self.seq_len,), device=self.device).float()

    def _sample_multinomial(self) -> torch.Tensor:
        """Sample from weighted multinomial distribution."""
        n_categories = random.randint(2, 20)
        probs = torch.rand(n_categories, device=self.device)
        x = torch.multinomial(probs, self.seq_len, replacement=True)
        x = x.float()
        return (x - x.mean()) / x.std()

    def _sample_zipf(self) -> torch.Tensor:
        """Sample from Zipf distribution (centered, not scaled)."""
        x = np.random.zipf(2.0 + random.random() * 2, (self.seq_len,))
        x = torch.tensor(x, device=self.device).clamp(max=10)
        x = x.float()
        return x - x.mean()

    def _sample_mixed(self) -> torch.Tensor:
        """Sample using probability-based mixture of distributions.

        Uses conditional probability logic that biases toward normal (~67%).
        """
        X = []
        zipf_p = random.random() * 0.66
        multi_p = random.random() * 0.66
        normal_p = random.random() * 0.66

        for n in range(self.num_causes):
            if random.random() > normal_p:
                x = self._sample_normal(n)
            elif random.random() > multi_p:
                x = self._sample_multinomial()
            elif random.random() > zipf_p:
                x = self._sample_zipf()
            else:
                x = torch.rand((self.seq_len,), device=self.device)
            X.append(x)

        return torch.stack(X, -1)


class MLPSCM(nn.Module):
    """
    Generates synthetic tabular datasets using an MLP-based Structural Causal Model.

    Creates regression data where features X and targets y are derived from
    intermediate representations of a randomly initialized MLP applied to
    sampled "cause" variables.
    """

    def __init__(
        self,
        seq_len: int = 1024,
        num_features: int = 10,
        num_outputs: int = 1,
        is_causal: bool = True,
        num_causes: int | None = None,
        y_is_effect: bool = True,
        in_clique: bool = False,
        sort_features: bool = True,
        num_layers: int = 4,
        hidden_dim: int = 64,
        mlp_activations: Any = nn.Tanh,
        init_std: float = 1.0,
        block_wise_dropout: bool = True,
        mlp_dropout_prob: float = 0.1,
        scale_init_std_by_dropout: bool = True,
        sampling: str = "mixed",
        pre_sample_cause_stats: bool = True,
        noise_std: float = 0.01,
        pre_sample_noise_std: bool = True,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.is_causal = is_causal
        self.y_is_effect = y_is_effect
        self.in_clique = in_clique
        self.sort_features = sort_features
        self.num_layers = max(num_layers, 2)
        self.hidden_dim = hidden_dim
        self.mlp_activations = mlp_activations
        self.init_std = init_std
        self.block_wise_dropout = block_wise_dropout
        self.mlp_dropout_prob = mlp_dropout_prob
        self.scale_init_std_by_dropout = scale_init_std_by_dropout
        self.sampling = sampling
        self.pre_sample_cause_stats = pre_sample_cause_stats
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std
        self.device = device

        # Set num_causes
        if num_causes is None:
            self.num_causes = max(1, num_features // 2)
        else:
            self.num_causes = num_causes

        if self.is_causal:
            # Ensure enough intermediate variables for sampling X and y
            self.hidden_dim = max(
                self.hidden_dim, self.num_outputs + 2 * self.num_features
            )
        else:
            self.num_causes = self.num_features

        # Input sampler
        self.xsampler = XSampler(
            self.seq_len,
            self.num_causes,
            pre_stats=self.pre_sample_cause_stats,
            sampling=self.sampling,
            device=self.device,
        )

        # Build MLP layers
        layers = [nn.Linear(self.num_causes, self.hidden_dim)]
        for _ in range(self.num_layers - 1):
            layers.append(self._make_layer_block())
        if not self.is_causal:
            layers.append(self._make_layer_block(is_output=True))

        self.layers = nn.Sequential(*layers).to(device)
        self._init_parameters()

    def _make_layer_block(self, is_output: bool = False) -> nn.Sequential:
        """Create activation -> linear -> noise block."""
        out_dim = self.num_outputs if is_output else self.hidden_dim

        if self.pre_sample_noise_std:
            noise_std = (
                torch.abs(torch.randn(1, out_dim, device=self.device) * self.noise_std)
                + 1e-6
            )
        else:
            noise_std = self.noise_std

        return nn.Sequential(
            self.mlp_activations(),
            nn.Linear(self.hidden_dim, out_dim),
            GaussianNoise(noise_std),
        )

    def _init_parameters(self):
        """Initialize MLP parameters."""
        for i, (name, param) in enumerate(self.layers.named_parameters()):
            if self.block_wise_dropout and param.dim() == 2:
                self._init_block_dropout(param)
            else:
                self._init_normal(param, i)

    def _init_block_dropout(self, param: torch.Tensor):
        """Block-wise sparse initialization."""
        nn.init.zeros_(param)
        n_blocks = random.randint(1, math.ceil(math.sqrt(min(param.shape))))
        block_size = [dim // n_blocks for dim in param.shape]
        if block_size[0] == 0 or block_size[1] == 0:
            nn.init.normal_(param, std=self.init_std)
            return

        keep_prob = (n_blocks * block_size[0] * block_size[1]) / param.numel()
        std = self.init_std / (
            math.sqrt(keep_prob) if self.scale_init_std_by_dropout else 1
        )

        for b in range(n_blocks):
            slices = tuple(slice(d * b, d * (b + 1)) for d in block_size)
            nn.init.normal_(param[slices], std=std)

    def _init_normal(self, param: torch.Tensor, idx: int):
        """Standard normal initialization with dropout."""
        if param.dim() != 2:
            return

        dropout = self.mlp_dropout_prob if idx > 0 else 0
        dropout = min(dropout, 0.99)
        std = self.init_std / (
            math.sqrt(1 - dropout) if self.scale_init_std_by_dropout else 1
        )

        nn.init.normal_(param, std=std)
        if dropout > 0:
            mask = torch.bernoulli(torch.full_like(param, 1 - dropout))
            param.data *= mask

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic (X, y) data."""
        causes = self.xsampler.sample()  # [seq_len, num_causes]

        # Forward through MLP, collecting intermediate outputs
        outputs = [causes]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))

        # Skip first two (causes and first linear-only layer)
        outputs = outputs[2:]

        X, y = self._extract_xy(causes, outputs)

        # Handle NaNs
        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)):
            X = torch.zeros_like(X)
            y = torch.zeros_like(y)

        if self.num_outputs == 1:
            y = y.squeeze(-1)

        return X, y

    def _extract_xy(
        self, causes: torch.Tensor, outputs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract features X and targets y from MLP outputs."""
        if not self.is_causal:
            return causes, outputs[-1]

        # Causal mode: sample X and y from intermediate representations
        outputs_flat = torch.cat(outputs, dim=-1)
        total_dim = outputs_flat.shape[-1]

        if self.in_clique:
            # Block sampling with random permutation within block
            max_start = total_dim - self.num_outputs - self.num_features
            start = random.randint(0, max(0, max_start))
            perm = start + torch.randperm(
                self.num_outputs + self.num_features, device=self.device
            )
        else:
            # Random sampling
            perm = torch.randperm(total_dim - 1, device=self.device)

        idx_X = perm[self.num_outputs : self.num_outputs + self.num_features]

        if self.y_is_effect:
            idx_y = list(range(-self.num_outputs, 0))
        else:
            idx_y = perm[: self.num_outputs]

        if self.sort_features:
            idx_X, _ = torch.sort(idx_X)

        X = outputs_flat[:, idx_X]
        y = outputs_flat[:, idx_y]

        return X, y
