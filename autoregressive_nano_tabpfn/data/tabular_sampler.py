"""Tabular data sampler using MLP-SCM prior for synthetic regression tasks."""

from typing import Optional, Tuple, List

import numpy as np
import torch

from .data import DataAttr
from .mlp_scm import MLPSCM


class TabularSampler:
    """
    Generate tabular regression data using MLP-SCM prior.

    Creates synthetic regression tasks where X and y are derived from
    a randomly initialized MLP's intermediate representations.
    """

    def __init__(
        self,
        dim_x: int | List[int] = 10,
        dim_y: int = 1,
        # MLP-SCM parameters
        is_causal: bool = True,
        num_causes: Optional[int] = None,
        num_layers: int = 4,
        hidden_dim: int = 64,
        noise_std: float = 0.01,
        sampling: str = "mixed",
        # Normalization
        normalize_y: bool = True,
        # Device/dtype
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize tabular sampler.

        Args:
            dim_x: Number of features (int or list to sample from)
            dim_y: Number of output dimensions (1 for regression)
            is_causal: Whether to use causal SCM generation
            num_causes: Number of root causes (None = dim_x // 2)
            num_layers: MLP depth
            hidden_dim: MLP hidden dimension
            noise_std: Gaussian noise std
            sampling: Initial cause sampling ("normal", "uniform", "mixed")
            normalize_y: Whether to z-normalize targets
            device: Device for generation
            dtype: Data type
        """
        self.dim_x_list = [dim_x] if isinstance(dim_x, int) else list(dim_x)
        self.dim_y = dim_y
        self.device = device
        self.dtype = dtype
        self.normalize_y = normalize_y

        # MLP-SCM parameters
        self.is_causal = is_causal
        self.num_causes = num_causes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.sampling = sampling

    def _generate_function(self, num_samples: int, dim_x: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a single regression function with fixed dimensionality."""

        # Randomize MLP-SCM hyperparameters slightly for diversity
        if self.num_causes is None:
            base_causes = max(1, dim_x // 2)
        else:
            base_causes = int(np.clip(self.num_causes, 1, dim_x))

        lo = max(1, base_causes - 2)
        hi = min(dim_x, base_causes + 3)
        actual_num_causes = int(np.random.randint(lo, hi + 1))

        actual_num_layers = np.random.randint(
            max(2, self.num_layers - 1),
            self.num_layers + 2
        )
        actual_hidden_dim = np.random.randint(
            max(16, self.hidden_dim - 16),
            self.hidden_dim + 32
        )

        model = MLPSCM(
            seq_len=num_samples,
            num_features=dim_x,
            num_outputs=self.dim_y,
            is_causal=self.is_causal,
            num_causes=actual_num_causes,
            y_is_effect=True,
            in_clique=False,
            sort_features=True,
            num_layers=actual_num_layers,
            hidden_dim=actual_hidden_dim,
            mlp_activations=torch.nn.Tanh,
            init_std=np.random.uniform(0.8, 2.0),
            block_wise_dropout=True,
            mlp_dropout_prob=np.random.uniform(0.05, 0.2),
            scale_init_std_by_dropout=True,
            sampling=self.sampling,
            pre_sample_cause_stats=True,
            noise_std=self.noise_std,
            pre_sample_noise_std=True,
            device=self.device,
        )

        with torch.no_grad():
            X, y = model()

        if y.dim() == 1:
            y = y.unsqueeze(-1)

        return X.to(self.dtype), y.to(self.dtype)

    def generate_batch(
        self,
        batch_size: int,
        num_context: Optional[int | List[int]] = None,
        num_buffer: int = 8,
        num_target: int = 128,
        context_range: Optional[Tuple[int, int]] = None,
    ) -> DataAttr:
        """
        Generate a batch of tabular regression tasks.

        All tasks in batch have same feature dimension and context size
        (no padding needed).

        Args:
            batch_size: Number of independent tasks
            num_context: Context size (int, list to sample from, or None for range)
            num_buffer: Buffer size (fixed)
            num_target: Target size (fixed)
            context_range: Range for random context if num_context is None

        Returns:
            DataAttr with xc, yc, xb, yb, xt, yt
        """
        # Choose dimensions ONCE for entire batch
        dim_x = int(np.random.choice(self.dim_x_list))

        # Choose context size
        if num_context is None:
            if context_range is None:
                context_range = (32, 256)
            nc = np.random.randint(context_range[0], context_range[1] + 1)
        elif isinstance(num_context, int):
            nc = num_context
        else:
            nc = int(np.random.choice(num_context))

        nb = num_buffer
        nt = num_target
        total_samples = nc + nb + nt

        xc_list, yc_list = [], []
        xb_list, yb_list = [], []
        xt_list, yt_list = [], []

        for _ in range(batch_size):
            X, y = self._generate_function(total_samples, dim_x)

            # Shuffle
            perm = torch.randperm(total_samples)
            X = X[perm]
            y = y[perm]

            # Split
            xc, yc = X[:nc], y[:nc]
            xb, yb = X[nc:nc + nb], y[nc:nc + nb]
            xt, yt = X[nc + nb:], y[nc + nb:]

            # Normalize y using context statistics
            if self.normalize_y:
                y_mean = yc.mean()
                y_std = yc.std().clamp(min=1e-6)
                yc = (yc - y_mean) / y_std
                yb = (yb - y_mean) / y_std
                yt = (yt - y_mean) / y_std

            xc_list.append(xc)
            yc_list.append(yc)
            xb_list.append(xb)
            yb_list.append(yb)
            xt_list.append(xt)
            yt_list.append(yt)

        # Stack into batches
        return DataAttr(
            xc=torch.stack(xc_list),
            yc=torch.stack(yc_list),
            xb=torch.stack(xb_list) if nb > 0 else torch.zeros(batch_size, 0, dim_x, device=self.device, dtype=self.dtype),
            yb=torch.stack(yb_list) if nb > 0 else torch.zeros(batch_size, 0, self.dim_y, device=self.device, dtype=self.dtype),
            xt=torch.stack(xt_list),
            yt=torch.stack(yt_list),
        )
