"""Online dataset for generating tabular batches on-the-fly."""

from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .data import DataAttr
from .tabular_sampler import TabularSampler


class OnlineTabularDataset(IterableDataset):
    """
    Dataset that generates tabular batches online via TabularSampler.

    Yields pre-batched DataAttr objects for training.

    Args:
        batch_size: Number of tasks per batch
        num_batches: Number of batches per epoch
        d_list: List of feature dimensions to sample from
        nc_list: List of context sizes to sample from
        num_buffer: Fixed buffer size
        num_target: Fixed target size
        normalize_y: Z-normalize targets using context stats
        dtype: Torch dtype
        device: Device for generation
        seed: Optional RNG seed
    """

    def __init__(
        self,
        *,
        batch_size: int,
        num_batches: int,
        d_list: List[int],
        nc_list: List[int],
        num_buffer: int,
        num_target: int,
        normalize_y: bool = True,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.d_list = list(d_list)
        self.nc_list = list(nc_list)
        self.nb = num_buffer
        self.nt = num_target
        self.normalize_y = normalize_y
        self.dtype = dtype
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.sampler = TabularSampler(
            dim_x=self.d_list,
            dim_y=1,
            is_causal=True,
            num_causes=None,
            num_layers=4,
            hidden_dim=64,
            noise_std=0.01,
            sampling="mixed",
            normalize_y=self.normalize_y,
            device=self.device,
            dtype=self.dtype,
        )

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[DataAttr]:
        for _ in range(self.num_batches):
            nc = int(np.random.choice(self.nc_list))
            batch = self.sampler.generate_batch(
                batch_size=self.batch_size,
                num_context=nc,
                num_buffer=self.nb,
                num_target=self.nt,
            )
            yield batch
