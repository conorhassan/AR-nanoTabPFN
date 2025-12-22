"""Data structures for autoregressive-nanoTabPFN."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DataAttr:
    """Dataclass with context/buffer/target components."""

    xc: Optional[torch.Tensor] = None  # [B, Nc, D] context features
    yc: Optional[torch.Tensor] = None  # [B, Nc, 1] context targets
    xb: Optional[torch.Tensor] = None  # [B, Nb, D] buffer features
    yb: Optional[torch.Tensor] = None  # [B, Nb, 1] buffer targets
    xt: Optional[torch.Tensor] = None  # [B, Nt, D] target features
    yt: Optional[torch.Tensor] = None  # [B, Nt, 1] target targets

    def to(self, device, non_blocking=False):
        """Move all tensors to the specified device."""
        return DataAttr(
            xc=self.xc.to(device, non_blocking=non_blocking) if self.xc is not None else None,
            yc=self.yc.to(device, non_blocking=non_blocking) if self.yc is not None else None,
            xb=self.xb.to(device, non_blocking=non_blocking) if self.xb is not None else None,
            yb=self.yb.to(device, non_blocking=non_blocking) if self.yb is not None else None,
            xt=self.xt.to(device, non_blocking=non_blocking) if self.xt is not None else None,
            yt=self.yt.to(device, non_blocking=non_blocking) if self.yt is not None else None,
        )
