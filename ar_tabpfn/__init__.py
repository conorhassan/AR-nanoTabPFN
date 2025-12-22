"""autoregressive-nanoTabPFN: Autoregressive TabPFN with two-stage attention."""

import torch

_MIN_TORCH_VERSION = (2, 5, 0)
_torch_version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:3])
if _torch_version < _MIN_TORCH_VERSION:
    raise ImportError(
        f"autoregressive-nano-tabpfn requires PyTorch >= {'.'.join(map(str, _MIN_TORCH_VERSION))} "
        f"for flex_attention support. Found torch=={torch.__version__}. "
        f"Please upgrade: pip install --upgrade torch"
    )

from .model import (
    ARTabPFN,
    Embedder,
    TwoStageTransformer,
    TwoStageTransformerLayer,
    MixtureGaussianHead,
    MultiheadAttention,
    create_dense_mask,
    create_row_mask,
    create_context_self_attention_mask,
    clear_mask_cache,
    triton_available,
    cross_attention,
)
from .data import (
    DataAttr,
    MLPSCM,
    TabularSampler,
    OnlineTabularDataset,
)

__all__ = [
    # Model
    "ARTabPFN",
    "Embedder",
    "TwoStageTransformer",
    "TwoStageTransformerLayer",
    "MixtureGaussianHead",
    # Attention
    "MultiheadAttention",
    "create_dense_mask",
    "create_row_mask",
    "create_context_self_attention_mask",
    "clear_mask_cache",
    # Triton
    "triton_available",
    "cross_attention",
    # Data
    "DataAttr",
    "MLPSCM",
    "TabularSampler",
    "OnlineTabularDataset",
]
