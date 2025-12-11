"""
Visualization utilities for attention masks.

Run this script to generate mask visualizations:
    python -m autoregressive_nano_tabpfn.tests.visualize_masks

Outputs PNG files showing the attention patterns for:
- Feature attention (dense)
- Row attention (ACE-style Context/Buffer/Target)
"""

from pathlib import Path
from typing import Optional

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl


def visualize_row_mask(
    context_len: int,
    buffer_len: int,
    target_len: int,
    attending_chunks: Optional[int] = None,
    path: Optional[Path] = None,
    title: Optional[str] = None,
) -> Path:
    """
    Visualize the ACE-style row attention mask.

    Args:
        context_len: Number of context (training) rows
        buffer_len: Number of buffer (AR) tokens
        target_len: Number of target (test) rows
        attending_chunks: Number of target chunks that attend to buffer.
            If None, defaults to half the target chunks (target_len // (2 * buffer_len)).
        path: Output path for PNG
        title: Optional figure title

    Returns:
        Path to saved PNG file
    """
    C = int(context_len)
    B = int(buffer_len)
    T = int(target_len)
    total = C + B + T

    if attending_chunks is None:
        attending_chunks = T // (2 * B)

    ctx_end = C
    buf_start, buf_end = C, C + B
    tgt_start = C + B

    # Build index grids
    m = torch.arange(0, total)
    n = torch.arange(0, total)
    M, N = torch.meshgrid(m, n, indexing="ij")

    # Regions
    q_ctx = M < ctx_end
    k_ctx = N < ctx_end
    q_buf = (M >= buf_start) & (M < buf_end)
    k_buf = (N >= buf_start) & (N < buf_end)
    q_tgt = M >= tgt_start
    causal = M >= N

    # Allowed attention patterns (matching ACE masks)
    comp_ctx_self = q_ctx & k_ctx  # Context dense self-attention
    comp_buf_self = q_buf & k_buf & causal  # Buffer causal self-attention
    comp_tgt_ctx = q_tgt & k_ctx  # Target -> Context
    comp_buf_ctx = q_buf & k_ctx  # Buffer -> Context

    # Target -> Buffer (chunked causal)
    target_offset = M - tgt_start
    buffer_offset = N - buf_start
    in_attending_region = target_offset < (attending_chunks * B)
    chunk_position = target_offset % max(B, 1)
    causal_chunk = buffer_offset <= chunk_position
    comp_tgt_buf = q_tgt & k_buf & in_attending_region & causal_chunk

    # Combined allowed mask
    allowed = comp_ctx_self | comp_buf_self | comp_tgt_ctx | comp_buf_ctx | comp_tgt_buf

    # Use same colormap as feature mask for consistency
    try:
        import seaborn as sns

        cmap = sns.color_palette("YlOrBr", as_cmap=True)
    except ImportError:
        cmap = plt.cm.YlOrBr

    # Navy for blocked, muted yellow for allowed (same as feature mask)
    navy = torch.tensor([13 / 255.0, 27 / 255.0, 42 / 255.0], dtype=torch.float32)
    rgba = cmap(0.35)

    h, w = total, total
    img = torch.zeros((h, w, 3), dtype=torch.float32)
    img[:, :, 0] = navy[0]
    img[:, :, 1] = navy[1]
    img[:, :, 2] = navy[2]

    img[:, :, 0][allowed] = rgba[0]
    img[:, :, 1][allowed] = rgba[1]
    img[:, :, 2][allowed] = rgba[2]

    # Font settings
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [
        "TeX Gyre Termes",
        "STIX Two Text",
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif",
    ]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img.numpy(), aspect="auto", interpolation="nearest")
    ax.set_xlabel(r"$K/V$ (rows)", labelpad=10, fontsize=16)
    ax.set_ylabel(r"$Q$ (rows)", labelpad=10, fontsize=16)

    if title is None:
        title = f"Row Attention Mask (C={C}, B={B}, T={T})"
    ax.set_title(title, pad=10, fontsize=16)

    ax.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.2)

    ax.add_patch(
        mpatches.Rectangle((-0.5, -0.5), w, h, fill=False, edgecolor="black", linewidth=1.0)
    )

    plt.tight_layout()

    out_path = Path(path) if path is not None else Path("row_mask.png")
    out_path = out_path.with_suffix(".png")
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)
    return out_path


def visualize_dense_mask(
    seq_len: int,
    path: Optional[Path] = None,
    title: Optional[str] = None,
) -> Path:
    """
    Visualize dense (all-to-all) attention mask for feature dimension.

    Args:
        seq_len: Sequence length (number of features/columns)
        path: Output path for PNG
        title: Optional figure title

    Returns:
        Path to saved PNG file
    """
    # Use same colormap as row mask for consistency
    try:
        import seaborn as sns

        cmap = sns.color_palette("YlOrBr", as_cmap=True)
    except ImportError:
        cmap = plt.cm.YlOrBr

    # All positions allowed - use same color as context self-attention (val=0.35)
    rgba = cmap(0.35)

    img = torch.zeros((seq_len, seq_len, 3), dtype=torch.float32)
    img[:, :, 0] = rgba[0]
    img[:, :, 1] = rgba[1]
    img[:, :, 2] = rgba[2]

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [
        "TeX Gyre Termes",
        "STIX Two Text",
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif",
    ]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img.numpy(), aspect="auto", interpolation="nearest")
    ax.set_xlabel(r"$K/V$ (features)", labelpad=10, fontsize=16)
    ax.set_ylabel(r"$Q$ (features)", labelpad=10, fontsize=16)

    if title is None:
        title = f"Feature Attention Mask (dense, {seq_len}x{seq_len})"
    ax.set_title(title, pad=10, fontsize=16)

    ax.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.2)

    ax.add_patch(
        mpatches.Rectangle(
            (-0.5, -0.5), seq_len, seq_len, fill=False, edgecolor="black", linewidth=1.0
        )
    )

    plt.tight_layout()

    out_path = Path(path) if path is not None else Path("feature_mask.png")
    out_path = out_path.with_suffix(".png")
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize attention masks")
    parser.add_argument("--context", "-c", type=int, default=16, help="Context (train) length")
    parser.add_argument("--buffer", "-b", type=int, default=8, help="Buffer length")
    parser.add_argument(
        "--target",
        "-t",
        type=int,
        default=32,
        help="Target (test) length. Must be 2*N*buffer for integer N.",
    )
    parser.add_argument("--features", "-f", type=int, default=8, help="Number of features")
    parser.add_argument("--output-dir", "-o", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating masks with C={args.context}, B={args.buffer}, T={args.target}, F={args.features}"
    )

    # Row attention mask (attending_chunks auto-computed as target // (2 * buffer))
    row_path = visualize_row_mask(
        context_len=args.context,
        buffer_len=args.buffer,
        target_len=args.target,
        path=output_dir / "row_mask.png",
    )
    print(f"Saved: {row_path}")

    # Feature attention mask (dense)
    feature_path = visualize_dense_mask(
        seq_len=args.features,
        path=output_dir / "feature_mask.png",
    )
    print(f"Saved: {feature_path}")

    print("\nDone! Check the output files to verify the attention patterns.")
