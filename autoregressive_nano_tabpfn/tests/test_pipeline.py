"""Minimal tests for data generation and model forward pass."""

import torch

from autoregressive_nano_tabpfn.data import DataAttr, TabularSampler
from autoregressive_nano_tabpfn.model import ARTabPFN, create_dense_mask, create_row_mask

CUDA_AVAILABLE = torch.cuda.is_available()


def test_data_generation():
    """Test that TabularSampler generates valid data."""
    sampler = TabularSampler(dim_x=[3, 5, 10], normalize_y=True, device="cpu")
    batch = sampler.generate_batch(
        batch_size=4,
        num_context=16,
        num_buffer=8,
        num_target=32,
    )

    # Check types
    assert isinstance(batch, DataAttr)
    assert all(
        isinstance(getattr(batch, k), torch.Tensor) for k in ["xc", "yc", "xb", "yb", "xt", "yt"]
    )

    # Check shapes
    B, Nc, D = batch.xc.shape
    assert batch.yc.shape == (B, Nc, 1)
    assert batch.xb.shape == (B, 8, D)
    assert batch.yb.shape == (B, 8, 1)
    assert batch.xt.shape == (B, 32, D)
    assert batch.yt.shape == (B, 32, 1)

    # Check normalization (y should be ~N(0,1))
    assert batch.yc.mean().abs() < 0.5
    assert 0.5 < batch.yc.std() < 2.0

    print(f"Data generation: B={B}, Nc={Nc}, D={D}")
    return batch


def test_model_forward():
    """Test ARTabPFN forward pass with generated data."""
    device = "cuda" if CUDA_AVAILABLE else "cpu"

    # Generate data
    sampler = TabularSampler(dim_x=5, normalize_y=True, device="cpu")
    batch = sampler.generate_batch(
        batch_size=4,
        num_context=16,
        num_buffer=8,
        num_target=32,
    )

    # Move to device
    batch = batch.to(device)

    # Create model
    model = ARTabPFN(
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        num_features=5,
        buffer_size=8,
        num_components=3,
    ).to(device)

    # Create masks
    Nc, Nb, Nt = batch.xc.size(1), batch.xb.size(1), batch.xt.size(1)
    num_rows = Nc + Nb + Nt

    mask_features = create_dense_mask(seq_len=1, device=device)
    mask_rows = create_row_mask(
        num_rows=num_rows,
        context_len=Nc,
        buffer_len=Nb,
        device=device,
    )

    # Forward pass
    with torch.no_grad():
        loss = model(
            x_context=batch.xc,
            y_context=batch.yc.squeeze(-1),
            x_buffer=batch.xb,
            y_buffer=batch.yb.squeeze(-1),
            x_target=batch.xt,
            mask_features=mask_features,
            mask_rows=mask_rows,
            y_target=batch.yt.squeeze(-1),
        )

    assert loss is not None
    assert not torch.isnan(loss)
    assert loss.ndim == 0  # scalar

    print(f"✓ Model forward: loss={loss.item():.4f}")
    return loss


def test_backward():
    """Test that gradients flow correctly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sampler = TabularSampler(dim_x=5, normalize_y=True, device="cpu")
    batch = sampler.generate_batch(
        batch_size=4,
        num_context=16,
        num_buffer=8,
        num_target=32,
    ).to(device)

    model = ARTabPFN(
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        num_features=5,
        buffer_size=8,
        num_components=3,
    ).to(device)

    Nc, Nb, Nt = batch.xc.size(1), batch.xb.size(1), batch.xt.size(1)
    mask_features = create_dense_mask(seq_len=1, device=device)
    mask_rows = create_row_mask(num_rows=Nc + Nb + Nt, context_len=Nc, buffer_len=Nb, device=device)

    loss = model(
        x_context=batch.xc,
        y_context=batch.yc.squeeze(-1),
        x_buffer=batch.xb,
        y_buffer=batch.yb.squeeze(-1),
        x_target=batch.xt,
        mask_features=mask_features,
        mask_rows=mask_rows,
        y_target=batch.yt.squeeze(-1),
    )

    loss.backward()

    # Check gradients exist
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0

    print(f"✓ Backward pass: grad_norm={grad_norm:.4f}")
    return grad_norm


def run_all_tests():
    """Run all tests."""
    print("Running pipeline tests...\n")
    test_data_generation()
    test_model_forward()
    test_backward()
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    run_all_tests()
