"""
Tests verifying inference methods match forward pass behavior.
"""

import torch
from torch.nn.attention.flex_attention import create_block_mask

from ar_tabpfn.model import ARTabPFN, ARTabPFNPredictor
from ar_tabpfn.model.attention import create_dense_mask


def create_inference_style_row_mask(
    context_len: int, buffer_len: int, target_len: int, device: str = "cpu"
):
    """
    Create row mask matching inference teacher forcing pattern.

    Structure: [context | buffer | target]
    - Context: dense self-attention
    - Buffer: attends to context + causal within buffer
    - Target_i: attends to context + buffer[0..i-1] (not buffer[i] or other targets)
    """
    total = context_len + buffer_len + target_len
    ctx_end = context_len
    buf_start = context_len
    buf_end = context_len + buffer_len
    tgt_start = buf_end

    def mask_mod(b, h, q_idx, kv_idx):
        # Context queries: attend to all context
        q_is_ctx = q_idx < ctx_end
        kv_is_ctx = kv_idx < ctx_end
        ctx_pattern = q_is_ctx & kv_is_ctx

        # Buffer queries: attend to context + causal buffer
        q_is_buf = (q_idx >= buf_start) & (q_idx < buf_end)
        buf_to_ctx = q_is_buf & kv_is_ctx
        kv_is_buf = (kv_idx >= buf_start) & (kv_idx < buf_end)
        buf_causal = q_is_buf & kv_is_buf & (kv_idx <= q_idx)

        # Target queries: attend to context + buffer[0..i-1]
        q_is_tgt = q_idx >= tgt_start
        tgt_to_ctx = q_is_tgt & kv_is_ctx
        tgt_idx = q_idx - tgt_start  # which target (0..Nt-1)
        buf_idx = kv_idx - buf_start  # which buffer (0..Nb-1)
        tgt_to_buf = q_is_tgt & kv_is_buf & (buf_idx < tgt_idx)

        return ctx_pattern | buf_to_ctx | buf_causal | tgt_to_ctx | tgt_to_buf

    return create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=total, KV_LEN=total, device=device
    )


def test_context_prefill_matches_forward():
    """Verify prefill_context produces same representations as forward (context only)."""
    torch.manual_seed(42)

    model = ARTabPFN(
        num_features=4,
        d_model=32,
        n_heads=2,
        n_layers=2,
        buffer_size=4,
        num_components=3,
    )
    model.eval()

    predictor = ARTabPFNPredictor.from_trained_model(model)

    B, Nc, F = 2, 8, 4
    x_ctx = torch.randn(B, Nc, F)
    y_ctx = torch.randn(B, Nc)

    # Forward pass with just context (empty buffer/target, or minimal)
    ctx_emb = model.embedder.embed_context(x_ctx, y_ctx)
    ctx_emb = ctx_emb.unsqueeze(2)  # [B, Nc, 1, D]

    feature_mask = create_dense_mask(seq_len=1, device="cpu")
    row_mask = create_dense_mask(seq_len=Nc, device="cpu")

    z_forward, kv_cache = model.backbone(ctx_emb, feature_mask, row_mask)

    # Inference prefill
    predictor.init_kv_cache(B, Nc, device="cpu", dtype=torch.float32)
    predictor.prefill_context(x_ctx, y_ctx)

    # Compare cached K,V to forward's K,V
    for i, layer in enumerate(model.backbone.layers):
        k_cached = layer.k_cache[:, :, :Nc, :]
        v_cached = layer.v_cache[:, :, :Nc, :]

        k_forward, v_forward = kv_cache[i]
        assert k_cached.shape == (B, 2, Nc, 16), f"Layer {i}: K shape mismatch"
        assert v_cached.shape == (B, 2, Nc, 16), f"Layer {i}: V shape mismatch"
        torch.testing.assert_close(k_cached, k_forward, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(v_cached, v_forward, rtol=1e-5, atol=1e-5)

    print("test_context_prefill_matches_forward PASSED")


def test_teacher_forcing_matches_forward():
    """Verify evaluate_joint_density matches forward with same data/mask."""
    torch.manual_seed(42)

    model = ARTabPFN(
        num_features=4,
        d_model=32,
        n_heads=2,
        n_layers=2,
        buffer_size=4,
        num_components=3,
    )
    model.eval()

    predictor = ARTabPFNPredictor.from_trained_model(model)

    B, Nc, Nt, F = 2, 8, 4, 4
    x_ctx = torch.randn(B, Nc, F)
    y_ctx = torch.randn(B, Nc)
    x_tgt = torch.randn(B, Nt, F)
    y_tgt = torch.randn(B, Nt)

    # Forward pass: set buffer = target data to match inference
    feature_mask = create_dense_mask(seq_len=1, device="cpu")
    row_mask = create_inference_style_row_mask(Nc, Nt, Nt, device="cpu")

    loss_forward = model(
        x_context=x_ctx,
        y_context=y_ctx,
        x_buffer=x_tgt,  # buffer = target data
        y_buffer=y_tgt,
        x_target=x_tgt,
        y_target=y_tgt,
        mask_features=feature_mask,
        mask_rows=row_mask,
    )

    # Inference: evaluate_joint_density
    log_density = predictor.evaluate_joint_density(x_ctx, y_ctx, x_tgt, y_tgt)
    loss_inference = -log_density.mean()

    print(f"Forward loss: {loss_forward.item():.4f}")
    print(f"Inference loss: {loss_inference.item():.4f}")

    # They should be close (not exact due to different code paths)
    assert torch.isfinite(loss_forward), "Forward loss is not finite"
    assert torch.isfinite(loss_inference), "Inference loss is not finite"

    # Check relative difference
    rel_diff = abs(loss_forward - loss_inference) / max(abs(loss_forward), 1e-6)
    print(f"Relative difference: {rel_diff.item():.6f}")

    # They should match closely if mask is correct
    assert rel_diff < 0.01, f"Losses differ too much: {rel_diff.item():.6f}"

    print("test_teacher_forcing_matches_forward PASSED")


def test_first_target_no_buffer():
    """First target should only see context (no buffer dependency)."""
    torch.manual_seed(42)

    model = ARTabPFN(
        num_features=4,
        d_model=32,
        n_heads=2,
        n_layers=2,
        buffer_size=4,
        num_components=3,
    )
    model.eval()

    predictor = ARTabPFNPredictor.from_trained_model(model)

    B, Nc, F = 2, 8, 4
    x_ctx = torch.randn(B, Nc, F)
    y_ctx = torch.randn(B, Nc)
    x_tgt = torch.randn(B, 1, F)  # single target
    y_tgt = torch.randn(B, 1)

    # With 1 target, Target_0 attends only to context (buffer[0..−1] = nothing)
    log_density = predictor.evaluate_joint_density(x_ctx, y_ctx, x_tgt, y_tgt)

    assert log_density.shape == (B, 1)
    assert torch.isfinite(log_density).all()

    print(f"Single target log-density: {log_density}")
    print("test_first_target_no_buffer PASSED")


def test_first_sample_matches_forward():
    """First sample's representation should match forward (only sees context)."""
    torch.manual_seed(42)

    model = ARTabPFN(
        num_features=4,
        d_model=32,
        n_heads=2,
        n_layers=2,
        buffer_size=4,
        num_components=3,
    )
    model.eval()

    B, Nc, Nt, F = 2, 8, 4, 4
    x_ctx = torch.randn(B, Nc, F)
    y_ctx = torch.randn(B, Nc)
    x_tgt = torch.randn(B, Nt, F)

    # Forward: [context, empty buffer, single target]
    # Target[0] only sees context (no buffer)
    ctx_emb = model.embedder.embed_context(x_ctx, y_ctx)
    tgt_emb = model.embedder.embed_target(x_tgt[:, :1])  # first target only

    embeddings = torch.cat([ctx_emb, tgt_emb], dim=1).unsqueeze(2)

    # Mask: context dense, target sees only context
    total = Nc + 1

    def mask_mod(b, h, q, k):
        q_is_ctx = q < Nc
        k_is_ctx = k < Nc
        ctx_dense = q_is_ctx & k_is_ctx
        tgt_to_ctx = (q >= Nc) & k_is_ctx
        return ctx_dense | tgt_to_ctx

    feature_mask = create_dense_mask(seq_len=1, device="cpu")
    row_mask = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=total, KV_LEN=total, device="cpu"
    )

    z_forward, _ = model.backbone(embeddings, feature_mask, row_mask)
    z_target_fwd = z_forward[:, Nc:, 0, :]  # [B, 1, D]

    # Inference: decode first target (no buffer)
    predictor = ARTabPFNPredictor.from_trained_model(model)
    predictor.init_kv_cache(B, Nc + 1, device="cpu", dtype=torch.float32)
    predictor.prefill_context(x_ctx, y_ctx)

    # Decode just the first target
    tgt_emb_inf = model.embedder.embed_target(x_tgt[:, :1])
    z_inf = predictor.transformer_decode(tgt_emb_inf, commit=0)  # [B, 1, D]

    # Compare representations
    diff = (z_target_fwd.squeeze(2) - z_inf).abs().max().item()
    print(f"First target representation diff: {diff:.10f}")
    assert diff < 1e-5, f"Representations differ: {diff}"

    print("test_first_sample_matches_forward PASSED")


def test_sample_sequence_shape_and_finiteness():
    """Verify sample_sequence produces correct shape and finite values."""
    torch.manual_seed(42)

    model = ARTabPFN(
        num_features=4,
        d_model=32,
        n_heads=2,
        n_layers=2,
        buffer_size=4,
        num_components=3,
    )
    model.eval()

    predictor = ARTabPFNPredictor.from_trained_model(model)

    B, Nc, Nt, F = 2, 8, 4, 4
    x_ctx = torch.randn(B, Nc, F)
    y_ctx = torch.randn(B, Nc)
    x_tgt = torch.randn(B, Nt, F)

    predictions = predictor.sample_sequence(x_ctx, y_ctx, x_tgt)

    assert predictions.shape == (B, Nt), f"Wrong shape: {predictions.shape}"
    assert torch.isfinite(predictions).all(), "Predictions contain non-finite values"

    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    print("test_sample_sequence_shape_and_finiteness PASSED")


if __name__ == "__main__":
    test_context_prefill_matches_forward()
    print()
    test_teacher_forcing_matches_forward()
    print()
    test_first_target_no_buffer()
    print()
    test_first_sample_matches_forward()
    print()
    test_sample_sequence_shape_and_finiteness()
    print()
    print("All tests passed!")
