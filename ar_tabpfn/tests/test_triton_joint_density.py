#!/usr/bin/env python3
"""
Standalone GPU test script for Triton joint density evaluation.

Tests the multi-query Triton kernel and teacher-forcing attention
used in evaluate_joint_density().

Run on a machine with CUDA:
    python scripts/test_triton_joint_density.py

Or in a Jupyter notebook / Google Colab:
    %run scripts/test_triton_joint_density.py
"""

import logging
import sys
import time
import torch

try:
    import torch._dynamo as dynamo
except Exception:  # pragma: no cover
    dynamo = None

# Prefer TF32 on supported GPUs
if torch.cuda.is_available():
    if hasattr(torch.backends.cuda, "matmul") and hasattr(
        torch.backends.cuda.matmul, "fp32_precision"
    ):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    if hasattr(torch.backends.cudnn, "conv") and hasattr(
        torch.backends.cudnn.conv, "fp32_precision"
    ):
        torch.backends.cudnn.conv.fp32_precision = "tf32"

# Check CUDA availability first
if not torch.cuda.is_available():
    print("CUDA not available. This script requires a GPU.")
    print("   Run on a machine with CUDA or try Google Colab.")
    sys.exit(1)

logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
if dynamo is not None:
    dynamo.config.recompile_limit = 512
    if hasattr(dynamo.config, "cache_size_limit"):
        dynamo.config.cache_size_limit = 512

print(f"CUDA available: {torch.cuda.get_device_name(0)}")
print(f"  PyTorch version: {torch.__version__}")

# Import Triton kernel functions
try:
    from ar_tabpfn.model.triton_kernels import (
        triton_available,
        triton_context_attention,
        pytorch_context_attention,
        pytorch_teacher_forcing_buffer_attention,
        hybrid_teacher_forcing_attention,
    )
    from ar_tabpfn.model import ARTabPFN, ARTabPFNPredictor, clear_mask_cache
except ImportError as e:
    print(f"Import error: {e}")
    print("   Make sure you're running from the nanoTabPFN directory")
    sys.exit(1)

if not triton_available():
    print("Triton not available. Install with: pip install triton")
    sys.exit(1)

print("Triton available")


def test_multiq_triton_vs_pytorch(B, H, Lq, N_CTX, D, dtype=torch.float32):
    """Compare multi-query Triton kernel output to PyTorch reference."""
    torch.manual_seed(42)
    device = "cuda"

    q = torch.randn(B, H, Lq, D, device=device, dtype=dtype)
    k_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)
    v_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)

    # Triton kernel
    out_triton, lse_triton = triton_context_attention(q, k_ctx, v_ctx)

    # PyTorch reference
    out_pytorch, lse_pytorch = pytorch_context_attention(q, k_ctx, v_ctx)

    # Compare - use 5e-3 tolerance (Triton vs PyTorch have numerical diffs due to
    # different accumulation order and online softmax vs standard softmax)
    out_close = torch.allclose(out_triton, out_pytorch, atol=5e-3, rtol=5e-3)
    lse_close = torch.allclose(lse_triton, lse_pytorch, atol=5e-3, rtol=5e-3)

    out_diff = (out_triton - out_pytorch).abs().max().item()
    lse_diff = (lse_triton - lse_pytorch).abs().max().item()

    return out_close and lse_close, out_diff, lse_diff


def test_hybrid_teacher_forcing(B, H, Nt, N_CTX, D, dtype=torch.float32):
    """Test hybrid teacher-forcing attention vs full attention ground truth."""
    torch.manual_seed(42)
    device = "cuda"

    Lq = 2 * Nt
    q = torch.randn(B, H, Lq, D, device=device, dtype=dtype)
    k_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)
    v_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)
    k_buf = torch.randn(B, H, Nt, D, device=device, dtype=dtype)
    v_buf = torch.randn(B, H, Nt, D, device=device, dtype=dtype)

    # Hybrid attention (Triton context + PyTorch buffer)
    out_hybrid = hybrid_teacher_forcing_attention(
        q, k_ctx, v_ctx, k_buf, v_buf, Nt, use_triton=True
    )

    # Ground truth: full attention with teacher-forcing mask
    scale = D**-0.5
    k_ctx_exp = k_ctx.unsqueeze(0).expand(B, -1, -1, -1)
    v_ctx_exp = v_ctx.unsqueeze(0).expand(B, -1, -1, -1)
    k_full = torch.cat([k_ctx_exp, k_buf], dim=2)
    v_full = torch.cat([v_ctx_exp, v_buf], dim=2)

    scores_full = torch.matmul(q, k_full.transpose(-2, -1)) * scale

    # Build teacher-forcing mask
    q_idx = torch.arange(Lq, device=device)
    kv_idx = torch.arange(Nt, device=device)
    is_buffer = q_idx < Nt
    buffer_mask = kv_idx[None, :] <= q_idx[:, None]
    target_mask = kv_idx[None, :] < (q_idx - Nt)[:, None]
    buf_mask = torch.where(is_buffer[:, None], buffer_mask, target_mask)

    ctx_mask = torch.ones(1, 1, Lq, N_CTX, device=device, dtype=torch.bool)
    full_mask = torch.cat([ctx_mask, buf_mask.unsqueeze(0).unsqueeze(0)], dim=-1)

    scores_full = scores_full.masked_fill(~full_mask, float("-inf"))
    attn_full = torch.softmax(scores_full, dim=-1)
    out_full = torch.matmul(attn_full, v_full)

    # Use 5e-3 tolerance (hybrid uses LSE merging which has numerical diffs)
    close = torch.allclose(out_hybrid, out_full, atol=5e-3, rtol=5e-3)
    diff = (out_hybrid - out_full).abs().max().item()

    return close, diff


def clear_layer_caches(model):
    """Remove KV cache tensors from layers."""
    for layer in model.backbone.layers:
        for name in (
            "k_cache",
            "v_cache",
            "k_ctx_cache",
            "v_ctx_cache",
            "k_buf_cache",
            "v_buf_cache",
        ):
            if hasattr(layer, name):
                delattr(layer, name)


def test_predictor_joint_density_equivalence(B=4, Nc=64, Nt=8, F=4, dtype=torch.float32):
    """Compare flex vs Triton evaluate_joint_density() outputs."""
    # Reset dynamo to avoid recompilation issues with different sizes
    if dynamo is not None:
        dynamo.reset()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = "cuda"

    model = ARTabPFN(
        num_features=F,
        d_model=64,
        n_heads=4,
        n_layers=2,
        buffer_size=8,
        num_components=3,
    ).to(device=device, dtype=dtype)
    model.eval()

    # Create test data with shared context
    x_ctx_base = torch.randn(1, Nc, F, device=device, dtype=dtype)
    y_ctx_base = torch.randn(1, Nc, device=device, dtype=dtype)
    x_context = x_ctx_base.expand(B, -1, -1).contiguous()
    y_context = y_ctx_base.expand(B, -1).contiguous()
    x_target = torch.randn(B, Nt, F, device=device, dtype=dtype)
    y_target = torch.randn(B, Nt, device=device, dtype=dtype)

    predictor_flex = ARTabPFNPredictor.from_trained_model(
        model, backend="flex_attention"
    )
    predictor_triton = ARTabPFNPredictor.from_trained_model(
        model, backend="triton_shared_context"
    )

    with torch.no_grad():
        log_density_flex = predictor_flex.evaluate_joint_density(
            x_context, y_context, x_target, y_target
        )
        log_density_triton = predictor_triton.evaluate_joint_density(
            x_context, y_context, x_target, y_target
        )

    diff = (log_density_triton - log_density_flex).abs().max().item()
    # Use 5e-3 tolerance - flex and Triton paths have different numerical characteristics
    close = torch.allclose(log_density_triton, log_density_flex, atol=5e-3, rtol=5e-3)

    return close, diff


def measure_joint_density_memory(predictor, x_context, y_context, x_target, y_target):
    """Measure peak allocated CUDA memory for evaluate_joint_density()."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    with torch.no_grad():
        predictor.evaluate_joint_density(x_context, y_context, x_target, y_target)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return max(peak - baseline, 0)


def main():
    print("\n" + "=" * 60)
    print("MULTI-QUERY TRITON KERNEL CORRECTNESS TESTS")
    print("=" * 60)

    # Test multi-query kernel (the core of joint density Triton path)
    multiq_configs = [
        (1, 4, 2, 32, 32),   # B, H, Lq, N_CTX, D
        (8, 4, 4, 64, 32),
        (16, 8, 8, 128, 64),
        (32, 4, 16, 256, 32),
        (4, 4, 10, 100, 32),  # Non-power-of-2
    ]

    all_passed = True
    for B, H, Lq, N_CTX, D in multiq_configs:
        ok, out_diff, lse_diff = test_multiq_triton_vs_pytorch(B, H, Lq, N_CTX, D)
        status = "PASS" if ok else "FAIL"
        print(
            f"{status} B={B:3d}, H={H}, Lq={Lq:2d}, N_CTX={N_CTX:3d}, D={D:2d} | "
            f"out_diff={out_diff:.2e}, lse_diff={lse_diff:.2e}"
        )
        if not ok:
            all_passed = False

    print("\n" + "=" * 60)
    print("HYBRID TEACHER-FORCING ATTENTION TESTS")
    print("=" * 60)

    # Test hybrid teacher-forcing attention (context + buffer with TF mask)
    tf_configs = [
        (2, 4, 4, 32, 32),   # B, H, Nt, N_CTX, D
        (8, 4, 8, 64, 32),
        (16, 8, 16, 128, 64),
        (4, 4, 1, 32, 32),   # Nt=1 edge case
    ]

    for B, H, Nt, N_CTX, D in tf_configs:
        ok, diff = test_hybrid_teacher_forcing(B, H, Nt, N_CTX, D)
        status = "PASS" if ok else "FAIL"
        print(
            f"{status} B={B:3d}, H={H}, Nt={Nt:2d}, N_CTX={N_CTX:3d}, D={D:2d} | "
            f"max_diff={diff:.2e}"
        )
        if not ok:
            all_passed = False

    print("\n" + "=" * 60)
    print("PREDICTOR JOINT DENSITY INTEGRATION TESTS")
    print("=" * 60)

    if dynamo is not None:
        dynamo.reset()

    # Single config matching test_triton_gpu.py pattern (avoids torch.compile issues)
    ok, diff = test_predictor_joint_density_equivalence(B=4, Nc=512, Nt=16)
    status = "PASS" if ok else "FAIL"
    print(f"{status} B=4, Nc=512, Nt=16 | max_diff={diff:.2e}")
    if not ok:
        all_passed = False

    print("\n" + "=" * 60)
    print("JOINT DENSITY PERFORMANCE BENCHMARKS")
    print("=" * 60)

    if dynamo is not None:
        dynamo.reset()
    clear_mask_cache()  # Clear cached masks from integration tests

    perf_dtype = torch.float16
    F = 4
    Nc = 512
    Nt = 32
    perf_B = [1, 8, 64]

    model = ARTabPFN(
        num_features=F,
        d_model=256,
        n_heads=8,
        n_layers=8,
        buffer_size=32,
        num_components=3,
    ).to(device="cuda", dtype=perf_dtype)
    model.eval()

    # Match test_triton_gpu.py pattern: create predictors fresh each iteration
    for B in perf_B:
        if dynamo is not None:
            dynamo.reset()

        x_ctx_base = torch.randn(1, Nc, F, device="cuda", dtype=perf_dtype)
        y_ctx_base = torch.randn(1, Nc, device="cuda", dtype=perf_dtype)
        x_context = x_ctx_base.expand(B, -1, -1).contiguous()
        y_context = y_ctx_base.expand(B, -1).contiguous()
        x_target = torch.randn(B, Nt, F, device="cuda", dtype=perf_dtype)
        y_target = torch.randn(B, Nt, device="cuda", dtype=perf_dtype)

        predictor_flex = ARTabPFNPredictor.from_trained_model(model, backend="flex_attention")
        predictor_triton = ARTabPFNPredictor.from_trained_model(
            model, backend="triton_shared_context"
        )

        # Warmup
        with torch.no_grad():
            predictor_flex.evaluate_joint_density(x_context, y_context, x_target, y_target)
            predictor_triton.evaluate_joint_density(x_context, y_context, x_target, y_target)
        torch.cuda.synchronize()

        # Benchmark flex
        clear_layer_caches(model)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        start = time.perf_counter()
        with torch.no_grad():
            predictor_flex.evaluate_joint_density(x_context, y_context, x_target, y_target)
        torch.cuda.synchronize()
        flex_ms = (time.perf_counter() - start) * 1000
        flex_peak = max(torch.cuda.max_memory_allocated() - baseline, 0)

        # Benchmark Triton
        clear_layer_caches(model)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        start = time.perf_counter()
        with torch.no_grad():
            predictor_triton.evaluate_joint_density(x_context, y_context, x_target, y_target)
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - start) * 1000
        triton_peak = max(torch.cuda.max_memory_allocated() - baseline, 0)

        speedup = flex_ms / triton_ms if triton_ms > 0 else 0
        mem_ratio = flex_peak / triton_peak if triton_peak > 0 else 0

        print(
            f"B={B:3d}, Nc={Nc}, Nt={Nt} | "
            f"flex: {flex_ms:.2f}ms ({flex_peak / (1024**2):.1f} MiB), "
            f"triton: {triton_ms:.2f}ms ({triton_peak / (1024**2):.1f} MiB), "
            f"Speedup: {speedup:.2f}x, Mem: {mem_ratio:.2f}x"
        )

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
