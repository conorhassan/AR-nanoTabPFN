#!/usr/bin/env python3
"""
Standalone GPU test script for the Triton shared-context attention kernel.

Run on a machine with CUDA:
    python scripts/test_triton_gpu.py

Or in a Jupyter notebook / Google Colab:
    %run scripts/test_triton_gpu.py
"""

import logging
import sys
import torch

try:
    import torch._dynamo as dynamo
except Exception:  # pragma: no cover
    dynamo = None

# Prefer TF32 on supported GPUs to align with PyTorch 2.9 recommendations.
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
    print("❌ CUDA not available. This script requires a GPU.")
    print("   Run on a machine with CUDA or try Google Colab.")
    sys.exit(1)

logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
if dynamo is not None:
    dynamo.config.recompile_limit = 512
    if hasattr(dynamo.config, "cache_size_limit"):
        dynamo.config.cache_size_limit = 512

print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
print(f"  PyTorch version: {torch.__version__}")

# Import Triton kernel functions
try:
    from ar_tabpfn.model.triton_kernels import (
        triton_available,
        triton_context_attention,
        pytorch_context_attention,
        pytorch_buffer_attention,
        merge_attention_outputs,
        hybrid_attention,
    )
    from ar_tabpfn.model import ARTabPFN, ARTabPFNPredictor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure you're running from the nanoTabPFN directory")
    sys.exit(1)

if not triton_available():
    print("❌ Triton not available. Install with: pip install triton")
    sys.exit(1)

print("✓ Triton available")


def test_triton_vs_pytorch(B, H, N_CTX, D, dtype=torch.float16):
    """Compare Triton kernel output to PyTorch reference."""
    torch.manual_seed(42)
    device = "cuda"

    q = torch.randn(B, H, 1, D, device=device, dtype=dtype)
    k_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)
    v_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)

    # Triton kernel
    out_triton, lse_triton = triton_context_attention(q, k_ctx, v_ctx)

    # PyTorch reference
    out_pytorch, lse_pytorch = pytorch_context_attention(q, k_ctx, v_ctx)

    # Compare
    out_close = torch.allclose(out_triton, out_pytorch, atol=1e-2, rtol=1e-2)
    lse_close = torch.allclose(lse_triton, lse_pytorch, atol=1e-1, rtol=1e-1)

    out_diff = (out_triton - out_pytorch).abs().max().item()
    lse_diff = (lse_triton - lse_pytorch).abs().max().item()

    return out_close, lse_close, out_diff, lse_diff


def test_hybrid_attention(B, H, N_CTX, N_BUF, D, dtype=torch.float16):
    """Test hybrid attention (Triton context + PyTorch buffer) vs full attention."""
    torch.manual_seed(42)
    device = "cuda"

    q = torch.randn(B, H, 1, D, device=device, dtype=dtype)
    k_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)
    v_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)
    k_buf = torch.randn(B, H, N_BUF, D, device=device, dtype=dtype)
    v_buf = torch.randn(B, H, N_BUF, D, device=device, dtype=dtype)

    # Hybrid attention
    out_hybrid = hybrid_attention(q, k_ctx, v_ctx, k_buf, v_buf, use_triton=True)

    # Ground truth: full attention over concatenated K/V
    scale = D**-0.5
    k_ctx_exp = k_ctx.unsqueeze(0).expand(B, -1, -1, -1)
    v_ctx_exp = v_ctx.unsqueeze(0).expand(B, -1, -1, -1)
    k_full = torch.cat([k_ctx_exp, k_buf], dim=2)
    v_full = torch.cat([v_ctx_exp, v_buf], dim=2)
    scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale
    attn = torch.softmax(scores.float(), dim=-1).to(dtype)
    out_full = torch.matmul(attn, v_full)

    close = torch.allclose(out_hybrid, out_full, atol=1e-2, rtol=1e-2)
    diff = (out_hybrid - out_full).abs().max().item()

    return close, diff


def run_fixed_buffer_decode(
    predictor,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_target: torch.Tensor,
    fixed_prev_y: torch.Tensor,
):
    """Decode with fixed buffer values to avoid sampling nondeterminism."""
    B, Nc, _ = x_context.shape
    Nt = x_target.shape[1]
    device, dtype = x_context.device, x_context.dtype

    if predictor.backend == "triton":
        max_buf = Nt
        predictor.init_kv_cache(B, max_buf, device, dtype)
    else:
        max_seq = Nc + Nt
        predictor.init_kv_cache(B, max_seq, device, dtype)
    predictor.prefill_context(x_context, y_context)

    outputs = []
    for t in range(Nt):
        x_t = x_target[:, t : t + 1, :]
        if t == 0:
            embedding = predictor.embedder.embed_target(x_t)
            commit = 0
        else:
            prev_x = x_target[:, t - 1 : t, :]
            prev_y = fixed_prev_y[:, t - 1 : t]
            buffer_emb = predictor.embedder.embed_buffer(prev_x, prev_y)
            ar_idx = t % predictor.ar_tokens.shape[0]
            buffer_emb = buffer_emb + predictor.ar_tokens[ar_idx]
            target_emb = predictor.embedder.embed_target(x_t)
            embedding = torch.cat([buffer_emb, target_emb], dim=1)
            commit = 1

        z = predictor.transformer_decode(embedding, commit=commit)
        outputs.append(z[:, -1:, :])

    return torch.cat(outputs, dim=1)


def clear_layer_caches(model):
    """Remove KV cache tensors from layers to avoid cross-benchmark leakage."""
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


def measure_peak_memory(
    predictor,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_target: torch.Tensor,
    fixed_prev_y: torch.Tensor,
):
    """Measure peak allocated CUDA memory for a fixed decode path."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    run_fixed_buffer_decode(predictor, x_context, y_context, x_target, fixed_prev_y)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return max(peak - baseline, 0)


def test_predictor_backend_equivalence(
    B=4, Nc=16, Nt=6, F=4, dtype=torch.float32
):
    """Compare flex vs Triton decode outputs with shared context and fixed buffers."""
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

    x_ctx_base = torch.randn(1, Nc, F, device=device, dtype=dtype)
    y_ctx_base = torch.randn(1, Nc, device=device, dtype=dtype)
    x_context = x_ctx_base.expand(B, -1, -1).contiguous()
    y_context = y_ctx_base.expand(B, -1).contiguous()
    x_target = torch.randn(B, Nt, F, device=device, dtype=dtype)
    fixed_prev_y = torch.randn(B, Nt, device=device, dtype=dtype)

    predictor_flex = ARTabPFNPredictor.from_trained_model(
        model, backend="flex_attention"
    )
    predictor_triton = ARTabPFNPredictor.from_trained_model(
        model, backend="triton_shared_context"
    )

    z_flex = run_fixed_buffer_decode(
        predictor_flex, x_context, y_context, x_target, fixed_prev_y
    )
    z_triton = run_fixed_buffer_decode(
        predictor_triton, x_context, y_context, x_target, fixed_prev_y
    )

    diff = (z_triton - z_flex).abs().max().item()
    close = torch.allclose(z_triton, z_flex, atol=1e-3, rtol=1e-3)

    clear_layer_caches(model)
    torch.cuda.empty_cache()
    peak_flex = measure_peak_memory(
        predictor_flex, x_context, y_context, x_target, fixed_prev_y
    )
    clear_layer_caches(model)
    torch.cuda.empty_cache()
    peak_triton = measure_peak_memory(
        predictor_triton, x_context, y_context, x_target, fixed_prev_y
    )
    return close, diff, peak_flex, peak_triton


def benchmark_kernel(B, H, N_CTX, D, num_iters=100, warmup=10):
    """Benchmark Triton kernel vs PyTorch reference, including peak memory."""
    import time

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float16

    q = torch.randn(B, H, 1, D, device=device, dtype=dtype)
    k_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)
    v_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)

    # Warmup
    for _ in range(warmup):
        triton_context_attention(q, k_ctx, v_ctx)
        pytorch_context_attention(q, k_ctx, v_ctx)
    torch.cuda.synchronize()

    # Benchmark Triton (time + peak memory)
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    start = time.perf_counter()
    for _ in range(num_iters):
        triton_context_attention(q, k_ctx, v_ctx)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / num_iters * 1000  # ms
    triton_peak = max(torch.cuda.max_memory_allocated() - baseline, 0)

    # Benchmark PyTorch (time + peak memory)
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    start = time.perf_counter()
    for _ in range(num_iters):
        pytorch_context_attention(q, k_ctx, v_ctx)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / num_iters * 1000  # ms
    pytorch_peak = max(torch.cuda.max_memory_allocated() - baseline, 0)

    return triton_time, pytorch_time, triton_peak, pytorch_peak


def main():
    import time

    print("\n" + "=" * 60)
    print("TRITON KERNEL CORRECTNESS TESTS")
    print("=" * 60)
    correctness_dtype = torch.float32
    print(f"Correctness test dtype: {correctness_dtype}")

    test_configs = [
        (1, 4, 32, 32),
        (16, 4, 64, 32),
        (64, 4, 128, 32),
        (128, 8, 256, 64),
        (256, 4, 100, 32),  # Non-power-of-2 context
        (37, 4, 64, 32),  # Non-power-of-2 batch
    ]

    all_passed = True
    for B, H, N_CTX, D in test_configs:
        out_ok, lse_ok, out_diff, lse_diff = test_triton_vs_pytorch(
            B, H, N_CTX, D, dtype=correctness_dtype
        )
        status = "✓" if (out_ok and lse_ok) else "✗"
        print(
            f"{status} B={B:3d}, H={H}, N_CTX={N_CTX:3d}, D={D:2d} | "
            f"out_diff={out_diff:.2e}, lse_diff={lse_diff:.2e}"
        )
        if not (out_ok and lse_ok):
            all_passed = False

    print("\n" + "=" * 60)
    print("HYBRID ATTENTION TESTS")
    print("=" * 60)

    hybrid_configs = [
        (64, 4, 128, 32, 32),
        (128, 4, 100, 64, 32),
        (256, 8, 256, 128, 64),
    ]

    for B, H, N_CTX, N_BUF, D in hybrid_configs:
        close, diff = test_hybrid_attention(B, H, N_CTX, N_BUF, D, dtype=correctness_dtype)
        status = "✓" if close else "✗"
        print(
            f"{status} B={B:3d}, H={H}, N_CTX={N_CTX:3d}, N_BUF={N_BUF:3d}, D={D:2d} | "
            f"max_diff={diff:.2e}"
        )
        if not close:
            all_passed = False

    print("\n" + "=" * 60)
    print("PREDICTOR INTEGRATION TESTS")
    print("=" * 60)

    if dynamo is not None:
        dynamo.reset()

    pred_ok, pred_diff, peak_flex, peak_triton = test_predictor_backend_equivalence(
        Nc=512
    )
    status = "✓" if pred_ok else "✗"
    print(f"{status} shared-context decode | max_diff={pred_diff:.2e}")
    if peak_triton > 0:
        ratio = peak_flex / peak_triton
    else:
        ratio = 0.0
    print(
        "Peak VRAM (alloc): "
        f"flex={peak_flex / (1024**2):.1f} MiB, "
        f"triton={peak_triton / (1024**2):.1f} MiB, "
        f"ratio={ratio:.2f}x"
    )
    if not pred_ok:
        all_passed = False

    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 60)

    bench_configs = [
        (1, 8, 512, 64),
        (8, 8, 512, 64),
        (64, 8, 512, 64),
    ]

    for B, H, N_CTX, D in bench_configs:
        triton_ms, pytorch_ms, triton_peak, pytorch_peak = benchmark_kernel(
            B, H, N_CTX, D
        )
        speedup = pytorch_ms / triton_ms if triton_ms > 0 else 0
        if triton_peak >= 0.1 * 1024**2:
            mem_ratio = pytorch_peak / triton_peak
            mem_label = f"{mem_ratio:.2f}x"
        else:
            mem_label = "n/a"
        print(
            f"B={B:3d}, H={H}, N_CTX={N_CTX:3d}, D={D:2d} | "
            f"Triton: {triton_ms:.3f}ms ({triton_peak / (1024**2):.1f} MiB), "
            f"PyTorch: {pytorch_ms:.3f}ms ({pytorch_peak / (1024**2):.1f} MiB), "
            f"Speedup: {speedup:.2f}x, Mem: {mem_label}"
        )

    print("\n" + "=" * 60)
    print("PREDICTOR PERFORMANCE (SHARED CONTEXT)")
    print("=" * 60)

    if dynamo is not None:
        dynamo.reset()

    perf_dtype = torch.float16
    perf_B = [1, 8, 64]
    Nc = 512
    Nt = 32
    F = 4

    model = ARTabPFN(
        num_features=F,
        d_model=256,
        n_heads=8,
        n_layers=8,
        buffer_size=32,
        num_components=3,
    ).to(device="cuda", dtype=perf_dtype)
    model.eval()

    for B in perf_B:
        if dynamo is not None:
            dynamo.reset()
        x_ctx_base = torch.randn(1, Nc, F, device="cuda", dtype=perf_dtype)
        y_ctx_base = torch.randn(1, Nc, device="cuda", dtype=perf_dtype)
        x_context = x_ctx_base.expand(B, -1, -1).contiguous()
        y_context = y_ctx_base.expand(B, -1).contiguous()
        x_target = torch.randn(B, Nt, F, device="cuda", dtype=perf_dtype)
        fixed_prev_y = torch.randn(B, Nt, device="cuda", dtype=perf_dtype)

        predictor_flex = ARTabPFNPredictor.from_trained_model(
            model, backend="flex_attention"
        )
        predictor_triton = ARTabPFNPredictor.from_trained_model(
            model, backend="triton_shared_context"
        )

        # Warmup (compile + cache allocation)
        run_fixed_buffer_decode(
            predictor_flex, x_context, y_context, x_target, fixed_prev_y
        )
        run_fixed_buffer_decode(
            predictor_triton, x_context, y_context, x_target, fixed_prev_y
        )
        torch.cuda.synchronize()

        clear_layer_caches(model)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        start = time.perf_counter()
        run_fixed_buffer_decode(
            predictor_flex, x_context, y_context, x_target, fixed_prev_y
        )
        torch.cuda.synchronize()
        flex_ms = (time.perf_counter() - start) * 1000
        flex_peak = max(torch.cuda.max_memory_allocated() - baseline, 0)

        clear_layer_caches(model)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        start = time.perf_counter()
        run_fixed_buffer_decode(
            predictor_triton, x_context, y_context, x_target, fixed_prev_y
        )
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - start) * 1000
        triton_peak = max(torch.cuda.max_memory_allocated() - baseline, 0)

        speedup = flex_ms / triton_ms if triton_ms > 0 else 0
        mem_ratio = (flex_peak / triton_peak) if triton_peak > 0 else 0
        print(
            f"B={B:3d}, Nc={Nc}, Nt={Nt} | "
            f"flex: {flex_ms:.3f}ms ({flex_peak / (1024**2):.1f} MiB), "
            f"triton: {triton_ms:.3f}ms ({triton_peak / (1024**2):.1f} MiB), "
            f"Speedup: {speedup:.2f}x, Mem: {mem_ratio:.2f}x"
        )

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
