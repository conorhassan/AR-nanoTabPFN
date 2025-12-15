"""Tests for the Triton shared-context attention kernel."""

import pytest
import torch

from autoregressive_nano_tabpfn.model.triton_kernels import (
    triton_available,
    triton_context_attention,
    pytorch_context_attention,
    pytorch_buffer_attention,
    merge_attention_outputs,
    hybrid_attention,
)


# Skip all tests if Triton is not available or no CUDA
requires_triton = pytest.mark.skipif(
    not triton_available() or not torch.cuda.is_available(),
    reason="Triton or CUDA not available",
)


class TestTritonContextAttention:
    """Test the Triton kernel against PyTorch reference."""

    @requires_triton
    @pytest.mark.parametrize("B", [1, 16, 64, 128])
    @pytest.mark.parametrize("H", [4, 8])
    @pytest.mark.parametrize("N_CTX", [32, 64, 128, 256])
    @pytest.mark.parametrize("D", [32, 64])
    def test_triton_matches_pytorch(self, B, H, N_CTX, D):
        """Verify Triton kernel output matches PyTorch reference."""
        torch.manual_seed(42)
        device = "cuda"
        dtype = torch.float16

        # Create inputs
        q = torch.randn(B, H, 1, D, device=device, dtype=dtype)
        k_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)
        v_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)

        # Triton kernel
        out_triton, lse_triton = triton_context_attention(q, k_ctx, v_ctx)

        # PyTorch reference
        out_pytorch, lse_pytorch = pytorch_context_attention(q, k_ctx, v_ctx)

        # Compare outputs
        torch.testing.assert_close(
            out_triton,
            out_pytorch,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Output mismatch for B={B}, H={H}, N_CTX={N_CTX}, D={D}",
        )

        # Compare LSE (more tolerance since it's derived)
        torch.testing.assert_close(
            lse_triton,
            lse_pytorch,
            atol=1e-1,
            rtol=1e-1,
            msg=f"LSE mismatch for B={B}, H={H}, N_CTX={N_CTX}, D={D}",
        )

    @requires_triton
    def test_non_divisible_batch_size(self):
        """Test kernel handles batch sizes not divisible by BLOCK_SIZE_B."""
        torch.manual_seed(42)
        device = "cuda"
        dtype = torch.float16

        # B=37 is not divisible by any typical block size
        B, H, N_CTX, D = 37, 4, 64, 32
        q = torch.randn(B, H, 1, D, device=device, dtype=dtype)
        k_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)
        v_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)

        out_triton, lse_triton = triton_context_attention(q, k_ctx, v_ctx)
        out_pytorch, lse_pytorch = pytorch_context_attention(q, k_ctx, v_ctx)

        torch.testing.assert_close(out_triton, out_pytorch, atol=1e-2, rtol=1e-2)

    @requires_triton
    def test_non_divisible_context_size(self):
        """Test kernel handles context sizes not divisible by BLOCK_SIZE_N."""
        torch.manual_seed(42)
        device = "cuda"
        dtype = torch.float16

        # N_CTX=100 is not divisible by typical block sizes
        B, H, N_CTX, D = 32, 4, 100, 32
        q = torch.randn(B, H, 1, D, device=device, dtype=dtype)
        k_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)
        v_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)

        out_triton, lse_triton = triton_context_attention(q, k_ctx, v_ctx)
        out_pytorch, lse_pytorch = pytorch_context_attention(q, k_ctx, v_ctx)

        torch.testing.assert_close(out_triton, out_pytorch, atol=1e-2, rtol=1e-2)


class TestMergeAttentionOutputs:
    """Test the LSE-based merging of attention outputs."""

    def test_merge_matches_full_attention(self):
        """Verify merged output matches computing attention over concatenated K/V."""
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32  # Use float32 for precision in this test

        B, H, Lq, D = 16, 4, 1, 32
        N_CTX, N_BUF = 64, 16

        q = torch.randn(B, H, Lq, D, device=device, dtype=dtype)
        k_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)
        v_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)
        k_buf = torch.randn(B, H, N_BUF, D, device=device, dtype=dtype)
        v_buf = torch.randn(B, H, N_BUF, D, device=device, dtype=dtype)

        # Compute separate attention outputs with LSE
        out_ctx, lse_ctx = pytorch_context_attention(q, k_ctx, v_ctx)
        out_buf, lse_buf = pytorch_buffer_attention(q, k_buf, v_buf)

        # Merge using LSE
        out_merged = merge_attention_outputs(out_ctx, lse_ctx, out_buf, lse_buf)

        # Ground truth: full attention over concatenated K/V
        scale = D**-0.5
        k_ctx_exp = k_ctx.unsqueeze(0).expand(B, -1, -1, -1)
        v_ctx_exp = v_ctx.unsqueeze(0).expand(B, -1, -1, -1)
        k_full = torch.cat([k_ctx_exp, k_buf], dim=2)  # [B, H, N_CTX+N_BUF, D]
        v_full = torch.cat([v_ctx_exp, v_buf], dim=2)

        scores_full = torch.matmul(q, k_full.transpose(-2, -1)) * scale
        attn_full = torch.softmax(scores_full, dim=-1)
        out_full = torch.matmul(attn_full, v_full)

        # Compare
        torch.testing.assert_close(
            out_merged, out_full, atol=1e-4, rtol=1e-4, msg="Merged output differs from full attention"
        )

    def test_merge_with_masked_buffer(self):
        """Test merge when buffer has causal masking."""
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        B, H, Lq, D = 8, 4, 1, 32
        N_CTX, N_BUF = 32, 8

        q = torch.randn(B, H, Lq, D, device=device, dtype=dtype)
        k_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)
        v_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)
        k_buf = torch.randn(B, H, N_BUF, D, device=device, dtype=dtype)
        v_buf = torch.randn(B, H, N_BUF, D, device=device, dtype=dtype)

        # Causal mask: query at position 0 can only see first 4 buffer positions
        buf_mask = torch.zeros(B, H, Lq, N_BUF, dtype=torch.bool, device=device)
        buf_mask[:, :, :, :4] = True  # Can see positions 0-3

        # Compute with mask
        out_ctx, lse_ctx = pytorch_context_attention(q, k_ctx, v_ctx)
        out_buf, lse_buf = pytorch_buffer_attention(q, k_buf, v_buf, mask=buf_mask)
        out_merged = merge_attention_outputs(out_ctx, lse_ctx, out_buf, lse_buf)

        # Ground truth with mask
        scale = D**-0.5
        k_ctx_exp = k_ctx.unsqueeze(0).expand(B, -1, -1, -1)
        v_ctx_exp = v_ctx.unsqueeze(0).expand(B, -1, -1, -1)
        k_full = torch.cat([k_ctx_exp, k_buf], dim=2)
        v_full = torch.cat([v_ctx_exp, v_buf], dim=2)

        scores_full = torch.matmul(q, k_full.transpose(-2, -1)) * scale
        # Apply mask to buffer portion
        full_mask = torch.ones(B, H, Lq, N_CTX + N_BUF, dtype=torch.bool, device=device)
        full_mask[:, :, :, N_CTX:] = buf_mask
        scores_full = scores_full.masked_fill(~full_mask, float("-inf"))

        attn_full = torch.softmax(scores_full, dim=-1)
        out_full = torch.matmul(attn_full, v_full)

        torch.testing.assert_close(out_merged, out_full, atol=1e-4, rtol=1e-4)


class TestHybridAttention:
    """Test the hybrid attention function."""

    @requires_triton
    def test_hybrid_matches_full_cuda(self):
        """Test hybrid attention on CUDA matches full attention."""
        torch.manual_seed(42)
        device = "cuda"
        dtype = torch.float16

        B, H, Lq, D = 64, 4, 1, 32
        N_CTX, N_BUF = 128, 32

        q = torch.randn(B, H, Lq, D, device=device, dtype=dtype)
        k_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)
        v_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)
        k_buf = torch.randn(B, H, N_BUF, D, device=device, dtype=dtype)
        v_buf = torch.randn(B, H, N_BUF, D, device=device, dtype=dtype)

        # Hybrid attention
        out_hybrid = hybrid_attention(q, k_ctx, v_ctx, k_buf, v_buf, use_triton=True)

        # Full attention ground truth
        scale = D**-0.5
        k_ctx_exp = k_ctx.unsqueeze(0).expand(B, -1, -1, -1)
        v_ctx_exp = v_ctx.unsqueeze(0).expand(B, -1, -1, -1)
        k_full = torch.cat([k_ctx_exp, k_buf], dim=2)
        v_full = torch.cat([v_ctx_exp, v_buf], dim=2)
        scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale
        attn = torch.softmax(scores.float(), dim=-1).to(dtype)
        out_full = torch.matmul(attn, v_full)

        torch.testing.assert_close(out_hybrid, out_full, atol=1e-2, rtol=1e-2)

    def test_hybrid_empty_buffer(self):
        """Test hybrid attention when buffer is empty."""
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        B, H, Lq, D = 32, 4, 1, 32
        N_CTX = 64

        q = torch.randn(B, H, Lq, D, device=device, dtype=dtype)
        k_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)
        v_ctx = torch.randn(H, N_CTX, D, device=device, dtype=dtype)
        k_buf = torch.empty(B, H, 0, D, device=device, dtype=dtype)
        v_buf = torch.empty(B, H, 0, D, device=device, dtype=dtype)

        # Should work without error
        out = hybrid_attention(q, k_ctx, v_ctx, k_buf, v_buf, use_triton=False)

        # Should match context-only attention
        out_ctx, _ = pytorch_context_attention(q, k_ctx, v_ctx)
        torch.testing.assert_close(out, out_ctx, atol=1e-5, rtol=1e-5)


class TestNumericalStability:
    """Test numerical stability edge cases."""

    def test_merge_with_dominant_source(self):
        """Test merge when one source has much higher attention scores."""
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        B, H, Lq, D = 8, 4, 1, 32

        # Context with high attention scores
        out_ctx = torch.randn(B, H, Lq, D, device=device)
        lse_ctx = torch.full((B, H, Lq), 10.0, device=device)  # High LSE

        # Buffer with low attention scores
        out_buf = torch.randn(B, H, Lq, D, device=device)
        lse_buf = torch.full((B, H, Lq), -10.0, device=device)  # Low LSE

        # Merge should heavily weight context
        out = merge_attention_outputs(out_ctx, lse_ctx, out_buf, lse_buf)

        # Output should be close to context output
        torch.testing.assert_close(out, out_ctx, atol=1e-4, rtol=1e-4)

    def test_merge_with_equal_sources(self):
        """Test merge when both sources have equal weight."""
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        B, H, Lq, D = 8, 4, 1, 32

        out_ctx = torch.randn(B, H, Lq, D, device=device)
        out_buf = torch.randn(B, H, Lq, D, device=device)
        lse_ctx = torch.full((B, H, Lq), 5.0, device=device)
        lse_buf = torch.full((B, H, Lq), 5.0, device=device)  # Equal LSE

        out = merge_attention_outputs(out_ctx, lse_ctx, out_buf, lse_buf)

        # Should be average of the two
        expected = (out_ctx + out_buf) / 2
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
