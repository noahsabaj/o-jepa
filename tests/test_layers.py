"""Tests for layer modules."""

import pytest
import torch
import torch.nn as nn

from src.layers.attention import MultiHeadAttention, CrossAttention
from src.layers.feedforward import SwiGLU, FeedForward
from src.layers.normalization import RMSNorm, get_norm_layer
from src.layers.positional import (
    RotaryPositionalEmbedding,
    apply_rotary_pos_emb,
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
)


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        attn = MultiHeadAttention(dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_forward_without_rope(self):
        """Test forward without RoPE."""
        attn = MultiHeadAttention(dim=64, num_heads=4, use_rope=False)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_forward_with_mask(self):
        """Test forward with attention mask."""
        attn = MultiHeadAttention(dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[:, 8:] = False  # Mask out second half
        out = attn(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_forward_with_4d_mask(self):
        """Test forward with 4D attention mask."""
        attn = MultiHeadAttention(dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        mask = torch.zeros(2, 1, 16, 16)  # Additive mask
        out = attn(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_forward_causal(self):
        """Test forward with causal masking."""
        attn = MultiHeadAttention(dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        out = attn(x, is_causal=True)
        assert out.shape == x.shape

    def test_invalid_dim_heads(self):
        """Test error when dim not divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadAttention(dim=63, num_heads=4)

    def test_with_bias(self):
        """Test with bias enabled."""
        attn = MultiHeadAttention(dim=64, num_heads=4, bias=True)
        assert attn.qkv.bias is not None
        assert attn.proj.bias is not None
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_dropout_training(self):
        """Test dropout in training mode."""
        attn = MultiHeadAttention(dim=64, num_heads=4, dropout=0.1)
        attn.train()
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_dropout_eval(self):
        """Test no dropout in eval mode."""
        attn = MultiHeadAttention(dim=64, num_heads=4, dropout=0.1)
        attn.eval()
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_extra_repr(self):
        """Test extra_repr method."""
        attn = MultiHeadAttention(dim=64, num_heads=4, use_rope=True)
        repr_str = attn.extra_repr()
        assert "dim=64" in repr_str
        assert "num_heads=4" in repr_str
        assert "rope=True" in repr_str

    def test_gradient_flow(self):
        """Test gradient flow."""
        attn = MultiHeadAttention(dim=64, num_heads=4)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestCrossAttention:
    """Tests for CrossAttention."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        attn = CrossAttention(dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        context = torch.randn(2, 32, 64)
        out = attn(x, context)
        assert out.shape == x.shape

    def test_forward_with_mask(self):
        """Test forward with context mask."""
        attn = CrossAttention(dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        context = torch.randn(2, 32, 64)
        mask = torch.ones(2, 32, dtype=torch.bool)
        mask[:, 16:] = False
        out = attn(x, context, context_mask=mask)
        assert out.shape == x.shape

    def test_invalid_dim_heads(self):
        """Test error when dim not divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible"):
            CrossAttention(dim=63, num_heads=4)

    def test_with_bias(self):
        """Test with bias enabled."""
        attn = CrossAttention(dim=64, num_heads=4, bias=True)
        assert attn.q_proj.bias is not None
        x = torch.randn(2, 16, 64)
        context = torch.randn(2, 32, 64)
        out = attn(x, context)
        assert out.shape == x.shape

    def test_extra_repr(self):
        """Test extra_repr method."""
        attn = CrossAttention(dim=64, num_heads=4)
        repr_str = attn.extra_repr()
        assert "dim=64" in repr_str
        assert "num_heads=4" in repr_str

    def test_gradient_flow(self):
        """Test gradient flow."""
        attn = CrossAttention(dim=64, num_heads=4)
        x = torch.randn(2, 16, 64, requires_grad=True)
        context = torch.randn(2, 32, 64, requires_grad=True)
        out = attn(x, context)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert context.grad is not None


class TestSwiGLU:
    """Tests for SwiGLU feedforward."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        ff = SwiGLU(dim=64)
        x = torch.randn(2, 16, 64)
        out = ff(x)
        assert out.shape == x.shape

    def test_custom_hidden_dim(self):
        """Test with custom hidden dimension."""
        ff = SwiGLU(dim=64, hidden_dim=128)
        assert ff.hidden_dim == 128
        x = torch.randn(2, 16, 64)
        out = ff(x)
        assert out.shape == x.shape

    def test_auto_hidden_dim(self):
        """Test auto-computed hidden dimension."""
        ff = SwiGLU(dim=64)
        # Should be multiple of 256
        assert ff.hidden_dim % 256 == 0

    def test_with_bias(self):
        """Test with bias enabled."""
        ff = SwiGLU(dim=64, bias=True)
        assert ff.w1.bias is not None
        x = torch.randn(2, 16, 64)
        out = ff(x)
        assert out.shape == x.shape

    def test_with_dropout(self):
        """Test with dropout."""
        ff = SwiGLU(dim=64, dropout=0.1)
        ff.train()
        x = torch.randn(2, 16, 64)
        out = ff(x)
        assert out.shape == x.shape

    def test_extra_repr(self):
        """Test extra_repr method."""
        ff = SwiGLU(dim=64, hidden_dim=128)
        repr_str = ff.extra_repr()
        assert "dim=64" in repr_str
        assert "hidden_dim=128" in repr_str

    def test_gradient_flow(self):
        """Test gradient flow."""
        ff = SwiGLU(dim=64)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = ff(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestFeedForward:
    """Tests for standard FeedForward."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        ff = FeedForward(dim=64)
        x = torch.randn(2, 16, 64)
        out = ff(x)
        assert out.shape == x.shape

    def test_custom_hidden_dim(self):
        """Test with custom hidden dimension."""
        ff = FeedForward(dim=64, hidden_dim=128)
        assert ff.hidden_dim == 128
        x = torch.randn(2, 16, 64)
        out = ff(x)
        assert out.shape == x.shape

    def test_default_hidden_dim(self):
        """Test default hidden dimension (4x)."""
        ff = FeedForward(dim=64)
        assert ff.hidden_dim == 256  # 4 * 64

    def test_with_bias(self):
        """Test with bias enabled."""
        ff = FeedForward(dim=64, bias=True)
        assert ff.w1.bias is not None
        x = torch.randn(2, 16, 64)
        out = ff(x)
        assert out.shape == x.shape

    def test_with_dropout(self):
        """Test with dropout."""
        ff = FeedForward(dim=64, dropout=0.1)
        ff.train()
        x = torch.randn(2, 16, 64)
        out = ff(x)
        assert out.shape == x.shape

    def test_extra_repr(self):
        """Test extra_repr method."""
        ff = FeedForward(dim=64, hidden_dim=128)
        repr_str = ff.extra_repr()
        assert "dim=64" in repr_str
        assert "hidden_dim=128" in repr_str

    def test_gradient_flow(self):
        """Test gradient flow."""
        ff = FeedForward(dim=64)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = ff(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization_effect(self):
        """Test that normalization reduces variance."""
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 16, 64) * 10  # Large values
        out = norm(x)
        # RMS of output should be more controlled
        assert out.std() < x.std()

    def test_learnable_weight(self):
        """Test learnable weight parameter."""
        norm = RMSNorm(dim=64)
        assert norm.weight.shape == (64,)
        assert norm.weight.requires_grad

    def test_extra_repr(self):
        """Test extra_repr method."""
        norm = RMSNorm(dim=64, eps=1e-5)
        repr_str = norm.extra_repr()
        assert "dim=64" in repr_str
        assert "eps=1e-05" in repr_str

    def test_gradient_flow(self):
        """Test gradient flow."""
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestGetNormLayer:
    """Tests for get_norm_layer factory."""

    def test_basic(self):
        """Test factory returns RMSNorm."""
        norm = get_norm_layer(dim=64)
        assert isinstance(norm, RMSNorm)


class TestRotaryPositionalEmbedding:
    """Tests for RotaryPositionalEmbedding."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        rope = RotaryPositionalEmbedding(dim=64, max_seq_len=512)
        cos, sin = rope(32)
        assert cos.shape == (32, 64)
        assert sin.shape == (32, 64)

    def test_caching(self):
        """Test that embeddings are cached."""
        rope = RotaryPositionalEmbedding(dim=64, max_seq_len=512)
        cos1, sin1 = rope(32)
        cos2, sin2 = rope(32)
        assert torch.equal(cos1, cos2)
        assert torch.equal(sin1, sin2)

    def test_different_lengths(self):
        """Test with different sequence lengths."""
        rope = RotaryPositionalEmbedding(dim=64, max_seq_len=512)
        cos1, sin1 = rope(32)
        cos2, sin2 = rope(64)
        assert cos1.shape == (32, 64)
        assert cos2.shape == (64, 64)


class TestApplyRotaryPosEmb:
    """Tests for apply_rotary_pos_emb function."""

    def test_basic_application(self):
        """Test basic RoPE application."""
        batch, heads, seq_len, head_dim = 2, 4, 16, 64
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        cos = torch.randn(seq_len, head_dim)
        sin = torch.randn(seq_len, head_dim)

        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)

        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_modifies_values(self):
        """Test that RoPE actually modifies the values."""
        batch, heads, seq_len, head_dim = 2, 4, 16, 64
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)

        rope = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=512)
        cos, sin = rope(seq_len)

        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)

        # Output should be different from input
        assert not torch.allclose(q, q_out)
        assert not torch.allclose(k, k_out)

    def test_odd_dim_error(self):
        """Test error with odd dimension."""
        with pytest.raises(ValueError, match="even"):
            RotaryPositionalEmbedding(dim=63, max_seq_len=512)

    def test_cache_rebuild_on_longer_seq(self):
        """Test cache rebuilds for longer sequences."""
        rope = RotaryPositionalEmbedding(dim=64, max_seq_len=32)

        # Request longer sequence than initial max
        cos, sin = rope(64)

        assert cos.shape == (64, 64)
        assert sin.shape == (64, 64)
        assert rope.max_seq_len >= 64


class TestLearnedPositionalEmbedding:
    """Tests for LearnedPositionalEmbedding."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        pos_emb = LearnedPositionalEmbedding(max_seq_len=512, dim=64)
        out = pos_emb(32)
        assert out.shape == (1, 32, 64)

    def test_learnable_parameters(self):
        """Test embedding is learnable."""
        pos_emb = LearnedPositionalEmbedding(max_seq_len=512, dim=64)
        assert pos_emb.embedding.requires_grad

    def test_exceed_max_seq_len_error(self):
        """Test error when exceeding max sequence length."""
        pos_emb = LearnedPositionalEmbedding(max_seq_len=32, dim=64)
        with pytest.raises(ValueError, match="exceeds max"):
            pos_emb(64)

    def test_different_lengths(self):
        """Test with different sequence lengths."""
        pos_emb = LearnedPositionalEmbedding(max_seq_len=512, dim=64)

        out1 = pos_emb(32)
        out2 = pos_emb(64)

        assert out1.shape == (1, 32, 64)
        assert out2.shape == (1, 64, 64)

    def test_gradient_flow(self):
        """Test gradient flow."""
        pos_emb = LearnedPositionalEmbedding(max_seq_len=512, dim=64)
        out = pos_emb(32)
        loss = out.sum()
        loss.backward()
        assert pos_emb.embedding.grad is not None


class TestSinusoidalPositionalEmbedding:
    """Tests for SinusoidalPositionalEmbedding."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        pos_emb = SinusoidalPositionalEmbedding(dim=64, max_seq_len=512)
        out = pos_emb(32)
        assert out.shape == (1, 32, 64)

    def test_non_learnable(self):
        """Test embeddings are not learnable."""
        pos_emb = SinusoidalPositionalEmbedding(dim=64, max_seq_len=512)
        # pe is a buffer, not parameter
        assert not pos_emb.pe.requires_grad

    def test_different_lengths(self):
        """Test with different sequence lengths."""
        pos_emb = SinusoidalPositionalEmbedding(dim=64, max_seq_len=512)

        out1 = pos_emb(32)
        out2 = pos_emb(64)

        assert out1.shape == (1, 32, 64)
        assert out2.shape == (1, 64, 64)

    def test_deterministic(self):
        """Test sinusoidal embeddings are deterministic."""
        pos_emb = SinusoidalPositionalEmbedding(dim=64, max_seq_len=512)

        out1 = pos_emb(32)
        out2 = pos_emb(32)

        assert torch.equal(out1, out2)

    def test_shape_attributes(self):
        """Test shape-related attributes."""
        pos_emb = SinusoidalPositionalEmbedding(dim=128, max_seq_len=1024)
        assert pos_emb.dim == 128
        assert pos_emb.max_seq_len == 1024
