"""Tests for ByteEncoder."""

import pytest
import torch

from src.byte_encoder import ByteEncoder, ByteEncoderConfig, TransformerBlock


class TestByteEncoderConfig:
    """Tests for ByteEncoderConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ByteEncoderConfig()
        assert config.vocab_size == 256
        assert config.hidden_dim == 512
        assert config.num_layers == 2
        assert config.num_heads == 8

    def test_custom_config(self):
        """Test custom configuration."""
        config = ByteEncoderConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
        )
        assert config.hidden_dim == 256
        assert config.num_layers == 4


class TestTransformerBlock:
    """Tests for TransformerBlock."""

    def test_forward_shape(self):
        """Test output shape."""
        block = TransformerBlock(dim=128, num_heads=4, mlp_dim=512)
        x = torch.randn(2, 64, 128)
        out = block(x)
        assert out.shape == x.shape

    def test_with_mask(self):
        """Test with attention mask."""
        block = TransformerBlock(dim=128, num_heads=4, mlp_dim=512)
        x = torch.randn(2, 64, 128)
        mask = torch.ones(2, 64, dtype=torch.bool)
        mask[:, 32:] = False
        out = block(x, mask)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        """Test that gradients flow through the block."""
        block = TransformerBlock(dim=128, num_heads=4, mlp_dim=512)
        x = torch.randn(2, 64, 128, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestByteEncoder:
    """Tests for ByteEncoder."""

    def test_forward_shape(self, byte_encoder, dummy_vision_bytes):
        """Test output shape."""
        out = byte_encoder(dummy_vision_bytes)
        batch_size, seq_len = dummy_vision_bytes.shape
        assert out.shape == (batch_size, seq_len, byte_encoder.config.hidden_dim)

    def test_byte_range(self, byte_encoder):
        """Test that encoder handles all byte values 0-255."""
        batch_size = 2
        seq_len = 256
        # Create tensor with all byte values
        byte_ids = torch.arange(256).unsqueeze(0).expand(batch_size, -1)
        out = byte_encoder(byte_ids)
        assert out.shape == (batch_size, seq_len, byte_encoder.config.hidden_dim)

    def test_with_mask(self, byte_encoder, dummy_vision_bytes, dummy_vision_mask):
        """Test with attention mask."""
        out = byte_encoder(dummy_vision_bytes, dummy_vision_mask)
        batch_size, seq_len = dummy_vision_bytes.shape
        assert out.shape == (batch_size, seq_len, byte_encoder.config.hidden_dim)

    def test_different_sequence_lengths(self, byte_encoder):
        """Test with different sequence lengths."""
        for seq_len in [64, 256, 1024]:
            byte_ids = torch.randint(0, 256, (2, seq_len), dtype=torch.long)
            out = byte_encoder(byte_ids)
            assert out.shape == (2, seq_len, byte_encoder.config.hidden_dim)

    def test_gradient_flow(self, byte_encoder, dummy_vision_bytes):
        """Test gradient flow through encoder."""
        out = byte_encoder(dummy_vision_bytes)
        loss = out.sum()
        loss.backward()
        # Check that embedding weights have gradients
        assert byte_encoder.byte_embed.weight.grad is not None

    def test_embedding_dimension(self, byte_encoder):
        """Test embedding table dimensions."""
        assert byte_encoder.byte_embed.num_embeddings == 256
        assert byte_encoder.byte_embed.embedding_dim == byte_encoder.config.hidden_dim

    def test_position_embedding(self, byte_encoder):
        """Test position embeddings are applied."""
        batch_size = 2
        seq_len = 100
        byte_ids = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)

        # Same byte at different positions should produce different outputs
        same_bytes = torch.zeros(batch_size, seq_len, dtype=torch.long)
        out = byte_encoder(same_bytes)

        # Different positions should have different representations
        # (due to position embeddings)
        pos_0 = out[:, 0, :]
        pos_1 = out[:, 1, :]
        assert not torch.allclose(pos_0, pos_1, atol=1e-6)

    def test_deterministic(self, byte_encoder, dummy_vision_bytes):
        """Test that encoder is deterministic in eval mode."""
        byte_encoder.eval()
        out1 = byte_encoder(dummy_vision_bytes)
        out2 = byte_encoder(dummy_vision_bytes)
        assert torch.allclose(out1, out2)
