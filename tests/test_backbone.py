"""Tests for SharedBackbone and BackboneTransformerBlock."""

import pytest
import torch

from src.backbone import (
    SharedBackbone,
    BackboneTransformerBlock,
)
from src.config import BackboneConfig


class TestBackboneConfig:
    """Tests for BackboneConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BackboneConfig()
        assert config.hidden_dim == 512
        assert config.num_layers == 6
        assert config.num_heads == 8
        assert "vision" in config.modalities
        assert "text" in config.modalities

    def test_custom_config(self):
        """Test custom configuration."""
        config = BackboneConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            modalities=("vision", "audio"),
        )
        assert config.hidden_dim == 256
        assert config.num_layers == 4
        assert config.modalities == ("vision", "audio")


class TestBackboneTransformerBlock:
    """Tests for BackboneTransformerBlock."""

    def test_forward_basic(self, batch_size, hidden_dim):
        """Test basic forward pass."""
        block = BackboneTransformerBlock(
            dim=hidden_dim,
            num_heads=4,
            mlp_ratio=4.0,
        )

        x = torch.randn(batch_size, 64, hidden_dim)
        out = block(x)
        assert out.shape == x.shape

    def test_with_mask(self, batch_size, hidden_dim):
        """Test forward with attention mask."""
        block = BackboneTransformerBlock(
            dim=hidden_dim,
            num_heads=4,
            mlp_ratio=4.0,
        )

        x = torch.randn(batch_size, 64, hidden_dim)
        mask = torch.ones(batch_size, 64, dtype=torch.bool)

        out = block(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_with_dropout(self, batch_size, hidden_dim):
        """Test with dropout."""
        block = BackboneTransformerBlock(
            dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
        )
        block.train()

        x = torch.randn(batch_size, 64, hidden_dim)
        out = block(x)
        assert out.shape == x.shape

    def test_with_bias(self, batch_size, hidden_dim):
        """Test with bias enabled."""
        block = BackboneTransformerBlock(
            dim=hidden_dim,
            num_heads=4,
            use_bias=True,
        )

        x = torch.randn(batch_size, 64, hidden_dim)
        out = block(x)
        assert out.shape == x.shape

    def test_gradient_flow(self, batch_size, hidden_dim):
        """Test gradient flow."""
        block = BackboneTransformerBlock(
            dim=hidden_dim,
            num_heads=4,
        )

        x = torch.randn(batch_size, 64, hidden_dim, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestSharedBackbone:
    """Tests for SharedBackbone."""

    def test_forward_vision(self, batch_size, hidden_dim):
        """Test forward pass with vision modality."""
        config = BackboneConfig(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
        )
        backbone = SharedBackbone(config)

        embeddings = torch.randn(batch_size, 64, hidden_dim)
        sequence, pooled = backbone(embeddings, modality="vision")

        assert sequence.shape == embeddings.shape
        assert pooled.shape == (batch_size, hidden_dim)

    def test_forward_text(self, batch_size, hidden_dim):
        """Test forward pass with text modality."""
        config = BackboneConfig(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
        )
        backbone = SharedBackbone(config)

        embeddings = torch.randn(batch_size, 32, hidden_dim)
        sequence, pooled = backbone(embeddings, modality="text")

        assert sequence.shape == embeddings.shape
        assert pooled.shape == (batch_size, hidden_dim)

    def test_forward_audio(self, batch_size, hidden_dim):
        """Test forward pass with audio modality."""
        config = BackboneConfig(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            modalities=("vision", "text", "audio"),
        )
        backbone = SharedBackbone(config)

        embeddings = torch.randn(batch_size, 128, hidden_dim)
        sequence, pooled = backbone(embeddings, modality="audio")

        assert sequence.shape == embeddings.shape
        assert pooled.shape == (batch_size, hidden_dim)

    def test_invalid_modality(self, batch_size, hidden_dim):
        """Test error with unknown modality."""
        config = BackboneConfig(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
        )
        backbone = SharedBackbone(config)

        embeddings = torch.randn(batch_size, 64, hidden_dim)
        with pytest.raises(ValueError, match="Unknown modality"):
            backbone(embeddings, modality="unknown")

    def test_with_attention_mask(self, batch_size, hidden_dim):
        """Test forward with attention mask."""
        config = BackboneConfig(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
        )
        backbone = SharedBackbone(config)

        embeddings = torch.randn(batch_size, 64, hidden_dim)
        mask = torch.ones(batch_size, 64, dtype=torch.bool)
        mask[:, 32:] = False

        sequence, pooled = backbone(embeddings, modality="vision", attention_mask=mask)
        assert sequence.shape == embeddings.shape
        assert pooled.shape == (batch_size, hidden_dim)

    def test_supported_modalities(self, batch_size, hidden_dim):
        """Test supported_modalities property."""
        config = BackboneConfig(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            modalities=("vision", "text", "audio"),
        )
        backbone = SharedBackbone(config)

        modalities = backbone.supported_modalities
        assert "vision" in modalities
        assert "text" in modalities
        assert "audio" in modalities
        assert len(modalities) == 3

    def test_get_modality_embedding(self, batch_size, hidden_dim):
        """Test get_modality_embedding method."""
        config = BackboneConfig(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
        )
        backbone = SharedBackbone(config)

        embeddings = torch.randn(batch_size, 64, hidden_dim)
        pooled = backbone.get_modality_embedding(embeddings, modality="vision")

        assert pooled.shape == (batch_size, hidden_dim)

    def test_get_sequence_embeddings(self, batch_size, hidden_dim):
        """Test get_sequence_embeddings method."""
        config = BackboneConfig(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
        )
        backbone = SharedBackbone(config)

        embeddings = torch.randn(batch_size, 64, hidden_dim)
        sequence = backbone.get_sequence_embeddings(embeddings, modality="vision")

        assert sequence.shape == embeddings.shape

    def test_get_num_params(self, batch_size, hidden_dim):
        """Test get_num_params method."""
        config = BackboneConfig(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
        )
        backbone = SharedBackbone(config)

        num_params = backbone.get_num_params()
        assert num_params > 0
        assert isinstance(num_params, int)

    def test_extra_repr(self, batch_size, hidden_dim):
        """Test extra_repr method."""
        config = BackboneConfig(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
        )
        backbone = SharedBackbone(config)

        repr_str = backbone.extra_repr()
        assert f"hidden_dim={hidden_dim}" in repr_str
        assert "num_layers=2" in repr_str
        assert "modalities=" in repr_str

    def test_gradient_flow(self, batch_size, hidden_dim):
        """Test gradient flow through backbone."""
        config = BackboneConfig(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
        )
        backbone = SharedBackbone(config)

        embeddings = torch.randn(batch_size, 64, hidden_dim, requires_grad=True)
        sequence, pooled = backbone(embeddings, modality="vision")
        loss = sequence.sum() + pooled.sum()
        loss.backward()
        assert embeddings.grad is not None

    def test_gradient_checkpointing(self, batch_size, hidden_dim):
        """Test with gradient checkpointing enabled."""
        config = BackboneConfig(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            use_gradient_checkpointing=True,
        )
        backbone = SharedBackbone(config)
        backbone.train()

        embeddings = torch.randn(batch_size, 64, hidden_dim, requires_grad=True)
        sequence, pooled = backbone(embeddings, modality="vision")
        loss = sequence.sum() + pooled.sum()
        loss.backward()
        assert embeddings.grad is not None

    def test_different_sequence_lengths(self, batch_size, hidden_dim):
        """Test with different sequence lengths."""
        config = BackboneConfig(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
        )
        backbone = SharedBackbone(config)

        for seq_len in [32, 64, 128]:
            embeddings = torch.randn(batch_size, seq_len, hidden_dim)
            sequence, pooled = backbone(embeddings, modality="vision")
            assert sequence.shape == embeddings.shape
            assert pooled.shape == (batch_size, hidden_dim)

    def test_modality_tokens_different(self, batch_size, hidden_dim):
        """Test that different modalities use different tokens."""
        config = BackboneConfig(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
        )
        backbone = SharedBackbone(config)

        # Check that vision and text tokens are different
        vision_token = backbone.modality_tokens["vision"]
        text_token = backbone.modality_tokens["text"]

        assert not torch.equal(vision_token, text_token)

    def test_deterministic_eval_mode(self, batch_size, hidden_dim):
        """Test deterministic output in eval mode."""
        config = BackboneConfig(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
        )
        backbone = SharedBackbone(config)
        backbone.eval()

        embeddings = torch.randn(batch_size, 64, hidden_dim)
        seq1, pool1 = backbone(embeddings, modality="vision")
        seq2, pool2 = backbone(embeddings, modality="vision")

        assert torch.allclose(seq1, seq2)
        assert torch.allclose(pool1, pool2)
