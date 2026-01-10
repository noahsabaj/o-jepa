"""
Tests for the shared TransformerBlock module.

S6: This module was created to deduplicate TransformerBlock code
that was previously duplicated in byte_encoder.py and backbone.py.
"""

import pytest
import torch
import torch.nn as nn

from src.transformer_block import TransformerBlock


class TestTransformerBlock:
    """Test unified TransformerBlock."""

    @pytest.fixture
    def config(self):
        """Standard test config."""
        return {
            "dim": 128,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "dropout": 0.0,
            "use_bias": False,
            "max_seq_len": 512,
        }

    @pytest.fixture
    def block(self, config):
        """Create test block."""
        return TransformerBlock(**config)

    def test_forward_shape(self, block, config):
        """Output should match input shape."""
        batch_size = 2
        seq_len = 64
        x = torch.randn(batch_size, seq_len, config["dim"])

        output = block(x)

        assert output.shape == x.shape

    def test_with_attention_mask(self, block, config):
        """Should work with attention mask."""
        batch_size = 2
        seq_len = 64
        x = torch.randn(batch_size, seq_len, config["dim"])
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[:, seq_len // 2:] = False

        output = block(x, attention_mask=mask)

        assert output.shape == x.shape

    def test_mlp_dim_override(self, config):
        """mlp_dim should override mlp_ratio."""
        custom_mlp_dim = 512
        block = TransformerBlock(
            dim=config["dim"],
            num_heads=config["num_heads"],
            mlp_dim=custom_mlp_dim,
        )

        # Check that FFN uses the custom dim
        assert block.ffn.hidden_dim == custom_mlp_dim

    def test_mlp_dim_alignment(self, config):
        """mlp_dim should be aligned to specified value."""
        alignment = 64
        block = TransformerBlock(
            dim=config["dim"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            mlp_dim_alignment=alignment,
        )

        # Check FFN hidden dim is aligned
        assert block.ffn.hidden_dim % alignment == 0

    def test_no_alignment_when_one(self, config):
        """mlp_dim_alignment=1 should not change mlp_dim."""
        block_aligned = TransformerBlock(
            dim=config["dim"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            mlp_dim_alignment=256,
        )
        block_unaligned = TransformerBlock(
            dim=config["dim"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            mlp_dim_alignment=1,
        )

        # Unaligned should have exact computation
        expected_dim = int(config["dim"] * config["mlp_ratio"] * 2 / 3)
        assert block_unaligned.ffn.hidden_dim == expected_dim

        # Aligned should be rounded up
        assert block_aligned.ffn.hidden_dim >= expected_dim
        assert block_aligned.ffn.hidden_dim % 256 == 0

    def test_gradient_flow(self, block, config):
        """Gradients should flow through the block."""
        x = torch.randn(2, 32, config["dim"], requires_grad=True)

        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_deterministic_eval(self, block, config):
        """Output should be deterministic in eval mode."""
        block.eval()
        x = torch.randn(2, 32, config["dim"])

        output1 = block(x)
        output2 = block(x)

        assert torch.allclose(output1, output2)

    def test_with_dropout(self, config):
        """Block should work with dropout."""
        config_with_dropout = {**config, "dropout": 0.1}
        block = TransformerBlock(**config_with_dropout)

        x = torch.randn(2, 32, config["dim"])
        block.train()
        output = block(x)

        assert output.shape == x.shape

    def test_with_bias(self, config):
        """Block should work with bias enabled."""
        config_with_bias = {**config, "use_bias": True}
        block = TransformerBlock(**config_with_bias)

        x = torch.randn(2, 32, config["dim"])
        output = block(x)

        assert output.shape == x.shape


class TestTransformerBlockBackwardsCompat:
    """Test that TransformerBlock works as drop-in replacement."""

    def test_byte_encoder_usage(self):
        """Should work with ByteEncoder-style parameters."""
        # ByteEncoder used to pass mlp_dim directly
        block = TransformerBlock(
            dim=256,
            num_heads=8,
            mlp_dim=1024,
            dropout=0.0,
            use_bias=False,
            max_seq_len=8192,
        )

        x = torch.randn(2, 64, 256)
        output = block(x)

        assert output.shape == x.shape

    def test_backbone_usage(self):
        """Should work with SharedBackbone-style parameters."""
        # Backbone used mlp_ratio with alignment
        block = TransformerBlock(
            dim=512,
            num_heads=8,
            mlp_ratio=4.0,
            mlp_dim_alignment=256,
            dropout=0.0,
            use_bias=False,
            max_seq_len=8192,
        )

        x = torch.randn(2, 64, 512)
        output = block(x)

        assert output.shape == x.shape
        assert block.ffn.hidden_dim % 256 == 0
