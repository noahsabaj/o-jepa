"""Tests for HierarchicalChunking module."""

import pytest
import torch

from src.byte_encoder import HierarchicalChunking
from src.config import ByteEncoderConfig


class TestHierarchicalChunking:
    """Tests for HierarchicalChunking."""

    @pytest.fixture
    def hierarchical(self):
        """Create default HierarchicalChunking module."""
        return HierarchicalChunking(
            input_dim=128,
            output_dim=128,
            scales=(4, 16, 64),
            gating="softmax",
        )

    def test_forward_shape(self, hierarchical):
        """Test output shape matches input."""
        x = torch.randn(2, 256, 128)
        out = hierarchical(x)
        assert out.shape == x.shape

    def test_different_batch_sizes(self, hierarchical):
        """Test with different batch sizes."""
        for batch_size in [1, 2, 8]:
            x = torch.randn(batch_size, 256, 128)
            out = hierarchical(x)
            assert out.shape == x.shape

    def test_different_sequence_lengths(self, hierarchical):
        """Test with different sequence lengths."""
        for seq_len in [64, 128, 512, 1024]:
            x = torch.randn(2, seq_len, 128)
            out = hierarchical(x)
            assert out.shape == x.shape

    def test_different_scales(self):
        """Test with different scale configurations."""
        for scales in [(4,), (4, 16), (4, 16, 64), (4, 16, 64, 128)]:
            h = HierarchicalChunking(128, 128, scales=scales)
            x = torch.randn(2, 256, 128)
            out = h(x)
            assert out.shape == x.shape
            assert h.num_scales == len(scales)

    def test_gating_softmax(self):
        """Test softmax gating."""
        h = HierarchicalChunking(128, 128, scales=(4, 16), gating="softmax")
        x = torch.randn(2, 64, 128)
        out = h(x)
        assert out.shape == x.shape

        # Check that gate weights sum to 1
        gate_logits = h.gate_proj(x)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        sums = gate_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gating_sigmoid(self):
        """Test sigmoid gating."""
        h = HierarchicalChunking(128, 128, scales=(4, 16), gating="sigmoid")
        x = torch.randn(2, 64, 128)
        out = h(x)
        assert out.shape == x.shape

    def test_gating_linear(self):
        """Test linear gating (no nonlinearity)."""
        h = HierarchicalChunking(128, 128, scales=(4, 16), gating="linear")
        x = torch.randn(2, 64, 128)
        out = h(x)
        assert out.shape == x.shape

    def test_gradient_flow(self, hierarchical):
        """Test gradients flow through module."""
        x = torch.randn(2, 256, 128, requires_grad=True)
        out = hierarchical(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_conv_weights_gradients(self):
        """Test gradients flow to convolution weights."""
        h = HierarchicalChunking(128, 128, scales=(4, 16))
        x = torch.randn(2, 64, 128)
        out = h(x)
        loss = out.sum()
        loss.backward()

        for conv in h.scale_convs:
            assert conv.weight.grad is not None

    def test_gate_proj_gradients(self):
        """Test gradients flow to gate projection."""
        h = HierarchicalChunking(128, 128, scales=(4, 16))
        x = torch.randn(2, 64, 128)
        out = h(x)
        loss = out.sum()
        loss.backward()

        assert h.gate_proj.weight.grad is not None
        assert h.out_proj.weight.grad is not None

    def test_different_input_output_dims(self):
        """Test with different input/output dimensions."""
        h = HierarchicalChunking(input_dim=128, output_dim=256)
        x = torch.randn(2, 64, 128)
        out = h(x)
        assert out.shape == (2, 64, 256)

    def test_dropout(self):
        """Test dropout is applied during training."""
        h = HierarchicalChunking(128, 128, scales=(4, 16), dropout=0.5)
        h.train()
        x = torch.randn(2, 64, 128)

        # Run multiple times and check outputs differ
        outputs = [h(x) for _ in range(5)]
        # At least some outputs should be different due to dropout
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Dropout should produce different outputs"

    def test_eval_mode_deterministic(self):
        """Test output is deterministic in eval mode."""
        h = HierarchicalChunking(128, 128, scales=(4, 16), dropout=0.5)
        h.eval()
        x = torch.randn(2, 64, 128)

        out1 = h(x)
        out2 = h(x)
        assert torch.allclose(out1, out2)

    def test_causal_padding(self):
        """Test that convolutions use causal padding (no future leakage)."""
        h = HierarchicalChunking(128, 128, scales=(4,))
        x = torch.randn(2, 256, 128)

        # Modify future positions
        x_modified = x.clone()
        x_modified[:, 200:, :] = 0

        out1 = h(x)
        out2 = h(x_modified)

        # First portion should be identical (causal - can't see future)
        # Due to kernel size 4, positions 0 to 196 should not see position 200+
        # Actually with causal padding, position i sees positions max(0, i-k+1) to i
        # So positions 0-196 shouldn't be affected by changes at 200+
        assert torch.allclose(out1[:, :196, :], out2[:, :196, :], atol=1e-5)

    def test_extra_repr(self, hierarchical):
        """Test string representation."""
        repr_str = hierarchical.extra_repr()
        assert "scales=" in repr_str
        assert "gating=" in repr_str
        assert "softmax" in repr_str

    def test_initialization(self):
        """Test weight initialization."""
        h = HierarchicalChunking(128, 128, scales=(4, 16, 64))

        # Conv weights should not be zero
        for conv in h.scale_convs:
            assert not torch.all(conv.weight == 0)

        # Gate proj should not be zero
        assert not torch.all(h.gate_proj.weight == 0)

    def test_very_short_sequence(self):
        """Test with sequence shorter than largest kernel."""
        h = HierarchicalChunking(128, 128, scales=(4, 16, 64))
        x = torch.randn(2, 32, 128)  # seq_len < max(scales)
        out = h(x)
        assert out.shape == x.shape

    def test_single_scale(self):
        """Test with single scale."""
        h = HierarchicalChunking(128, 128, scales=(8,))
        x = torch.randn(2, 64, 128)
        out = h(x)
        assert out.shape == x.shape
        assert h.num_scales == 1


class TestByteEncoderWithHierarchical:
    """Tests for ByteEncoder with hierarchical encoding (always enabled)."""

    @pytest.fixture
    def encoder_config(self):
        """Config with custom hierarchical scales."""
        return ByteEncoderConfig(
            hidden_dim=128,
            num_layers=1,
            num_heads=4,
            max_seq_len=1024,
            hierarchical_scales=(4, 16),
            hierarchical_gating="softmax",
        )

    @pytest.fixture
    def encoder(self, encoder_config):
        """ByteEncoder instance."""
        from src.byte_encoder import ByteEncoder
        return ByteEncoder(encoder_config)

    def test_forward_with_hierarchical(self, encoder):
        """Test forward pass with hierarchical processing."""
        byte_ids = torch.randint(0, 256, (2, 256), dtype=torch.long)
        out = encoder(byte_ids)
        assert out.shape == (2, 256, 128)

    def test_hierarchical_always_enabled(self):
        """Hierarchical module is always present (not optional)."""
        from src.byte_encoder import ByteEncoder
        config = ByteEncoderConfig(hidden_dim=128, num_layers=1, num_heads=4)
        encoder = ByteEncoder(config)
        assert hasattr(encoder, 'hierarchical')
        assert encoder.hierarchical is not None

    def test_default_hierarchical_scales(self):
        """Default config should have hierarchical scales (4, 16, 64)."""
        config = ByteEncoderConfig()
        assert config.hierarchical_scales == (4, 16, 64)
        assert config.hierarchical_gating == "softmax"

    def test_hierarchical_module_exists(self, encoder):
        """Hierarchical module should always exist."""
        assert hasattr(encoder, 'hierarchical')
        assert encoder.hierarchical is not None

    def test_hierarchical_gradient_flow(self, encoder):
        """Test gradients flow through hierarchical module."""
        byte_ids = torch.randint(0, 256, (2, 128), dtype=torch.long)
        out = encoder(byte_ids)
        loss = out.sum()
        loss.backward()

        # Check hierarchical module has gradients
        assert encoder.hierarchical.gate_proj.weight.grad is not None

    def test_different_gating_types(self):
        """Test different gating types work with ByteEncoder."""
        from src.byte_encoder import ByteEncoder

        for gating in ["softmax", "sigmoid", "linear"]:
            config = ByteEncoderConfig(
                hidden_dim=128,
                num_layers=1,
                num_heads=4,
                hierarchical_gating=gating,
            )
            encoder = ByteEncoder(config)
            byte_ids = torch.randint(0, 256, (2, 128), dtype=torch.long)
            out = encoder(byte_ids)
            assert out.shape == (2, 128, 128)
