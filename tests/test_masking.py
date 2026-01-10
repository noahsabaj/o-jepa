"""
Tests for JEPA masking strategies.
"""

import pytest
import torch

from src.masking import (
    SpanMaskGenerator,
    BlockMaskGenerator,
    MaskingConfig,
    create_mask_generator,
)


class TestMaskingConfig:
    """Tests for MaskingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MaskingConfig()
        assert config.num_target_blocks == 4
        assert config.target_scale_min == 0.15
        assert config.target_scale_max == 0.40
        assert config.use_hierarchical_masking == False

    def test_hierarchical_config(self):
        """Y3: Test hierarchical masking configuration."""
        config = MaskingConfig(
            use_hierarchical_masking=True,
            small_scale_range=(0.05, 0.15),
            large_scale_range=(0.40, 0.70),
            hierarchical_small_prob=0.5,
        )
        assert config.use_hierarchical_masking == True
        assert config.small_scale_range == (0.05, 0.15)
        assert config.large_scale_range == (0.40, 0.70)


class TestSpanMaskGenerator:
    """Tests for SpanMaskGenerator."""

    @pytest.fixture
    def config(self):
        return MaskingConfig(num_target_blocks=2)

    @pytest.fixture
    def generator(self, config):
        return SpanMaskGenerator(config)

    def test_output_shapes(self, generator):
        """Test output shapes."""
        batch_size = 4
        seq_len = 128
        device = torch.device("cpu")

        context_mask, target_mask, target_positions = generator(
            batch_size, seq_len, device
        )

        assert context_mask.shape == (batch_size, seq_len)
        assert target_mask.shape == (batch_size, seq_len)
        assert len(target_positions) == batch_size

    def test_masks_are_complementary(self, generator):
        """Context and target masks should be complementary."""
        batch_size = 4
        seq_len = 128
        device = torch.device("cpu")

        context_mask, target_mask, _ = generator(batch_size, seq_len, device)

        # Every position should be either context OR target
        combined = context_mask | target_mask
        assert combined.all()

        # No position should be both
        overlap = context_mask & target_mask
        assert not overlap.any()

    def test_hierarchical_masking(self):
        """Y3: Test hierarchical masking produces valid masks."""
        config = MaskingConfig(
            num_target_blocks=4,
            use_hierarchical_masking=True,
            small_scale_range=(0.05, 0.10),
            large_scale_range=(0.30, 0.50),
            hierarchical_small_prob=0.5,
        )
        generator = SpanMaskGenerator(config)

        batch_size = 8
        seq_len = 200
        device = torch.device("cpu")

        # Run and verify masks are valid
        context_mask, target_mask, target_positions = generator(
            batch_size, seq_len, device
        )

        # Masks should be valid
        assert context_mask.shape == (batch_size, seq_len)
        assert target_mask.shape == (batch_size, seq_len)

        # Some targets should be masked
        assert target_mask.sum() > 0

        # Masks should be complementary
        combined = context_mask | target_mask
        assert combined.all()

    def test_hierarchical_vs_standard_masking(self):
        """Y3: Hierarchical masking should produce different scale distributions."""
        # Standard masking config
        standard_config = MaskingConfig(
            num_target_blocks=1,  # Single span for clearer measurement
            target_scale_min=0.15,
            target_scale_max=0.40,
            use_hierarchical_masking=False,
        )

        # Hierarchical masking config
        hierarchical_config = MaskingConfig(
            num_target_blocks=1,
            use_hierarchical_masking=True,
            small_scale_range=(0.05, 0.10),
            large_scale_range=(0.40, 0.60),
            hierarchical_small_prob=0.5,
        )

        standard_gen = SpanMaskGenerator(standard_config)
        hierarchical_gen = SpanMaskGenerator(hierarchical_config)

        batch_size = 100
        seq_len = 200
        device = torch.device("cpu")

        # Measure span lengths from both generators
        standard_lengths = []
        hierarchical_lengths = []

        for _ in range(5):
            _, _, standard_pos = standard_gen(batch_size, seq_len, device)
            _, _, hierarchical_pos = hierarchical_gen(batch_size, seq_len, device)

            for b in range(batch_size):
                standard_lengths.append(len(standard_pos[b]))
                hierarchical_lengths.append(len(hierarchical_pos[b]))

        # Standard should be mostly in middle range (15-40% = 30-80 positions)
        standard_middle = sum(1 for l in standard_lengths if 25 < l < 85)
        assert standard_middle > len(standard_lengths) * 0.5, "Standard should cluster in middle"

        # Hierarchical should have more extreme values (small or large)
        hierarchical_small = sum(1 for l in hierarchical_lengths if l < 25)
        hierarchical_large = sum(1 for l in hierarchical_lengths if l > 75)
        extreme_ratio = (hierarchical_small + hierarchical_large) / len(hierarchical_lengths)

        # At least 30% of hierarchical samples should be extreme
        assert extreme_ratio > 0.3, f"Hierarchical should have more extreme spans, got {extreme_ratio:.2%}"


class TestBlockMaskGenerator:
    """Tests for BlockMaskGenerator."""

    @pytest.fixture
    def config(self):
        return MaskingConfig(num_target_blocks=2, masking_type="block")

    @pytest.fixture
    def generator(self, config):
        return BlockMaskGenerator(config)

    def test_output_shapes(self, generator):
        """Test output shapes with 2D masking."""
        batch_size = 4
        height, width = 16, 16
        seq_len = height * width
        device = torch.device("cpu")

        context_mask, target_mask, target_positions = generator(
            batch_size, seq_len, device, height=height, width=width
        )

        assert context_mask.shape == (batch_size, seq_len)
        assert target_mask.shape == (batch_size, seq_len)
        assert len(target_positions) == batch_size

    def test_hierarchical_masking_block(self):
        """Y3: Test hierarchical masking with block generator."""
        config = MaskingConfig(
            num_target_blocks=4,
            use_hierarchical_masking=True,
            small_scale_range=(0.05, 0.10),
            large_scale_range=(0.30, 0.50),
        )
        generator = BlockMaskGenerator(config)

        batch_size = 4
        height, width = 32, 32
        seq_len = height * width
        device = torch.device("cpu")

        context_mask, target_mask, target_positions = generator(
            batch_size, seq_len, device, height=height, width=width
        )

        # Should produce valid masks
        assert context_mask.shape == (batch_size, seq_len)
        assert target_mask.sum() > 0  # Some targets masked


class TestCreateMaskGenerator:
    """Tests for factory function."""

    def test_create_span(self):
        """Test creating span generator."""
        generator = create_mask_generator("span")
        assert isinstance(generator, SpanMaskGenerator)

    def test_create_block(self):
        """Test creating block generator."""
        generator = create_mask_generator("block")
        assert isinstance(generator, BlockMaskGenerator)

    def test_with_config(self):
        """Test creating with custom config."""
        config = MaskingConfig(num_target_blocks=8)
        generator = create_mask_generator("span", config)
        assert generator.config.num_target_blocks == 8
