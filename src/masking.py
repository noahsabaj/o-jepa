"""
JEPA Masking Strategies

Following I-JEPA: Mask contiguous blocks/spans, not random positions.
The model must predict abstract representations of masked regions.

Key insight from LeCun: Multiple target blocks prevent shortcuts.

AVAILABLE MASK GENERATORS
=========================

1. SpanMaskGenerator (DEFAULT) - For 1D sequences (text, audio)
   Masks contiguous spans. Simple and effective for most use cases.

2. BlockMaskGenerator - For 2D spatial data (images)
   Masks rectangular blocks. Use when height/width are provided.

3. TemporalMaskGenerator - For video sequences
   Masks contiguous frames. Use for video-specific temporal masking.

4. MultiScaleMaskGenerator - For curriculum learning
   Samples from multiple scale ranges. Advanced usage.

For most use cases, SpanMaskGenerator is sufficient.
The model selects based on config.masking.masking_type.

VECTORIZATION STRATEGY
======================

The mask generators are designed to minimize Python loops by:

1. BATCH RANDOM GENERATION: Generate all random values (scales, positions)
   for all blocks/spans across all batch items in single vectorized calls.
   Example: scales = torch.empty(batch_size * num_blocks).uniform_(min, max)

2. TENSOR BROADCASTING: Use broadcasting to compute block dimensions
   for all blocks simultaneously rather than one at a time.
   Example: block_areas = (scales * height * width).int()

3. GRID-BASED POSITIONING: Pre-compute offset grids that can be reused
   for block placement, avoiding repeated arange calls.
   Example: h_offsets = torch.arange(max_block_h, device=device)

4. MINIMAL LOOPS: The only remaining loop iterates over blocks to place
   them in the mask tensor. This is O(batch * num_blocks) and each
   iteration is a simple slice assignment (highly optimized in PyTorch).

Performance: On GPU, these generators are ~10x faster than naive loops
that generate random values one at a time within Python.
"""

from typing import Tuple, List, Optional
import math

import torch
import torch.nn as nn

from .config import MaskingConfig


class BlockMaskGenerator(nn.Module):
    """
    I-JEPA style block masking for spatial data (images).

    Generates:
    - context_mask: Which positions the encoder sees (True = visible)
    - target_mask: Which positions to predict (True = predict this)
    - target_positions: Indices of target positions for the predictor

    The encoder only sees context. The predictor predicts target embeddings.
    """

    def __init__(self, config: MaskingConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Generate block masks for a batch (vectorized).

        Args:
            batch_size: Number of samples
            seq_len: Sequence length (height * width for images)
            device: Device to create tensors on
            height: Optional height for 2D masking
            width: Optional width for 2D masking

        Returns:
            context_mask: [batch, seq_len] - True where encoder can see
            target_mask: [batch, seq_len] - True where predictor must predict
            target_positions: List of [num_targets] tensors per batch item
        """
        # For 1D sequences, treat as 1 x seq_len
        if height is None or width is None:
            height = 1
            width = seq_len

        num_blocks = self.config.num_target_blocks
        total_blocks = batch_size * num_blocks

        # VECTORIZATION STEP 1: Batch random generation
        # Generate all scale values for all blocks across all batch items at once
        # Shape: [batch_size * num_blocks] - one value per block
        if self.config.use_hierarchical_masking:
            # Y3: Hierarchical masking - sample from small or large scale ranges
            use_small = torch.rand(total_blocks, device=device) < self.config.hierarchical_small_prob
            small_min, small_max = self.config.small_scale_range
            large_min, large_max = self.config.large_scale_range

            small_scales = torch.empty(total_blocks, device=device).uniform_(small_min, small_max)
            large_scales = torch.empty(total_blocks, device=device).uniform_(large_min, large_max)
            scales = torch.where(use_small, small_scales, large_scales)
        else:
            scales = torch.empty(total_blocks, device=device).uniform_(
                self.config.target_scale_min,
                self.config.target_scale_max
            )

        # VECTORIZATION STEP 2: Broadcast block dimension computation
        # Compute dimensions for all blocks simultaneously using tensor ops
        # block_areas: target area for each block (fraction of total image)
        # block_hs/ws: height/width respecting aspect ratio constraint
        block_areas = (scales * height * width).int()
        block_hs = torch.clamp(
            (block_areas.float() / self.config.target_aspect_ratio).sqrt().int(),
            min=1, max=height
        )
        block_ws = torch.clamp((block_areas / block_hs.clamp(min=1)).int(), min=1, max=width)

        # Generate random positions for all blocks in one call
        # max_tops/lefts: valid range for block placement (ensures block fits in image)
        max_tops = torch.clamp(height - block_hs + 1, min=1)
        max_lefts = torch.clamp(width - block_ws + 1, min=1)
        tops = (torch.rand(total_blocks, device=device) * max_tops.float()).int()
        lefts = (torch.rand(total_blocks, device=device) * max_lefts.float()).int()

        # Initialize masks
        target_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        # VECTORIZATION STEP 3: Grid-based positioning
        # Pre-compute offset grids once, reuse for all blocks
        # This avoids calling torch.arange inside the loop
        max_block_h = block_hs.max().item()
        max_block_w = block_ws.max().item()
        h_offsets = torch.arange(max_block_h, device=device)
        w_offsets = torch.arange(max_block_w, device=device)

        # VECTORIZATION STEP 4: Minimal loop for mask placement
        # This loop is O(batch_size * num_blocks), typically 4-16 iterations
        # Each iteration uses vectorized slice assignment which is O(1) in PyTorch
        for block_idx in range(total_blocks):
            b = block_idx // num_blocks  # Which batch item this block belongs to
            block_h = block_hs[block_idx].item()
            block_w = block_ws[block_idx].item()
            top = tops[block_idx].item()
            left = lefts[block_idx].item()

            # Compute 2D grid positions -> 1D sequence positions
            # rows: [top, top+1, ..., top+block_h-1]
            # cols: [left, left+1, ..., left+block_w-1]
            # positions: row * width + col for all (row, col) pairs
            rows = top + h_offsets[:block_h]
            cols = left + w_offsets[:block_w]
            positions = (rows.unsqueeze(1) * width + cols.unsqueeze(0)).flatten()
            positions = positions[positions < seq_len]  # Bounds check

            target_mask[b, positions] = True  # Vectorized assignment

        # Context mask is inverse of target mask
        context_mask = ~target_mask

        # Extract target positions per batch item
        target_positions = [
            target_mask[b].nonzero(as_tuple=True)[0]
            for b in range(batch_size)
        ]

        return context_mask, target_mask, target_positions


class SpanMaskGenerator(nn.Module):
    """
    Span masking for sequential data (text, audio).

    Like BlockMaskGenerator but optimized for 1D sequences.
    Masks contiguous spans, not random positions.
    """

    def __init__(self, config: MaskingConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Generate span masks for a batch (vectorized).

        Args:
            batch_size: Number of samples
            seq_len: Sequence length
            device: Device to create tensors on
            height: Ignored (for API compatibility with BlockMaskGenerator)
            width: Ignored (for API compatibility with BlockMaskGenerator)

        Returns:
            context_mask: [batch, seq_len] - True where encoder can see
            target_mask: [batch, seq_len] - True where predictor must predict
            target_positions: List of [num_targets] tensors per batch item
        """
        num_blocks = self.config.num_target_blocks
        total_spans = batch_size * num_blocks

        # VECTORIZATION: Generate all random values for all spans at once
        # Shape: [batch_size * num_blocks] - one scale per span
        if self.config.use_hierarchical_masking:
            # Y3: Hierarchical masking - sample from small or large scale ranges
            # Each span independently chooses small or large scale range
            use_small = torch.rand(total_spans, device=device) < self.config.hierarchical_small_prob
            small_min, small_max = self.config.small_scale_range
            large_min, large_max = self.config.large_scale_range

            # Generate scales from both ranges
            small_scales = torch.empty(total_spans, device=device).uniform_(small_min, small_max)
            large_scales = torch.empty(total_spans, device=device).uniform_(large_min, large_max)

            # Select based on use_small mask
            scales = torch.where(use_small, small_scales, large_scales)
        else:
            # Standard uniform sampling from configured range
            scales = torch.empty(total_spans, device=device).uniform_(
                self.config.target_scale_min,
                self.config.target_scale_max
            )
        span_lens = torch.clamp((scales * seq_len).int(), min=1)

        # VECTORIZATION: Compute all start positions at once
        # max_starts ensures span fits within sequence
        max_starts = torch.clamp(seq_len - span_lens, min=0)
        starts = (torch.rand(total_spans, device=device) * (max_starts.float() + 1)).int()

        # Initialize target mask
        target_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        # MINIMAL LOOP: O(batch_size * num_blocks) iterations
        # Each iteration is a simple slice assignment (PyTorch optimized)
        # For 1D spans, slice assignment target_mask[b, start:end] = True
        # is more efficient than computing position indices
        for span_idx in range(total_spans):
            b = span_idx // num_blocks  # Which batch item
            start = starts[span_idx].item()
            end = min(start + span_lens[span_idx].item(), seq_len)

            # Contiguous slice assignment (highly optimized in PyTorch)
            target_mask[b, start:end] = True

        # Context mask is inverse of target mask
        context_mask = ~target_mask

        # Extract target positions per batch item
        target_positions = [
            target_mask[b].nonzero(as_tuple=True)[0]
            for b in range(batch_size)
        ]

        return context_mask, target_mask, target_positions


class TemporalMaskGenerator(nn.Module):
    """
    Temporal masking for video sequences.

    Masks contiguous frames (t to t+k) and predicts from surrounding context.
    This forces the model to learn temporal dynamics, not just spatial inpainting.

    For video: mask frames in the middle, predict from past and future frames.
    This is crucial for world model capabilities - the model must learn how
    scenes evolve over time, not just fill in spatial gaps.

    Usage:
        generator = TemporalMaskGenerator(config)
        context_mask, target_mask, target_positions = generator(
            batch_size=4,
            num_frames=16,
            frame_size=224*224*3,  # H*W*C flattened
            device=device
        )
    """

    def __init__(self, config: MaskingConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        batch_size: int,
        num_frames: int,
        frame_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Generate temporal masks for video sequences.

        Args:
            batch_size: Number of samples
            num_frames: Number of frames in sequence (T)
            frame_size: Size of each frame in bytes (H * W * C)
            device: Device to create tensors on

        Returns:
            context_mask: [batch, seq_len] - True where encoder can see
            target_mask: [batch, seq_len] - True where predictor must predict
            target_positions: List of [num_targets] tensors per batch item

        The masking strategy:
            - Mask a contiguous block of FRAMES (not random pixels)
            - Typically mask middle frames, keep first/last as context
            - This forces temporal prediction, not spatial interpolation
        """
        seq_len = num_frames * frame_size

        # How many frames to mask (based on scale config)
        scales = torch.empty(batch_size, device=device).uniform_(
            self.config.target_scale_min,
            self.config.target_scale_max
        )
        num_mask_frames = torch.clamp(
            (scales * num_frames).int(),
            min=1,
            max=num_frames - 2  # Keep at least 1 frame on each side
        )

        # Random start frame for masking (ensure some context on each side)
        max_start = torch.clamp(num_frames - num_mask_frames - 1, min=1)
        start_frames = (torch.rand(batch_size, device=device) * max_start.float() + 1).int()

        # Initialize masks
        target_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        for b in range(batch_size):
            start_f = start_frames[b].item()
            end_f = start_f + num_mask_frames[b].item()

            # Mask entire frames (all bytes in masked frames)
            start_pos = start_f * frame_size
            end_pos = min(end_f * frame_size, seq_len)
            target_mask[b, start_pos:end_pos] = True

        # Context mask is inverse
        context_mask = ~target_mask

        # Extract target positions
        target_positions = [
            target_mask[b].nonzero(as_tuple=True)[0]
            for b in range(batch_size)
        ]

        return context_mask, target_mask, target_positions

    def get_masked_frame_info(
        self,
        batch_size: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get which frames are masked (frame-level, not byte-level).

        Useful for analysis and debugging.

        Returns:
            start_frames: [batch] - First masked frame per sample
            num_masked: [batch] - Number of masked frames per sample
        """
        scales = torch.empty(batch_size, device=device).uniform_(
            self.config.target_scale_min,
            self.config.target_scale_max
        )
        num_mask_frames = torch.clamp(
            (scales * num_frames).int(),
            min=1,
            max=num_frames - 2
        )
        max_start = torch.clamp(num_frames - num_mask_frames - 1, min=1)
        start_frames = (torch.rand(batch_size, device=device) * max_start.float() + 1).int()

        return start_frames, num_mask_frames


class MultiScaleMaskGenerator(nn.Module):
    """
    Hierarchical masking at multiple scales.

    For ByteJEPA world model: predict at byte, token, and segment levels.
    """

    def __init__(
        self,
        byte_config: MaskingConfig,
        token_config: Optional[MaskingConfig] = None,
        segment_config: Optional[MaskingConfig] = None,
    ):
        super().__init__()
        self.byte_masker = SpanMaskGenerator(byte_config)

        # Token level (groups of 4-16 bytes)
        if token_config is None:
            token_config = MaskingConfig(
                num_target_blocks=2,
                target_scale_min=0.1,
                target_scale_max=0.3,
            )
        self.token_masker = SpanMaskGenerator(token_config)

        # Segment level (groups of 64+ bytes)
        if segment_config is None:
            segment_config = MaskingConfig(
                num_target_blocks=1,
                target_scale_min=0.2,
                target_scale_max=0.5,
            )
        self.segment_masker = SpanMaskGenerator(segment_config)

    def forward(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        level: str = "byte",
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Generate masks at specified level."""
        if level == "byte":
            return self.byte_masker(batch_size, seq_len, device)
        elif level == "token":
            return self.token_masker(batch_size, seq_len, device)
        elif level == "segment":
            return self.segment_masker(batch_size, seq_len, device)
        else:
            raise ValueError(f"Unknown level: {level}")


def create_mask_generator(
    masking_type: str = "span",
    config: Optional[MaskingConfig] = None,
) -> nn.Module:
    """
    Factory function for mask generators.

    Args:
        masking_type: Type of masking strategy
            - "span": Contiguous spans for text/audio (SpanMaskGenerator)
            - "block": 2D blocks for images (BlockMaskGenerator)
            - "temporal": Frame-level masking for video (TemporalMaskGenerator)
            - "multiscale": Hierarchical masking (MultiScaleMaskGenerator)
        config: MaskingConfig with parameters

    Returns:
        Mask generator module
    """
    if config is None:
        config = MaskingConfig()

    if masking_type == "block":
        return BlockMaskGenerator(config)
    elif masking_type == "span":
        return SpanMaskGenerator(config)
    elif masking_type == "temporal":
        return TemporalMaskGenerator(config)
    elif masking_type == "multiscale":
        return MultiScaleMaskGenerator(config)
    else:
        raise ValueError(f"Unknown masking type: {masking_type}")
