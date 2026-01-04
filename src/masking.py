"""
JEPA Masking Strategies

Following I-JEPA: Mask contiguous blocks/spans, not random positions.
The model must predict abstract representations of masked regions.

Key insight from LeCun: Multiple target blocks prevent shortcuts.
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

        # Generate all random values at once
        scales = torch.empty(total_blocks, device=device).uniform_(
            self.config.target_scale_min,
            self.config.target_scale_max
        )

        # Compute block dimensions for all blocks
        block_areas = (scales * height * width).int()
        block_hs = torch.clamp(
            (block_areas.float() / self.config.target_aspect_ratio).sqrt().int(),
            min=1, max=height
        )
        block_ws = torch.clamp((block_areas / block_hs.clamp(min=1)).int(), min=1, max=width)

        # Random positions for all blocks
        max_tops = torch.clamp(height - block_hs + 1, min=1)
        max_lefts = torch.clamp(width - block_ws + 1, min=1)
        tops = (torch.rand(total_blocks, device=device) * max_tops.float()).int()
        lefts = (torch.rand(total_blocks, device=device) * max_lefts.float()).int()

        # Initialize masks
        target_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        # Build masks using vectorized block placement
        # Create position grids for each block
        max_block_h = block_hs.max().item()
        max_block_w = block_ws.max().item()

        # Generate offset grids
        h_offsets = torch.arange(max_block_h, device=device)
        w_offsets = torch.arange(max_block_w, device=device)

        for block_idx in range(total_blocks):
            b = block_idx // num_blocks
            block_h = block_hs[block_idx].item()
            block_w = block_ws[block_idx].item()
            top = tops[block_idx].item()
            left = lefts[block_idx].item()

            # Create position indices for this block
            rows = top + h_offsets[:block_h]
            cols = left + w_offsets[:block_w]
            positions = (rows.unsqueeze(1) * width + cols.unsqueeze(0)).flatten()
            positions = positions[positions < seq_len]

            target_mask[b, positions] = True

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

        # Generate all random values at once
        scales = torch.empty(total_spans, device=device).uniform_(
            self.config.target_scale_min,
            self.config.target_scale_max
        )
        span_lens = torch.clamp((scales * seq_len).int(), min=1)

        # Random start positions
        max_starts = torch.clamp(seq_len - span_lens, min=0)
        starts = (torch.rand(total_spans, device=device) * (max_starts.float() + 1)).int()

        # Initialize target mask
        target_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=device)

        # For each span, mark positions
        for span_idx in range(total_spans):
            b = span_idx // num_blocks
            start = starts[span_idx].item()
            end = min(start + span_lens[span_idx].item(), seq_len)

            # Mark span positions (vectorized within each span)
            target_mask[b, start:end] = True

        # Context mask is inverse of target mask
        context_mask = ~target_mask

        # Extract target positions per batch item
        target_positions = [
            target_mask[b].nonzero(as_tuple=True)[0]
            for b in range(batch_size)
        ]

        return context_mask, target_mask, target_positions


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
    """Factory function for mask generators."""
    if config is None:
        config = MaskingConfig()

    if masking_type == "block":
        return BlockMaskGenerator(config)
    elif masking_type == "span":
        return SpanMaskGenerator(config)
    elif masking_type == "multiscale":
        return MultiScaleMaskGenerator(config)
    else:
        raise ValueError(f"Unknown masking type: {masking_type}")
