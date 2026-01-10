"""
Shared Transformer Block for O-JEPA.

Unified transformer block used by both ByteEncoder and SharedBackbone.
Eliminates code duplication between byte_encoder.py and backbone.py.

Design:
    - Pre-normalization architecture (RMSNorm before attention/FFN)
    - Multi-head self-attention with RoPE
    - SwiGLU feedforward network
    - Optional MLP dimension alignment for tensor core efficiency
"""

from typing import Optional

import torch
import torch.nn as nn

from .config import SWIGLU_HIDDEN_RATIO
from .layers import (
    MultiHeadAttention,
    SwiGLU,
    RMSNorm,
)


class TransformerBlock(nn.Module):
    """
    Unified transformer block with pre-normalization.

    Used by both ByteEncoder and SharedBackbone. Supports two ways to
    specify FFN hidden dimension:

    1. Direct: Pass mlp_dim directly
    2. Ratio-based: Pass mlp_ratio to compute mlp_dim from input dim

    When both are provided, mlp_dim takes precedence.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        mlp_dim: FFN hidden dimension (if provided, takes precedence over mlp_ratio)
        mlp_ratio: Ratio to compute mlp_dim = dim * mlp_ratio * SWIGLU_HIDDEN_RATIO
        mlp_dim_alignment: Align mlp_dim to this value for tensor core efficiency.
            Use 1 for no alignment (default), 64 or 256 for GPU optimization.
        dropout: Dropout probability
        use_bias: Whether to use bias in linear layers
        max_seq_len: Maximum sequence length for RoPE
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        mlp_dim_alignment: int = 1,
        dropout: float = 0.0,
        use_bias: bool = False,
        max_seq_len: int = 8192,
    ):
        super().__init__()

        # Compute mlp_dim if not provided directly
        if mlp_dim is None:
            mlp_dim = int(dim * mlp_ratio * SWIGLU_HIDDEN_RATIO)

        # Apply alignment for tensor core efficiency
        if mlp_dim_alignment > 1:
            mlp_dim = ((mlp_dim + mlp_dim_alignment - 1) // mlp_dim_alignment) * mlp_dim_alignment

        # Pre-normalization layers
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        # Multi-head self-attention with RoPE
        self.attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=use_bias,
            use_rope=True,
            max_seq_len=max_seq_len,
        )

        # SwiGLU feedforward network
        self.ffn = SwiGLU(
            dim=dim,
            hidden_dim=mlp_dim,
            bias=use_bias,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            x: Input tensor [batch, seq_len, dim]
            attention_mask: Optional attention mask [batch, seq_len]

        Returns:
            Output tensor [batch, seq_len, dim]
        """
        # Attention with residual
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x
