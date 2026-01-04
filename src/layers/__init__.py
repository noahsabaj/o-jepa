"""
Transformer layers for byte-level O-JEPA.

This module provides the building blocks for the transformer architecture:
- Attention: Multi-Head Self-Attention with RoPE
- Feedforward: SwiGLU feedforward network
- Normalization: RMSNorm
- Positional: Rotary Position Embeddings
"""

from .attention import MultiHeadAttention, CrossAttention
from .feedforward import SwiGLU, FeedForward
from .normalization import RMSNorm, get_norm_layer
from .positional import (
    RotaryPositionalEmbedding,
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    apply_rotary_pos_emb,
)

__all__ = [
    # Attention
    "MultiHeadAttention",
    "CrossAttention",
    # Feedforward
    "SwiGLU",
    "FeedForward",
    # Normalization
    "RMSNorm",
    "get_norm_layer",
    # Positional
    "RotaryPositionalEmbedding",
    "LearnedPositionalEmbedding",
    "SinusoidalPositionalEmbedding",
    "apply_rotary_pos_emb",
]
