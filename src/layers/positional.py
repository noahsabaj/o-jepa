"""
Positional embeddings for byte-level O-JEPA.

Implements Rotary Position Embeddings (RoPE) for better length generalization.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    RoPE encodes position information by rotating query and key vectors,
    which provides better length generalization than absolute position embeddings.

    Reference: https://arxiv.org/abs/2104.09864

    Args:
        dim: Dimension of the embedding (must be even)
        max_seq_len: Maximum sequence length to pre-compute
        base: Base for computing frequencies (default 10000)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos and sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build cos/sin cache for given sequence length."""
        self.max_seq_len = max(self.max_seq_len, seq_len)

        # Create position indices
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device)

        # Compute frequencies: [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)

        # Create cos/sin embeddings: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin for given sequence length.

        Args:
            seq_len: Length of sequence

        Returns:
            Tuple of (cos, sin) tensors of shape [seq_len, dim]
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        cos: Cosine tensor [seq_len, head_dim]
        sin: Sine tensor [seq_len, head_dim]

    Returns:
        Tuple of rotated (q, k) tensors
    """

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Reshape cos/sin to broadcast correctly
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned absolute positional embeddings.

    Simple but effective positional encoding using learnable embeddings.

    Args:
        max_seq_len: Maximum sequence length
        dim: Embedding dimension
    """

    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.embedding = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, seq_len_or_indices) -> torch.Tensor:
        """
        Get positional embeddings for given sequence length or indices.

        Args:
            seq_len_or_indices: Either:
                - int: sequence length, returns [1, seq_len, dim]
                - Tensor[N]: position indices, returns [N, dim]

        Returns:
            Positional embeddings
        """
        if isinstance(seq_len_or_indices, int):
            # Standard mode: return embeddings for sequence length
            seq_len = seq_len_or_indices
            if seq_len > self.max_seq_len:
                raise ValueError(
                    f"Sequence length {seq_len} exceeds max {self.max_seq_len}"
                )
            return self.embedding[:, :seq_len, :]
        else:
            # Index mode: return embeddings for specific positions
            indices = seq_len_or_indices
            # Clamp indices to valid range
            indices = torch.clamp(indices, 0, self.max_seq_len - 1)
            return self.embedding[0, indices, :]  # [N, dim]


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings (non-learnable).

    Classic positional encoding from "Attention Is All You Need".

    Args:
        dim: Embedding dimension
        max_seq_len: Maximum sequence length
    """

    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Build sinusoidal embeddings
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_len, dim]

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Get positional embeddings for given sequence length.

        Args:
            seq_len: Length of sequence

        Returns:
            Positional embeddings [1, seq_len, dim]
        """
        return self.pe[:, :seq_len, :]
