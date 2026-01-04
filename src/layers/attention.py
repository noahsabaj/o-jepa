"""
Attention layers for byte-level O-JEPA.

Implements Multi-Head Attention with RoPE (Rotary Position Embeddings).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .positional import RotaryPositionalEmbedding, apply_rotary_pos_emb


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with optional RoPE.

    Uses PyTorch's scaled_dot_product_attention for efficiency
    (automatically uses Flash Attention 2 when available).

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        dropout: Attention dropout rate
        bias: Whether to use bias in projections
        use_rope: Whether to use Rotary Position Embeddings
        max_seq_len: Maximum sequence length for RoPE
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
        use_rope: bool = True,
        max_seq_len: int = 8192,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.use_rope = use_rope

        # QKV projection (fused for efficiency)
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)

        # Output projection
        self.proj = nn.Linear(dim, dim, bias=bias)

        # RoPE
        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        else:
            self.rope = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize with small values for stable training."""
        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Apply multi-head self-attention.

        Args:
            x: Input tensor [batch, seq_len, dim]
            attention_mask: Optional mask [batch, seq_len] or [batch, 1, seq_len, seq_len]
                           True/1 = attend, False/0 = mask out
            is_causal: Whether to apply causal masking

        Returns:
            Output tensor [batch, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE if enabled
        if self.use_rope and self.rope is not None:
            cos, sin = self.rope(seq_len)
            cos = cos.to(q.device)
            sin = sin.to(q.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Prepare attention mask for SDPA
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                # [batch, seq_len] -> [batch, 1, 1, seq_len]
                attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # Convert to additive mask (True -> 0, False -> -inf)
                attn_mask = attn_mask.float()
                attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf"))
                attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
            elif attention_mask.ndim == 4:
                attn_mask = attention_mask

        # Use PyTorch's efficient SDPA (Flash Attention 2 when available)
        dropout_p = self.dropout if self.training else 0.0

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal and attn_mask is None,
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        output = self.proj(attn_output)

        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, rope={self.use_rope}"


class CrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention for decoder operations.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        dropout: Attention dropout rate
        bias: Whether to use bias in projections
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout

        # Separate Q projection for queries
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        # KV projection for context
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=bias)
        # Output projection
        self.proj = nn.Linear(dim, dim, bias=bias)

        self._init_weights()

    def _init_weights(self):
        for module in [self.q_proj, self.kv_proj, self.proj]:
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply cross-attention.

        Args:
            x: Query tensor [batch, seq_len, dim]
            context: Context tensor [batch, context_len, dim]
            context_mask: Optional mask [batch, context_len]

        Returns:
            Output tensor [batch, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        context_len = context.shape[1]

        # Compute Q from x, K/V from context
        q = self.q_proj(x)
        kv = self.kv_proj(context)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv = kv.reshape(batch_size, context_len, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Prepare attention mask
        attn_mask = None
        if context_mask is not None:
            attn_mask = context_mask.unsqueeze(1).unsqueeze(2).float()
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf"))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)

        # Compute attention
        dropout_p = self.dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        output = self.proj(attn_output)

        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}"
