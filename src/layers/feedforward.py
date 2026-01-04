"""
Feedforward layers for byte-level O-JEPA.

Implements SwiGLU activation for improved training dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU activation function with feedforward network.

    SwiGLU combines Swish activation with a gating mechanism:
    SwiGLU(x) = Swish(xW) * (xV)

    This provides better training dynamics than standard ReLU or GELU.

    Reference: https://arxiv.org/abs/2002.05202

    Args:
        dim: Input/output dimension
        hidden_dim: Hidden dimension (if None, uses 4 * dim * 2/3 rounded to multiple of 256)
        bias: Whether to use bias in linear layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim

        # Compute hidden dimension
        if hidden_dim is None:
            # Standard transformer uses 4x, but SwiGLU needs 2/3 of that
            # to maintain similar param count (since we have 3 projections)
            hidden_dim = int(4 * dim * 2 / 3)
            # Round to nearest multiple of 256 for efficiency
            hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.hidden_dim = hidden_dim

        # Gate and up projections (fused for efficiency)
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)  # Gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)  # Down
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)  # Up

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize with small values for stable training."""
        for module in [self.w1, self.w2, self.w3]:
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU feedforward.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Output tensor [..., dim]
        """
        # SwiGLU: swish(x @ W1) * (x @ W3) @ W2
        gate = F.silu(self.w1(x))  # Swish = SiLU
        up = self.w3(x)
        x = gate * up
        x = self.dropout(x)
        x = self.w2(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, hidden_dim={self.hidden_dim}"


class FeedForward(nn.Module):
    """
    Standard feedforward network with GELU activation.

    Provided for comparison with SwiGLU.

    Args:
        dim: Input/output dimension
        hidden_dim: Hidden dimension (default: 4 * dim)
        bias: Whether to use bias in linear layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or 4 * dim

        self.w1 = nn.Linear(dim, self.hidden_dim, bias=bias)
        self.w2 = nn.Linear(self.hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for module in [self.w1, self.w2]:
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feedforward with GELU.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Output tensor [..., dim]
        """
        x = self.w1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, hidden_dim={self.hidden_dim}"
