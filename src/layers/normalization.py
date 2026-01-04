"""
Normalization layers for byte-level O-JEPA.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm simplifies LayerNorm by removing the mean-centering step,
    computing only: x * weight / sqrt(mean(x^2) + eps)

    This is more computationally efficient and works well for transformers.

    Args:
        dim: Hidden dimension to normalize over
        eps: Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Normalized tensor of same shape
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


def get_norm_layer(dim: int, eps: float = 1e-6) -> nn.Module:
    """
    Factory function for normalization layers.

    Args:
        dim: Hidden dimension
        eps: Epsilon for numerical stability

    Returns:
        RMSNorm module
    """
    return RMSNorm(dim, eps)
