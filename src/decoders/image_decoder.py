"""
Image decoder for byte-level O-JEPA.

Generates RGB image bytes from embeddings.
"""

from typing import Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDecoder
from ..config import ImageDecoderConfig
from ..layers import RMSNorm


class ImageDecoder(BaseDecoder):
    """
    Image decoder that generates RGB bytes from embeddings.

    Architecture:
    1. Project embedding through MLP
    2. Reshape to spatial grid
    3. Upsample with transposed convolutions
    4. Output RGB values (0-255)

    This is a lightweight decoder focused on small images (32×32).
    For high-resolution generation, a more sophisticated architecture
    (e.g., diffusion) would be needed.

    Args:
        config: ImageDecoderConfig with model hyperparameters
    """

    def __init__(self, config: ImageDecoderConfig):
        super().__init__(config.input_dim, config.hidden_dim)
        self.config = config
        self.output_size = config.output_size  # (H, W)
        self.output_channels = config.output_channels  # 3 for RGB

        # Calculate spatial dimensions
        self.H, self.W = config.output_size
        self.output_bytes = self.H * self.W * self.output_channels

        # Initial spatial size (before upsampling)
        self.init_H = self.H // 4  # 8 for 32×32
        self.init_W = self.W // 4  # 8 for 32×32

        # Project embedding to initial spatial grid
        self.proj = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim * self.init_H * self.init_W),
        )

        # Upsampling blocks (8×8 -> 16×16 -> 32×32)
        self.upsample = nn.Sequential(
            # Block 1: 8×8 -> 16×16
            nn.ConvTranspose2d(config.hidden_dim, config.hidden_dim // 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, config.hidden_dim // 2),
            nn.GELU(),

            # Block 2: 16×16 -> 32×32
            nn.ConvTranspose2d(config.hidden_dim // 2, config.hidden_dim // 4, 4, stride=2, padding=1),
            nn.GroupNorm(4, config.hidden_dim // 4),
            nn.GELU(),

            # Output layer
            nn.Conv2d(config.hidden_dim // 4, self.output_channels, 3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Decode embedding to RGB image.

        Args:
            embedding: Input embedding [batch, embed_dim]

        Returns:
            RGB values [batch, 3, H, W] in range [0, 255]
        """
        batch_size = embedding.shape[0]

        # Project to spatial grid
        x = self.proj(embedding)  # [batch, hidden_dim * H * W]
        x = x.view(batch_size, self.hidden_dim, self.init_H, self.init_W)

        # Upsample
        x = self.upsample(x)  # [batch, 3, H, W]

        # Sigmoid to [0, 1], then scale to [0, 255]
        x = torch.sigmoid(x) * 255.0

        return x

    def forward_bytes(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Decode embedding to flattened RGB bytes.

        Args:
            embedding: Input embedding [batch, embed_dim]

        Returns:
            RGB bytes [batch, H*W*3] in range [0, 255]
        """
        x = self.forward(embedding)  # [batch, 3, H, W]

        # Permute to [batch, H, W, 3] then flatten
        x = x.permute(0, 2, 3, 1)  # [batch, H, W, 3]
        x = x.reshape(x.shape[0], -1)  # [batch, H*W*3]

        return x

    def generate(
        self,
        embedding: torch.Tensor,
        return_tensor: bool = False,
    ) -> np.ndarray:
        """
        Generate RGB image from embedding.

        Args:
            embedding: Input embedding [batch, embed_dim]
            return_tensor: If True, return torch.Tensor instead of numpy

        Returns:
            RGB image(s) as numpy array [batch, H, W, 3] uint8
            or torch.Tensor if return_tensor=True
        """
        with torch.no_grad():
            x = self.forward(embedding)  # [batch, 3, H, W]

        # Permute to [batch, H, W, 3]
        x = x.permute(0, 2, 3, 1)

        # Clamp and convert to uint8
        x = x.clamp(0, 255)

        if return_tensor:
            return x.to(torch.uint8)
        else:
            return x.cpu().numpy().astype(np.uint8)

    def compute_loss(
        self,
        embedding: torch.Tensor,
        target_bytes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for image generation.

        Args:
            embedding: Input embedding [batch, embed_dim]
            target_bytes: Target image bytes [batch, H*W*3] in [0, 255]

        Returns:
            MSE loss (scalar)
        """
        # Get predictions
        pred_bytes = self.forward_bytes(embedding)  # [batch, H*W*3]

        # Normalize to [0, 1] for loss computation
        pred_norm = pred_bytes / 255.0
        target_norm = target_bytes.float() / 255.0

        # MSE loss
        loss = F.mse_loss(pred_norm, target_norm)

        return loss

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, "
            f"output_size={self.output_size}, "
            f"output_bytes={self.output_bytes}"
        )
