"""
Audio decoder for byte-level O-JEPA.

Generates PCM audio bytes from embeddings.
"""

from typing import Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDecoder
from ..config import AudioDecoderConfig
from ..layers import RMSNorm


class AudioDecoder(BaseDecoder):
    """
    Audio decoder that generates PCM bytes from embeddings.

    Architecture:
    1. Project embedding through MLP
    2. Upsample with transposed 1D convolutions
    3. Output PCM values as bytes

    This is a lightweight decoder for short audio clips.
    For high-quality audio generation, a more sophisticated architecture
    (e.g., vocoder) would be needed.

    Args:
        config: AudioDecoderConfig with model hyperparameters
    """

    def __init__(self, config: AudioDecoderConfig):
        super().__init__(config.input_dim, config.hidden_dim)
        self.config = config
        self.output_length = config.output_length  # Number of bytes
        self.sample_rate = config.sample_rate

        # Initial sequence length (before upsampling)
        # For 8000 bytes output with 4x upsampling: 8000 / 8 = 1000
        self.init_length = self.output_length // 8

        # Project embedding to initial sequence
        self.proj = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim * self.init_length),
        )

        # Upsampling blocks (1000 -> 2000 -> 4000 -> 8000)
        self.upsample = nn.Sequential(
            # Block 1: 2x upsample
            nn.ConvTranspose1d(config.hidden_dim, config.hidden_dim, 4, stride=2, padding=1),
            nn.GroupNorm(8, config.hidden_dim),
            nn.GELU(),

            # Block 2: 2x upsample
            nn.ConvTranspose1d(config.hidden_dim, config.hidden_dim // 2, 4, stride=2, padding=1),
            nn.GroupNorm(4, config.hidden_dim // 2),
            nn.GELU(),

            # Block 3: 2x upsample
            nn.ConvTranspose1d(config.hidden_dim // 2, config.hidden_dim // 4, 4, stride=2, padding=1),
            nn.GroupNorm(2, config.hidden_dim // 4),
            nn.GELU(),

            # Output layer (1 channel for mono audio)
            nn.Conv1d(config.hidden_dim // 4, 1, 3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Decode embedding to audio waveform.

        Args:
            embedding: Input embedding [batch, embed_dim]

        Returns:
            Audio waveform [batch, output_length] normalized to [-1, 1]
        """
        batch_size = embedding.shape[0]

        # Project to initial sequence
        x = self.proj(embedding)  # [batch, hidden_dim * init_length]
        x = x.view(batch_size, self.hidden_dim, self.init_length)

        # Upsample
        x = self.upsample(x)  # [batch, 1, output_length]
        x = x.squeeze(1)  # [batch, output_length]

        # Tanh to [-1, 1]
        x = torch.tanh(x)

        return x

    def forward_bytes(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Decode embedding to PCM bytes.

        Converts waveform to unsigned 8-bit PCM values [0, 255].

        Args:
            embedding: Input embedding [batch, embed_dim]

        Returns:
            PCM bytes [batch, output_length] in [0, 255]
        """
        # Get waveform in [-1, 1]
        waveform = self.forward(embedding)

        # Convert to [0, 255] (8-bit unsigned PCM)
        # [-1, 1] -> [0, 1] -> [0, 255]
        pcm = ((waveform + 1.0) / 2.0 * 255.0)

        return pcm

    def generate(
        self,
        embedding: torch.Tensor,
        return_tensor: bool = False,
    ) -> np.ndarray:
        """
        Generate audio from embedding.

        Args:
            embedding: Input embedding [batch, embed_dim]
            return_tensor: If True, return torch.Tensor

        Returns:
            Audio waveform as numpy array [batch, output_length] float32 in [-1, 1]
            or torch.Tensor if return_tensor=True
        """
        with torch.no_grad():
            waveform = self.forward(embedding)

        if return_tensor:
            return waveform
        else:
            return waveform.cpu().numpy()

    def generate_bytes(
        self,
        embedding: torch.Tensor,
    ) -> bytes:
        """
        Generate audio as raw PCM bytes.

        Args:
            embedding: Input embedding [1, embed_dim] (single sample)

        Returns:
            Raw PCM bytes (8-bit unsigned)
        """
        with torch.no_grad():
            pcm = self.forward_bytes(embedding)  # [1, output_length]

        # Convert to bytes
        pcm = pcm[0].clamp(0, 255).to(torch.uint8)
        return bytes(pcm.cpu().numpy())

    def compute_loss(
        self,
        embedding: torch.Tensor,
        target_bytes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for audio generation.

        Args:
            embedding: Input embedding [batch, embed_dim]
            target_bytes: Target PCM bytes [batch, output_length] in [0, 255]

        Returns:
            MSE loss (scalar)
        """
        # Get predictions
        pred_bytes = self.forward_bytes(embedding)  # [batch, output_length]

        # Normalize to [0, 1] for loss computation
        pred_norm = pred_bytes / 255.0
        target_norm = target_bytes.float() / 255.0

        # MSE loss
        loss = F.mse_loss(pred_norm, target_norm)

        return loss

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, "
            f"output_length={self.output_length}, "
            f"sample_rate={self.sample_rate}"
        )
