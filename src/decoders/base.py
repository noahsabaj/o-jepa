"""
Base decoder class for byte-level O-JEPA.

All modality-specific decoders inherit from this abstract base class.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseDecoder(nn.Module, ABC):
    """
    Abstract base class for modality decoders.

    Decoders take embeddings from the predictor and generate
    modality-specific outputs (text bytes, image pixels, audio samples).

    All decoders must implement:
    - forward(): Compute loss/logits during training
    - generate(): Generate full output during inference
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize base decoder.

        Args:
            input_dim: Dimension of input embeddings (from predictor)
            hidden_dim: Hidden dimension for decoder layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    @abstractmethod
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Decode embedding to modality-specific output.

        Args:
            embedding: Input embedding [batch, embed_dim]

        Returns:
            Logits or predictions depending on modality
        """
        pass

    @abstractmethod
    def generate(self, embedding: torch.Tensor, **kwargs) -> Any:
        """
        Generate full output from embedding.

        Args:
            embedding: Input embedding [batch, embed_dim]
            **kwargs: Modality-specific generation parameters

        Returns:
            Generated output (format depends on modality)
        """
        pass

    def get_num_params(self) -> int:
        """Count parameters in the decoder."""
        return sum(p.numel() for p in self.parameters())
