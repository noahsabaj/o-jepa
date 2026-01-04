"""
Byte Encoder for byte-level O-JEPA.

Unified encoder that processes raw bytes (0-255) from any modality
into a shared embedding space.
"""

import torch
import torch.nn as nn
from typing import Optional

from .config import ByteEncoderConfig
from .layers import (
    MultiHeadAttention,
    SwiGLU,
    RMSNorm,
    LearnedPositionalEmbedding,
)


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-normalization.

    Uses:
    - Pre-norm architecture (norm before attention/ffn)
    - Multi-head self-attention with RoPE
    - SwiGLU feedforward network
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        use_bias: bool = False,
        max_seq_len: int = 8192,
    ):
        super().__init__()

        # Pre-normalization
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        # Attention
        self.attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=use_bias,
            use_rope=True,
            max_seq_len=max_seq_len,
        )

        # Feedforward (SwiGLU)
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


class ByteEncoder(nn.Module):
    """
    Unified byte encoder for all modalities.

    Takes raw bytes (values 0-255) and produces embeddings.
    Works for any modality: vision (RGB bytes), text (UTF-8), audio (PCM).

    Architecture:
    1. Byte embedding (256-entry lookup table)
    2. Positional embedding (learned)
    3. Lightweight transformer blocks for local context
    4. Output normalization

    Args:
        config: ByteEncoderConfig with model hyperparameters
    """

    def __init__(self, config: ByteEncoderConfig):
        super().__init__()

        self.config = config
        self.hidden_dim = config.hidden_dim

        # Byte embedding table (all 256 possible byte values)
        self.byte_embed = nn.Embedding(
            num_embeddings=config.vocab_size,  # 256
            embedding_dim=config.hidden_dim,
        )

        # Learned positional embeddings
        self.pos_embed = LearnedPositionalEmbedding(
            max_seq_len=config.max_seq_len,
            dim=config.hidden_dim,
        )

        # Transformer blocks for local context extraction
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.hidden_dim,
                num_heads=config.num_heads,
                mlp_dim=config.mlp_dim,
                dropout=config.dropout,
                use_bias=config.use_bias,
                max_seq_len=config.max_seq_len,
            )
            for _ in range(config.num_layers)
        ])

        # Output normalization
        self.norm = RMSNorm(config.hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.trunc_normal_(self.byte_embed.weight, std=0.02)

    def forward(
        self,
        byte_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode raw bytes to embeddings.

        Args:
            byte_ids: Tensor of byte values [batch, seq_len] with values in [0, 255]
            attention_mask: Optional mask [batch, seq_len] where True = attend

        Returns:
            Embeddings [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len = byte_ids.shape

        # Validate input range
        if self.training:
            assert byte_ids.min() >= 0 and byte_ids.max() <= 255, \
                f"Byte values must be in [0, 255], got [{byte_ids.min()}, {byte_ids.max()}]"

        # Embed bytes
        x = self.byte_embed(byte_ids)  # [batch, seq_len, hidden_dim]

        # Add positional embeddings
        pos_emb = self.pos_embed(seq_len)  # [1, seq_len, hidden_dim]
        x = x + pos_emb

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        # Final normalization
        x = self.norm(x)

        return x

    def encode_vision(
        self,
        image_bytes: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode raw image bytes (RGB).

        Args:
            image_bytes: Raw RGB bytes [batch, H*W*3] with values in [0, 255]
            attention_mask: Optional mask

        Returns:
            Image embeddings [batch, H*W*3, hidden_dim]
        """
        return self.forward(image_bytes, attention_mask)

    def encode_text(
        self,
        text_bytes: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode UTF-8 text bytes.

        Args:
            text_bytes: UTF-8 bytes [batch, seq_len] with values in [0, 255]
            attention_mask: Optional mask for padding

        Returns:
            Text embeddings [batch, seq_len, hidden_dim]
        """
        return self.forward(text_bytes, attention_mask)

    def encode_audio(
        self,
        audio_bytes: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode raw PCM audio bytes.

        Args:
            audio_bytes: PCM bytes [batch, num_bytes] with values in [0, 255]
            attention_mask: Optional mask

        Returns:
            Audio embeddings [batch, num_bytes, hidden_dim]
        """
        return self.forward(audio_bytes, attention_mask)

    def get_output_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.hidden_dim

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.config.vocab_size}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.config.num_layers}, "
            f"num_heads={self.config.num_heads}"
        )
