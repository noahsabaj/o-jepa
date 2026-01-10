"""
Byte Encoder for byte-level O-JEPA.

Unified encoder that processes raw bytes (0-255) from any modality
into a shared embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import ByteEncoderConfig
from .layers import RMSNorm, LearnedPositionalEmbedding
from .transformer_block import TransformerBlock


# =============================================================================
# HIERARCHICAL CHUNKING (Loom-inspired)
# =============================================================================

class HierarchicalChunking(nn.Module):
    """
    Multi-scale byte chunking with learned content-dependent gating.

    Inspired by Loom's hierarchical chunking: processes bytes at multiple
    scales (e.g., 4, 16, 64 bytes) and combines using learned gating.

    Architecture:
        Input embeddings [B, L, D] ->
        Scale 1 (k=4):  Conv1d -> features_1
        Scale 2 (k=16): Conv1d -> features_2
        Scale 3 (k=64): Conv1d -> features_3
        Gating: softmax(gate_proj(input)) * [f1, f2, f3]
        Output: weighted sum [B, L, D]

    This allows the model to learn patterns at different granularities:
    - 4 bytes: UTF-8 characters, small patterns
    - 16 bytes: Words, small code tokens
    - 64 bytes: Sentences, function signatures

    Args:
        input_dim: Input embedding dimension
        output_dim: Output dimension (usually same as input)
        scales: Tuple of kernel sizes for multi-scale convolutions
        gating: Gating type ("softmax", "sigmoid", "linear")
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        scales: Tuple[int, ...] = (4, 16, 64),
        gating: str = "softmax",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scales = scales
        self.num_scales = len(scales)
        self.gating_type = gating

        # Multi-scale 1D convolutions (causal padding: pad left only)
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=k,
                padding=k - 1,  # Causal: pad left side
                groups=1,
            )
            for k in scales
        ])

        # Content-dependent gating
        # Projects input to num_scales weights per position
        self.gate_proj = nn.Linear(input_dim, self.num_scales, bias=False)

        # Output projection (combines scales)
        self.out_proj = nn.Linear(output_dim, output_dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for conv in self.scale_convs:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='linear')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply hierarchical multi-scale processing.

        Args:
            x: Input embeddings [batch, seq_len, input_dim]

        Returns:
            Multi-scale features [batch, seq_len, output_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Compute gating weights from input content
        # [batch, seq_len, num_scales]
        gate_logits = self.gate_proj(x)

        if self.gating_type == "softmax":
            gate_weights = F.softmax(gate_logits, dim=-1)
        elif self.gating_type == "sigmoid":
            gate_weights = torch.sigmoid(gate_logits)
        else:  # linear
            gate_weights = gate_logits

        # Apply convolutions at each scale
        # Transpose for Conv1d: [batch, dim, seq_len]
        x_t = x.transpose(1, 2)

        scale_features = []
        for conv in self.scale_convs:
            # Apply conv and trim to original length (remove right padding for causal)
            conv_out = conv(x_t)[:, :, :seq_len]
            scale_features.append(conv_out.transpose(1, 2))  # Back to [B, L, D]

        # Stack scales: [batch, seq_len, num_scales, output_dim]
        stacked = torch.stack(scale_features, dim=2)

        # Weighted combination using gating
        # gate_weights: [B, L, num_scales] -> [B, L, num_scales, 1]
        weighted = stacked * gate_weights.unsqueeze(-1)

        # Sum over scales: [batch, seq_len, output_dim]
        combined = weighted.sum(dim=2)

        # Output projection
        output = self.out_proj(combined)
        output = self.dropout(output)

        return output

    def extra_repr(self) -> str:
        return (
            f"scales={self.scales}, "
            f"gating={self.gating_type}, "
            f"input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}"
        )


# =============================================================================
# BYTE ENCODER
# =============================================================================

class ByteEncoder(nn.Module):
    """
    Unified byte encoder for all modalities with hierarchical multi-scale processing.

    Takes raw bytes (values 0-255) and produces embeddings.
    Works for any modality: vision (RGB bytes), text (UTF-8), audio (PCM).

    Architecture:
    1. Byte embedding (256-entry lookup table)
    2. Positional embedding (learned)
    3. Hierarchical multi-scale processing (Loom-inspired) - ALWAYS ON
    4. Lightweight transformer blocks for local context
    5. Output normalization

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

        # Hierarchical multi-scale processing (before transformer)
        self.hierarchical = HierarchicalChunking(
            input_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            scales=config.hierarchical_scales,
            gating=config.hierarchical_gating,
            dropout=config.dropout,
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

        # Apply hierarchical multi-scale processing (before transformer)
        hierarchical_features = self.hierarchical(x)
        x = x + hierarchical_features  # Residual connection

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        # Final normalization
        x = self.norm(x)

        return x

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
