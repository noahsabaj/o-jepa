"""
Byte-Level O-JEPA Shared Backbone

A unified transformer backbone that processes embeddings from any modality.
Uses learnable modality tokens to identify which modality is being processed.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .config import BackboneConfig, BACKBONE_MLP_ALIGNMENT
from .layers import (
    MultiHeadAttention,
    SwiGLU,
    RMSNorm,
)


class BackboneTransformerBlock(nn.Module):
    """
    Transformer block for the shared backbone.

    Uses pre-normalization architecture:
    - RMSNorm
    - Multi-head self-attention with RoPE
    - SwiGLU feedforward network
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_bias: bool = False,
        max_seq_len: int = 8192,
    ):
        super().__init__()

        # Pre-normalization layers
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        # Attention with RoPE
        self.attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=use_bias,
            use_rope=True,
            max_seq_len=max_seq_len,
        )

        # SwiGLU feedforward (aligned for tensor core efficiency)
        mlp_dim = int(dim * mlp_ratio * 2 / 3)
        mlp_dim = ((mlp_dim + BACKBONE_MLP_ALIGNMENT - 1) // BACKBONE_MLP_ALIGNMENT) * BACKBONE_MLP_ALIGNMENT
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
            attention_mask: Optional mask [batch, seq_len]

        Returns:
            Output tensor [batch, seq_len, dim]
        """
        # Attention with residual
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x


class SharedBackbone(nn.Module):
    """
    Shared transformer backbone for all modalities in byte-level O-JEPA.

    Architecture:
        1. Prepend learnable modality token to input sequence
        2. Process through N transformer blocks
        3. Return contextualized tokens (including modality token)

    The same weights process all modalities, enabling:
        - Memory efficiency (one backbone instead of N encoders)
        - Cross-modal knowledge transfer
        - Emergent multimodal alignment

    Key design: Modality tokens tell the backbone which modality it's
    processing, similar to task prompts in instruction-tuned models.

    Args:
        config: BackboneConfig with model hyperparameters
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

        # Learnable modality tokens (one per modality)
        # These are prepended to the input and help the model identify modality
        self.modality_tokens = nn.ParameterDict({
            modality: nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
            for modality in config.modalities
        })

        # Determine max sequence length from config or use default
        max_seq_len = getattr(config, 'max_seq_len', 8192)

        # Transformer layers
        self.layers = nn.ModuleList([
            BackboneTransformerBlock(
                dim=config.hidden_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                use_bias=config.use_bias,
                max_seq_len=max_seq_len,
            )
            for _ in range(config.num_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.hidden_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize linear layers with truncated normal."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def supported_modalities(self) -> Tuple[str, ...]:
        """Return tuple of supported modalities."""
        return tuple(self.modality_tokens.keys())

    def forward(
        self,
        embeddings: torch.Tensor,
        modality: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process embeddings through the shared backbone.

        Args:
            embeddings: Input embeddings from ByteEncoder [batch, seq_len, hidden_dim]
            modality: Name of the modality ('vision', 'text', or 'audio')
            attention_mask: Optional attention mask [batch, seq_len]

        Returns:
            Tuple of:
                - Sequence embeddings [batch, seq_len, hidden_dim] (without modality token)
                - Pooled embedding [batch, hidden_dim] (the modality token)
        """
        if modality not in self.modality_tokens:
            raise ValueError(
                f"Unknown modality: {modality}. "
                f"Supported: {self.supported_modalities}"
            )

        batch_size = embeddings.shape[0]

        # Get modality token and expand for batch
        mod_token = self.modality_tokens[modality]  # [1, 1, hidden_dim]
        mod_token = mod_token.expand(batch_size, -1, -1)  # [batch, 1, hidden_dim]

        # Prepend modality token to sequence
        x = torch.cat([mod_token, embeddings], dim=1)  # [batch, seq_len + 1, hidden_dim]

        # Extend attention mask if provided
        if attention_mask is not None:
            # Add 1 for modality token (always attended)
            mod_mask = torch.ones(
                batch_size, 1,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attention_mask = torch.cat([mod_mask, attention_mask], dim=1)

        # Apply transformer layers
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, attention_mask, use_reentrant=False)
            else:
                x = layer(x, attention_mask)

        # Final normalization
        x = self.norm(x)

        # Split into pooled (modality token) and sequence embeddings
        pooled = x[:, 0, :]  # [batch, hidden_dim]
        sequence = x[:, 1:, :]  # [batch, seq_len, hidden_dim]

        return sequence, pooled

    def get_modality_embedding(
        self,
        embeddings: torch.Tensor,
        modality: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get the modality token embedding after processing.

        This is useful for getting a single embedding per sample,
        which can be used for cross-modal prediction.

        Args:
            embeddings: Input embeddings from ByteEncoder [batch, seq_len, hidden_dim]
            modality: Name of the modality
            attention_mask: Optional attention mask [batch, seq_len]

        Returns:
            Modality token embedding [batch, hidden_dim]
        """
        _, pooled = self.forward(embeddings, modality, attention_mask)
        return pooled

    def get_sequence_embeddings(
        self,
        embeddings: torch.Tensor,
        modality: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get the sequence embeddings after processing (excluding modality token).

        Args:
            embeddings: Input embeddings from ByteEncoder [batch, seq_len, hidden_dim]
            modality: Name of the modality
            attention_mask: Optional attention mask [batch, seq_len]

        Returns:
            Sequence embeddings [batch, seq_len, hidden_dim]
        """
        sequence, _ = self.forward(embeddings, modality, attention_mask)
        return sequence

    def get_num_params(self) -> int:
        """Count parameters in the backbone."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={len(self.layers)}, "
            f"modalities={self.supported_modalities}"
        )
