"""
ByteJEPA Predictor: Cross-Attention Predictor for Byte-Level JEPA

Predicts target embeddings from context using cross-attention.
Following I-JEPA's design: predictor must learn position through attention.

Key insight: Predictor capacity should be limited to force encoder to learn.

DESIGN: NO EXPLICIT POSITION EMBEDDINGS (I-JEPA Style)
======================================================
Following Y1 recommendation from Yann LeCun review:

The predictor receives NO position information about which tokens to predict.
Position must be inferred implicitly through cross-attention to context:
- Context tokens have position information (from encoder's position embeddings)
- Query tokens (mask tokens) attend to context to learn "where" to predict
- This forces the encoder to embed position information in its representations

Why this matters:
    1. Pure I-JEPA: This matches the original I-JEPA formulation
    2. Better representations: Encoder must learn position-aware features
    3. No shortcut: Predictor cannot "cheat" by using explicit position

BREAKING CHANGE (v0.6.0):
    Models trained with explicit position embeddings (v0.5.x) are NOT
    compatible with this version. Retraining is required.

Note:
    While the predictor still uses query self-attention (which I-JEPA doesn't),
    this is a separate design choice that can be disabled via config (Y4).
"""

from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import PredictorConfig, PREDICTOR_MLP_ALIGNMENT, SWIGLU_HIDDEN_RATIO
from .layers import (
    CrossAttention,
    MultiHeadAttention,
    SwiGLU,
    RMSNorm,
)


class ByteJEPAPredictorBlock(nn.Module):
    """
    ByteJEPA predictor block with self-attention and cross-attention.

    Architecture:
        1. Self-attention among mask queries
        2. Cross-attention to context embeddings
        3. FFN

    Note: Named ByteJEPA to distinguish from I-JEPA which doesn't use
    self-attention among query tokens.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        context_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_self_attention: bool = True,  # Y4: Optional query self-attention
    ):
        super().__init__()
        self.use_self_attention = use_self_attention

        # Self-attention among queries (optional - Y4)
        if use_self_attention:
            self.norm1 = RMSNorm(dim)
            self.self_attn = MultiHeadAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                use_rope=False,
            )
        else:
            self.norm1 = None
            self.self_attn = None

        # Cross-attention to context
        self.norm2 = RMSNorm(dim)
        self.cross_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # For cross-attention, project context if different dim
        self.context_proj = None
        if context_dim != dim:
            self.context_proj = nn.Linear(context_dim, dim, bias=False)
            self.context_norm = RMSNorm(context_dim)

        # FFN (aligned for tensor core efficiency)
        self.norm3 = RMSNorm(dim)
        mlp_dim = int(dim * mlp_ratio * SWIGLU_HIDDEN_RATIO)
        mlp_dim = ((mlp_dim + PREDICTOR_MLP_ALIGNMENT - 1) // PREDICTOR_MLP_ALIGNMENT) * PREDICTOR_MLP_ALIGNMENT
        self.ffn = SwiGLU(dim=dim, hidden_dim=mlp_dim, dropout=dropout)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process predictor block.

        Args:
            queries: Mask queries [batch, num_targets, dim]
            context: Context embeddings [batch, context_len, context_dim]
            context_mask: Which context positions are valid [batch, context_len]

        Returns:
            Updated queries [batch, num_targets, dim]
        """
        # Y4: Self-attention among queries (optional)
        if self.use_self_attention:
            queries = queries + self.self_attn(self.norm1(queries))

        # Project context if needed
        if self.context_proj is not None:
            ctx = self.context_proj(self.context_norm(context))
        else:
            ctx = context

        # Cross-attention to context
        queries = queries + self.cross_attn(self.norm2(queries), ctx, context_mask)

        # FFN
        queries = queries + self.ffn(self.norm3(queries))

        return queries


class ByteJEPAPredictor(nn.Module):
    """
    ByteJEPA World Model Predictor.

    Takes context embeddings and target positions, predicts target embeddings.

    Architecture (inspired by I-JEPA, adapted for bytes):
        1. Learnable mask token (query for each target)
        2. Add position embeddings to queries (differs from I-JEPA)
        3. Cross-attention blocks: queries attend to context
        4. Project to output dimension

    Key design choices:
        - Narrow architecture (predictor_dim = encoder_dim // 2)
        - Limited depth (forces encoder to do heavy lifting)
        - Explicit position embeddings (pragmatic for long byte sequences)

    Note:
        Named ByteJEPAPredictor to clarify this is a byte-level variant,
        not a direct I-JEPA implementation. See module docstring for details.

    Args:
        config: PredictorConfig with hyperparameters
    """

    def __init__(self, config: PredictorConfig):
        super().__init__()
        self.config = config

        # Predictor dimension (narrower than encoder, configurable via predictor_ratio)
        self.predictor_dim = config.predictor_dim
        self.output_dim = config.output_dim
        self.context_dim = config.hidden_dim

        # Learnable mask token (will be expanded for each target position)
        # Y1: NO position embeddings - pure I-JEPA style
        # Position is inferred through cross-attention to context tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Input projection (context -> predictor dim)
        self.input_proj = nn.Linear(self.context_dim, self.predictor_dim, bias=False)
        self.input_norm = RMSNorm(self.context_dim)

        # Predictor blocks
        # Y4: use_query_self_attention controls whether queries attend to each other
        self.blocks = nn.ModuleList([
            ByteJEPAPredictorBlock(
                dim=self.predictor_dim,
                num_heads=max(1, config.num_heads // 2),
                context_dim=self.predictor_dim,  # After input_proj
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                use_self_attention=config.use_query_self_attention,
            )
            for _ in range(config.num_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(self.predictor_dim)

        # Output projection to match encoder output dimension
        self.output_proj = nn.Linear(self.predictor_dim, self.output_dim, bias=False)

        # Gradient checkpointing
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.output_proj.weight, std=0.02)

    def forward(
        self,
        context_emb: torch.Tensor,
        target_positions: List[torch.Tensor],
        context_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict target embeddings from context.

        Args:
            context_emb: Context embeddings from encoder [batch, context_len, hidden_dim]
            target_positions: List of target position indices per batch item
            context_mask: Which context positions are valid [batch, context_len]

        Returns:
            predictions: Predicted target embeddings [batch, max_targets, output_dim]
            pred_mask: Which predictions are valid [batch, max_targets]
        """
        batch_size = context_emb.shape[0]
        device = context_emb.device

        # Project context to predictor dimension
        context = self.input_proj(self.input_norm(context_emb))

        # Find max number of targets for padding
        max_targets = max(len(pos) for pos in target_positions)

        # Create queries: expand mask token for all targets
        # Y1: NO position embeddings added - pure I-JEPA style
        # Position is learned through cross-attention to context tokens
        queries = self.mask_token.expand(batch_size, max_targets, -1).clone()

        # Create validity mask for predictions
        pred_mask = torch.zeros(batch_size, max_targets, dtype=torch.bool, device=device)
        for b in range(batch_size):
            num_targets = len(target_positions[b])
            pred_mask[b, :num_targets] = True

        # Apply predictor blocks
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                queries = checkpoint(block, queries, context, context_mask, use_reentrant=False)
            else:
                queries = block(queries, context, context_mask)

        # Final norm and projection
        queries = self.norm(queries)
        predictions = self.output_proj(queries)

        return predictions, pred_mask

    def predict_single(
        self,
        context_emb: torch.Tensor,
        position: int,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict embedding for a single position.

        Convenience method for inference.

        Args:
            context_emb: Context [batch, context_len, hidden_dim]
            position: Target position index
            context_mask: Optional mask

        Returns:
            Predicted embedding [batch, output_dim]
        """
        batch_size = context_emb.shape[0]
        target_positions = [torch.tensor([position]) for _ in range(batch_size)]
        predictions, _ = self.forward(context_emb, target_positions, context_mask)
        return predictions[:, 0, :]

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"predictor_dim={self.predictor_dim}, "
            f"output_dim={self.output_dim}, "
            f"num_blocks={len(self.blocks)}"
        )


# =============================================================================
# BACKWARDS COMPATIBILITY ALIASES
# =============================================================================
# These aliases maintain backwards compatibility with code using old names.
# New code should use ByteJEPAPredictor and ByteJEPAPredictorBlock.

JEPAPredictor = ByteJEPAPredictor
JEPAPredictorBlock = ByteJEPAPredictorBlock
