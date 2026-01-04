"""
JEPA World Model Predictor

Predicts target embeddings from context using cross-attention.
Following I-JEPA: narrow predictor, learnable mask tokens, position-aware.

Key insight: Predictor capacity should be limited to force encoder to learn.
"""

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import PredictorConfig, PREDICTOR_MLP_ALIGNMENT
from .layers import (
    CrossAttention,
    MultiHeadAttention,
    SwiGLU,
    RMSNorm,
    LearnedPositionalEmbedding,
)


class JEPAPredictorBlock(nn.Module):
    """
    Predictor block with self-attention and cross-attention.

    Architecture:
        1. Self-attention among mask queries
        2. Cross-attention to context embeddings
        3. FFN
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        context_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Self-attention among queries
        self.norm1 = RMSNorm(dim)
        self.self_attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=False,  # Using learned position embeddings instead
        )

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
        mlp_dim = int(dim * mlp_ratio * 2 / 3)
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
        # Self-attention among queries
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


class JEPAPredictor(nn.Module):
    """
    JEPA World Model Predictor.

    Takes context embeddings and target positions, predicts target embeddings.

    Architecture (following I-JEPA):
        1. Learnable mask token (query for each target)
        2. Add position embeddings to queries
        3. Cross-attention blocks: queries attend to context
        4. Project to output dimension

    Key design choices:
        - Narrow architecture (predictor_dim = encoder_dim // 2)
        - Limited depth (forces encoder to do heavy lifting)
        - Position-aware predictions

    Args:
        config: PredictorConfig with hyperparameters
    """

    def __init__(self, config: PredictorConfig):
        super().__init__()
        self.config = config

        # Predictor dimension (narrower than encoder)
        self.predictor_dim = config.hidden_dim // 2
        self.output_dim = config.output_dim
        self.context_dim = config.hidden_dim

        # Learnable mask token (will be expanded for each target position)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Position embeddings for target positions
        self.pos_embed = LearnedPositionalEmbedding(
            max_seq_len=config.max_seq_len,
            dim=self.predictor_dim,
        )

        # Input projection (context -> predictor dim)
        self.input_proj = nn.Linear(self.context_dim, self.predictor_dim, bias=False)
        self.input_norm = RMSNorm(self.context_dim)

        # Predictor blocks
        self.blocks = nn.ModuleList([
            JEPAPredictorBlock(
                dim=self.predictor_dim,
                num_heads=max(1, config.num_heads // 2),
                context_dim=self.predictor_dim,  # After input_proj
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
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
    ) -> torch.Tensor:
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

        # Create queries: expand mask token + add position embeddings
        queries = self.mask_token.expand(batch_size, max_targets, -1).clone()

        # Create validity mask for predictions
        pred_mask = torch.zeros(batch_size, max_targets, dtype=torch.bool, device=device)

        # Add position embeddings for each target
        for b in range(batch_size):
            num_targets = len(target_positions[b])
            pred_mask[b, :num_targets] = True

            if num_targets > 0:
                pos_indices = target_positions[b].to(device)
                pos_emb = self.pos_embed(pos_indices)
                queries[b, :num_targets] = queries[b, :num_targets] + pos_emb

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


class SimplePredictor(nn.Module):
    """
    Simple MLP predictor without cross-attention.

    For ablation studies or simpler use cases.
    """

    def __init__(self, config: PredictorConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim

        # Simple MLP: context + position -> prediction
        self.proj = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.output_dim, bias=False),
        )

        # Position embedding
        self.pos_embed = LearnedPositionalEmbedding(
            max_seq_len=config.max_seq_len,
            dim=config.hidden_dim,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(
        self,
        context_emb: torch.Tensor,
        target_positions: List[torch.Tensor],
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict target embeddings."""
        batch_size = context_emb.shape[0]
        device = context_emb.device

        max_targets = max(len(pos) for pos in target_positions)
        predictions = torch.zeros(batch_size, max_targets, self.output_dim, device=device)
        pred_mask = torch.zeros(batch_size, max_targets, dtype=torch.bool, device=device)

        # Pool context (simple mean)
        if context_mask is not None:
            mask = context_mask.unsqueeze(-1)
            context_pooled = (context_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            context_pooled = context_emb.mean(dim=1)

        for b in range(batch_size):
            num_targets = len(target_positions[b])
            pred_mask[b, :num_targets] = True

            if num_targets > 0:
                pos_indices = target_positions[b].to(device)
                pos_emb = self.pos_embed(pos_indices)

                # Expand context for each target
                ctx = context_pooled[b:b+1].expand(num_targets, -1)

                # Concatenate context and position, project
                combined = torch.cat([ctx, pos_emb], dim=-1)
                predictions[b, :num_targets] = self.proj(combined)

        return predictions, pred_mask

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
