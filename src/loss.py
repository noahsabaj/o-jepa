"""
JEPA World Model Loss Functions

Non-contrastive prediction loss in latent space.
Following LeCun: predict abstract representations, not pixels.

Key insight: No negative samples needed. Energy = prediction error.
"""

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig as JEPALossConfig


class JEPALoss(nn.Module):
    """
    JEPA World Model Loss: Non-contrastive prediction in latent space.

    Core equation:
        L = ||predictor(context) - sg(target_encoder(targets))||

    Where sg = stop_gradient (target encoder is frozen/EMA).

    No negative samples. Energy-based: low loss = consistent prediction.
    """

    def __init__(self, config: Optional[JEPALossConfig] = None):
        super().__init__()
        self.config = config or JEPALossConfig()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute JEPA prediction loss.

        Args:
            predictions: Predicted embeddings [batch, num_targets, dim] or [batch, dim]
            targets: Target embeddings (already detached) [batch, num_targets, dim] or [batch, dim]
            mask: Optional mask for valid positions [batch, num_targets]

        Returns:
            loss: Scalar loss value
            metrics: Dictionary with loss components
        """
        # Ensure targets are detached (should be from EMA encoder)
        targets = targets.detach()

        # Optional normalization
        if self.config.normalize_predictions:
            predictions = F.normalize(predictions, p=2, dim=-1)
        if self.config.normalize_targets:
            targets = F.normalize(targets, p=2, dim=-1)

        # Core prediction loss
        if self.config.loss_type == "mse":
            pred_loss = F.mse_loss(predictions, targets, reduction="none")
        else:  # smooth_l1
            pred_loss = F.smooth_l1_loss(
                predictions, targets,
                beta=self.config.beta,
                reduction="none"
            )

        # Apply mask if provided
        if mask is not None:
            pred_loss = pred_loss * mask.unsqueeze(-1)
            pred_loss = pred_loss.sum() / (mask.sum() * predictions.shape[-1] + 1e-8)
        else:
            pred_loss = pred_loss.mean()

        # Variance regularization (prevents embedding collapse)
        var_loss = torch.tensor(0.0, device=predictions.device)
        std_mean = 0.0

        if self.config.use_variance_loss:
            # Flatten to [total_predictions, dim]
            pred_flat = predictions.reshape(-1, predictions.shape[-1])

            if pred_flat.shape[0] > 1:
                std = pred_flat.std(dim=0)
                std_mean = std.mean().item()
                # Penalize if std drops below target
                var_loss = F.relu(self.config.variance_target - std).mean()

        # Combined loss
        total_loss = pred_loss + self.config.variance_weight * var_loss

        metrics = {
            "loss": total_loss.item(),
            "pred_loss": pred_loss.item(),
            "var_loss": var_loss.item() if isinstance(var_loss, torch.Tensor) else var_loss,
            "std_mean": std_mean,
        }

        return total_loss, metrics


class SmoothL1Loss(nn.Module):
    """Simple smooth L1 loss wrapper for compatibility."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        targets = targets.detach()
        loss = F.smooth_l1_loss(predictions, targets, beta=self.beta)
        return loss, {"loss": loss.item()}


class MSELoss(nn.Module):
    """Simple MSE loss wrapper for compatibility."""

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        targets = targets.detach()
        loss = F.mse_loss(predictions, targets)
        return loss, {"loss": loss.item()}


def compute_energy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute energy (prediction error) for world model evaluation.

    Energy-based interpretation:
    - Low energy = prediction matches target = consistent with world model
    - High energy = prediction differs = inconsistent/novel input

    Args:
        predictions: Predicted embeddings [batch, dim]
        targets: Actual embeddings [batch, dim]

    Returns:
        energy: Per-sample energy [batch]
    """
    return F.mse_loss(predictions, targets, reduction="none").mean(dim=-1)


def compute_prediction_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute metrics for prediction quality.

    Args:
        predictions: Predicted embeddings [batch, *, dim]
        targets: Target embeddings [batch, *, dim]

    Returns:
        Dictionary with prediction metrics
    """
    # Flatten for metrics
    pred_flat = predictions.reshape(-1, predictions.shape[-1])
    target_flat = targets.reshape(-1, targets.shape[-1])

    # MSE
    mse = F.mse_loss(pred_flat, target_flat).item()

    # Cosine similarity
    pred_norm = F.normalize(pred_flat, p=2, dim=-1)
    target_norm = F.normalize(target_flat, p=2, dim=-1)
    cosine_sim = (pred_norm * target_norm).sum(dim=-1).mean().item()

    # Relative error
    target_norm_val = target_flat.norm(dim=-1).mean().item()
    rel_error = mse / (target_norm_val ** 2 + 1e-8)

    # Variance of predictions (collapse detection)
    pred_std = pred_flat.std(dim=0).mean().item()
    target_std = target_flat.std(dim=0).mean().item()

    return {
        "mse": mse,
        "cosine_sim": cosine_sim,
        "rel_error": rel_error,
        "pred_std": pred_std,
        "target_std": target_std,
        "std_ratio": pred_std / (target_std + 1e-8),
    }


