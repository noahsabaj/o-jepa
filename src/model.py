"""
JEPA World Model: Byte-Level Joint Embedding Predictive Architecture

Following Yann LeCun's vision: predict abstract representations, not pixels.
The model learns to understand the world by predicting masked region embeddings.

Key insight: Low prediction error = consistent with world model.
"""

from typing import Dict, Optional, Tuple, Any, List
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ByteJEPAConfig, get_default_config, get_tiny_config
from .byte_encoder import ByteEncoder
from .backbone import SharedBackbone
from .predictor import JEPAPredictor
from .masking import SpanMaskGenerator, BlockMaskGenerator, MaskingConfig
from .loss import JEPALoss, compute_energy, compute_prediction_metrics


class TargetEncoder(nn.Module):
    """
    EMA Target Encoder for JEPA.

    Produces stable prediction targets. Updated via exponential moving average.
    Never receives gradients directly - only through EMA update.
    """

    def __init__(
        self,
        byte_encoder: ByteEncoder,
        backbone: SharedBackbone,
        ema_decay_start: float = 0.996,
        ema_decay_end: float = 0.9999,
        ema_warmup_steps: int = 10000,
    ):
        super().__init__()

        # Deep copy encoder components
        self.byte_encoder = copy.deepcopy(byte_encoder)
        self.backbone = copy.deepcopy(backbone)

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # EMA schedule
        self.ema_decay_start = ema_decay_start
        self.ema_decay_end = ema_decay_end
        self.ema_warmup_steps = ema_warmup_steps
        self.current_step = 0

    @property
    def ema_decay(self) -> float:
        """Get current EMA decay value (scheduled)."""
        if self.ema_warmup_steps <= 0:
            return self.ema_decay_end
        progress = min(self.current_step / self.ema_warmup_steps, 1.0)
        return self.ema_decay_start + progress * (self.ema_decay_end - self.ema_decay_start)

    @torch.no_grad()
    def update(self, online_byte_encoder: nn.Module, online_backbone: nn.Module):
        """Update target encoder parameters via EMA."""
        decay = self.ema_decay
        self.current_step += 1

        # Update byte_encoder
        for target_param, online_param in zip(
            self.byte_encoder.parameters(), online_byte_encoder.parameters()
        ):
            target_param.data.lerp_(online_param.data, 1.0 - decay)

        # Update backbone
        for target_param, online_param in zip(
            self.backbone.parameters(), online_backbone.parameters()
        ):
            target_param.data.lerp_(online_param.data, 1.0 - decay)

    @torch.no_grad()
    def forward(
        self,
        byte_ids: torch.Tensor,
        modality: str,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode bytes to target embeddings.

        Args:
            byte_ids: Raw bytes [batch, seq_len]
            modality: Modality name
            mask: Which positions to encode [batch, seq_len] (True = encode)

        Returns:
            Target embeddings [batch, seq_len, hidden_dim]
        """
        # Encode all bytes
        byte_emb = self.byte_encoder(byte_ids, attention_mask=None)

        # Process through backbone
        sequence, _ = self.backbone(byte_emb, modality, attention_mask=None)

        return sequence

    def state_dict_ema(self) -> dict:
        """Get state dict including EMA metadata."""
        return {
            "byte_encoder": self.byte_encoder.state_dict(),
            "backbone": self.backbone.state_dict(),
            "current_step": self.current_step,
            "ema_decay_start": self.ema_decay_start,
            "ema_decay_end": self.ema_decay_end,
            "ema_warmup_steps": self.ema_warmup_steps,
        }

    def load_state_dict_ema(self, state_dict: dict):
        """Load state dict including EMA metadata."""
        self.byte_encoder.load_state_dict(state_dict["byte_encoder"])
        self.backbone.load_state_dict(state_dict["backbone"])
        self.current_step = state_dict["current_step"]
        self.ema_decay_start = state_dict.get("ema_decay_start", self.ema_decay_start)
        self.ema_decay_end = state_dict.get("ema_decay_end", self.ema_decay_end)
        self.ema_warmup_steps = state_dict.get("ema_warmup_steps", self.ema_warmup_steps)


class JEPAWorldModel(nn.Module):
    """
    JEPA World Model: Learns to predict the world.

    Architecture:
        Input bytes -> [MASK] -> Context | Targets
                          |           |
                     Online Encoder   Target Encoder (EMA)
                          |           |
                     Context Emb   Target Emb
                          |           |
                     Predictor ------>|
                          |           |
                     Predictions      |
                          |           |
                     MSE Loss <-------

    Training:
        1. Mask random regions of input
        2. Encode visible context (online encoder)
        3. Encode masked targets (EMA target encoder, no grad)
        4. Predict target embeddings from context
        5. Loss = MSE(predictions, targets)

    World Model capabilities:
        - predict_future(): Given state, predict future embeddings
        - evaluate_action(): Compute energy of a state-action pair
    """

    def __init__(self, config: ByteJEPAConfig):
        super().__init__()
        self.config = config

        # Online encoder (receives gradients)
        self.byte_encoder = ByteEncoder(config.byte_encoder)
        self.backbone = SharedBackbone(config.backbone)

        # Predictor (cross-attention to context)
        self.predictor = JEPAPredictor(config.predictor)

        # Target encoder (EMA copy, frozen)
        self.target_encoder = TargetEncoder(
            byte_encoder=self.byte_encoder,
            backbone=self.backbone,
            ema_decay_start=config.ema.ema_decay_start,
            ema_decay_end=config.ema.ema_decay_end,
            ema_warmup_steps=config.ema.ema_warmup_steps,
        )

        # Mask generator
        masking_config = MaskingConfig(
            num_target_blocks=config.masking.num_target_blocks,
            target_scale_min=config.masking.target_scale_min,
            target_scale_max=config.masking.target_scale_max,
            target_aspect_ratio=config.masking.target_aspect_ratio,
            allow_target_overlap=config.masking.allow_target_overlap,
        )
        if config.masking.masking_type == "block":
            self.mask_generator = BlockMaskGenerator(masking_config)
        else:
            self.mask_generator = SpanMaskGenerator(masking_config)

        # Loss function
        self.loss_fn = JEPALoss(config.loss)

        self._print_info()

    def _print_info(self):
        """Print model info."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nJEPA World Model initialized")
        print(f"  Hidden dim: {self.config.hidden_dim}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  EMA decay: {self.config.ema.ema_decay_start} -> {self.config.ema.ema_decay_end}")

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # =========================================================================
    # CORE METHODS
    # =========================================================================

    def encode_context(
        self,
        byte_ids: torch.Tensor,
        modality: str,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode visible context using online encoder.

        Args:
            byte_ids: Raw bytes [batch, seq_len]
            modality: Modality name
            context_mask: Which positions are visible [batch, seq_len]

        Returns:
            Context embeddings [batch, context_len, hidden_dim]
        """
        # Encode all bytes (we'll mask in backbone)
        byte_emb = self.byte_encoder(byte_ids, attention_mask=context_mask)

        # Process through backbone with mask
        sequence, _ = self.backbone(byte_emb, modality, attention_mask=context_mask)

        return sequence

    def forward(
        self,
        byte_ids: torch.Tensor,
        modality: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        JEPA training forward pass.

        Args:
            byte_ids: Raw bytes [batch, seq_len]
            modality: Modality name
            height: Optional height for 2D masking
            width: Optional width for 2D masking

        Returns:
            loss: Scalar loss value
            outputs: Dictionary with metrics and debug info
        """
        batch_size, seq_len = byte_ids.shape
        device = byte_ids.device

        # 1. Generate masks
        context_mask, target_mask, target_positions = self.mask_generator(
            batch_size, seq_len, device, height, width
        )

        # 2. Encode context (online encoder, receives gradients)
        context_emb = self.encode_context(byte_ids, modality, context_mask)

        # 3. Encode targets (EMA encoder, no gradients)
        with torch.no_grad():
            target_emb_full = self.target_encoder(byte_ids, modality)

        # 4. Extract target embeddings at masked positions (vectorized)
        max_targets = max(len(pos) for pos in target_positions)
        hidden_dim = target_emb_full.shape[-1]
        max_seq = target_emb_full.shape[1]

        # Pad positions to uniform length and create validity mask
        padded_positions = torch.zeros(batch_size, max_targets, dtype=torch.long, device=device)
        target_valid = torch.zeros(batch_size, max_targets, dtype=torch.bool, device=device)

        for b in range(batch_size):
            num_targets = len(target_positions[b])
            if num_targets > 0:
                # Shift by 1 for modality token, clamp to valid range
                positions = (target_positions[b].to(device) + 1).clamp(max=max_seq - 1)
                padded_positions[b, :num_targets] = positions
                target_valid[b, :num_targets] = True

        # Gather embeddings using advanced indexing (vectorized)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_targets)
        target_emb = target_emb_full[batch_indices, padded_positions]

        # Zero out invalid positions
        target_emb = target_emb * target_valid.unsqueeze(-1)

        # 5. Predict target embeddings
        predictions, pred_valid = self.predictor(
            context_emb, target_positions, context_mask
        )

        # 6. Compute loss
        loss, metrics = self.loss_fn(predictions, target_emb, target_valid.float())

        # Add prediction quality metrics
        with torch.no_grad():
            pred_metrics = compute_prediction_metrics(
                predictions[target_valid],
                target_emb[target_valid]
            )
            metrics.update(pred_metrics)
            metrics["ema_decay"] = self.target_encoder.ema_decay
            metrics["num_targets"] = sum(len(p) for p in target_positions) / batch_size

        outputs = {
            "metrics": metrics,
            "predictions": predictions,
            "targets": target_emb,
            "target_positions": target_positions,
            "context_mask": context_mask,
            "target_mask": target_mask,
        }

        return loss, outputs

    def update_target_encoder(self):
        """Update EMA target encoder after optimizer step."""
        self.target_encoder.update(self.byte_encoder, self.backbone)

    # =========================================================================
    # WORLD MODEL CAPABILITIES
    # =========================================================================

    @torch.no_grad()
    def predict_future(
        self,
        byte_ids: torch.Tensor,
        modality: str,
        future_positions: List[torch.Tensor],
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict embeddings at future positions.

        This is the core world model capability: given current state,
        predict what future states would look like in embedding space.

        Args:
            byte_ids: Current state as bytes [batch, seq_len]
            modality: Modality name
            future_positions: Positions to predict for each batch item
            context_mask: Which positions are known (default: all)

        Returns:
            Predicted embeddings [batch, max_positions, hidden_dim]
        """
        batch_size, seq_len = byte_ids.shape
        device = byte_ids.device

        # Default context mask: all visible
        if context_mask is None:
            context_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        # Encode context
        context_emb = self.encode_context(byte_ids, modality, context_mask)

        # Predict at future positions
        predictions, _ = self.predictor(context_emb, future_positions, context_mask)

        return predictions

    @torch.no_grad()
    def compute_energy(
        self,
        byte_ids: torch.Tensor,
        modality: str,
        target_positions: List[torch.Tensor],
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute energy (prediction error) for world model evaluation.

        Low energy = prediction matches reality = consistent with world model.
        High energy = prediction differs = inconsistent/novel/surprising.

        Args:
            byte_ids: Full sequence including targets [batch, seq_len]
            modality: Modality name
            target_positions: Which positions to evaluate
            context_mask: Which positions are context (not targets)

        Returns:
            Energy per sample [batch]
        """
        batch_size, seq_len = byte_ids.shape
        device = byte_ids.device

        # Default: treat non-target positions as context (vectorized)
        if context_mask is None:
            context_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
            for b in range(batch_size):
                positions = target_positions[b]
                valid_positions = positions[positions < seq_len]
                context_mask[b, valid_positions] = False

        # Get predictions
        predictions = self.predict_future(
            byte_ids, modality, target_positions, context_mask
        )

        # Get actual target embeddings
        target_emb_full = self.target_encoder(byte_ids, modality)
        max_seq = target_emb_full.shape[1]

        # Extract target embeddings (vectorized)
        max_targets = predictions.shape[1]
        padded_positions = torch.zeros(batch_size, max_targets, dtype=torch.long, device=device)

        for b in range(batch_size):
            num_targets = len(target_positions[b])
            if num_targets > 0:
                positions = (target_positions[b].to(device) + 1).clamp(max=max_seq - 1)
                padded_positions[b, :num_targets] = positions

        # Gather using advanced indexing
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_targets)
        target_emb = target_emb_full[batch_indices, padded_positions]

        # Energy = MSE
        energy = compute_energy(
            predictions.reshape(-1, predictions.shape[-1]),
            target_emb.reshape(-1, target_emb.shape[-1])
        )

        return energy.reshape(batch_size, -1).mean(dim=-1)

    @torch.no_grad()
    def encode(
        self,
        byte_ids: torch.Tensor,
        modality: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode bytes to embeddings (inference).

        Args:
            byte_ids: Raw bytes [batch, seq_len]
            modality: Modality name
            attention_mask: Optional mask

        Returns:
            Embeddings [batch, hidden_dim] (pooled) or [batch, seq_len, hidden_dim]
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(byte_ids, dtype=torch.bool)

        byte_emb = self.byte_encoder(byte_ids, attention_mask)
        sequence, pooled = self.backbone(byte_emb, modality, attention_mask)

        return pooled

    # =========================================================================
    # CHECKPOINT METHODS
    # =========================================================================

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        step: int = 0,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "target_encoder_ema": self.target_encoder.state_dict_ema(),
            "config": self.config.to_dict(),
            "step": step,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        device: str = "cpu",
    ) -> "JEPAWorldModel":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = ByteJEPAConfig.from_dict(checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        if "target_encoder_ema" in checkpoint:
            model.target_encoder.load_state_dict_ema(checkpoint["target_encoder_ema"])

        print(f"Loaded checkpoint from {path} (step {checkpoint.get('step', 0)})")
        return model




if __name__ == "__main__":  # pragma: no cover
    print("Testing JEPA World Model")
    print("=" * 50)

    config = get_tiny_config()
    model = JEPAWorldModel(config)

    # Test data
    batch_size = 2
    seq_len = 256
    byte_ids = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)

    # Test forward pass
    print("\nTesting forward pass:")
    loss, outputs = model(byte_ids, modality="text")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Metrics: {outputs['metrics']}")

    # Test EMA update
    print("\nTesting EMA update:")
    model.update_target_encoder()
    print(f"  EMA decay: {model.target_encoder.ema_decay:.6f}")

    # Test world model capabilities
    print("\nTesting world model capabilities:")

    # Predict future
    future_positions = [torch.tensor([50, 100, 150]) for _ in range(batch_size)]
    predictions = model.predict_future(byte_ids, "text", future_positions)
    print(f"  Future predictions shape: {predictions.shape}")

    # Compute energy
    energy = model.compute_energy(byte_ids, "text", future_positions)
    print(f"  Energy: {energy}")

    # Encode
    embedding = model.encode(byte_ids, "text")
    print(f"  Encoding shape: {embedding.shape}")

    print("\nWorld model test passed!")
