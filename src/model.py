"""
JEPA World Model: Byte-Level Joint Embedding Predictive Architecture

Following Yann LeCun's vision: predict abstract representations, not pixels.
The model learns to understand the world by predicting masked region embeddings.

Key insight: Low prediction error = consistent with world model.

API PARADIGM: FLATTENED BYTE SEQUENCES
======================================
ALL inputs are 1D flattened byte sequences [batch, seq_len], regardless of modality:
- Text:  UTF-8 bytes, naturally 1D
- Audio: PCM samples as bytes, naturally 1D
- Vision: RGB pixels MUST be pre-flattened to [batch, H*W*C]
  Example: 32x32 RGB image -> [batch, 3072] where 3072 = 32*32*3

The height/width parameters in forward() are ONLY used for 2D block masking.
They do NOT change the input format - inputs are always flattened byte sequences.

When using block masking for images:
    loss, outputs = model(flat_image_bytes, modality="vision", height=32, width=32)
    # flat_image_bytes shape: [batch, 32*32*3] NOT [batch, 32, 32, 3]

When using span masking (default) for any modality:
    loss, outputs = model(byte_ids, modality="text")  # height/width ignored
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
        ema_decay_initial: float = 0.996,
        ema_decay_final: float = 0.9999,
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
        self.ema_decay_initial = ema_decay_initial
        self.ema_decay_final = ema_decay_final
        self.ema_warmup_steps = ema_warmup_steps
        self.current_step = 0

    @property
    def ema_decay(self) -> float:
        """Get current EMA decay value (scheduled)."""
        if self.ema_warmup_steps <= 0:
            return self.ema_decay_final
        progress = min(self.current_step / self.ema_warmup_steps, 1.0)
        return self.ema_decay_initial + progress * (self.ema_decay_final - self.ema_decay_initial)

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

        # Y7: Gradient leakage assertion - target encoder outputs must never require gradients
        # This ensures no gradients flow back through the EMA-updated target encoder,
        # which would violate the JEPA training paradigm (predictor learns, target encoder follows via EMA)
        assert not sequence.requires_grad, (
            "Target encoder output requires gradients! This would cause gradient leakage. "
            "Ensure forward() is called within torch.no_grad() context."
        )

        return sequence

    def state_dict_ema(self) -> dict:
        """Get state dict including EMA metadata."""
        return {
            "byte_encoder": self.byte_encoder.state_dict(),
            "backbone": self.backbone.state_dict(),
            "current_step": self.current_step,
            "ema_decay_initial": self.ema_decay_initial,
            "ema_decay_final": self.ema_decay_final,
            "ema_warmup_steps": self.ema_warmup_steps,
        }

    def load_state_dict_ema(self, state_dict: dict):
        """Load state dict including EMA metadata."""
        self.byte_encoder.load_state_dict(state_dict["byte_encoder"])
        self.backbone.load_state_dict(state_dict["backbone"])
        self.current_step = state_dict["current_step"]
        self.ema_decay_initial = state_dict.get("ema_decay_initial", self.ema_decay_initial)
        self.ema_decay_final = state_dict.get("ema_decay_final", self.ema_decay_final)
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

    def __init__(
        self,
        config: ByteJEPAConfig,
        mask_generator: Optional[nn.Module] = None,
    ):
        """
        Initialize JEPA World Model.

        Args:
            config: Model configuration
            mask_generator: Optional custom mask generator. If not provided,
                creates default based on config.masking settings.
                Must implement: __call__(batch_size, seq_len, device) ->
                    (context_mask, target_mask, target_positions)
        """
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
            ema_decay_initial=config.ema.ema_decay_initial,
            ema_decay_final=config.ema.ema_decay_final,
            ema_warmup_steps=config.ema.ema_warmup_steps,
        )

        # Mask generator (user-provided or default from config)
        if mask_generator is not None:
            self.mask_generator = mask_generator
        else:
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
        print(f"  EMA decay: {self.config.ema.ema_decay_initial} -> {self.config.ema.ema_decay_final}")

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
            byte_ids: Raw bytes [batch, seq_len]. MUST be flattened 1D sequence.
                For images: pre-flatten [B, H, W, C] -> [B, H*W*C] before passing.
                For text/audio: naturally 1D, pass directly.
            modality: Modality name ("vision", "text", or "audio")
            height: Optional spatial height for 2D BLOCK masking only.
                Only used when config.masking.masking_type == "block".
                For span masking (default), this is ignored.
            width: Optional spatial width for 2D BLOCK masking only.
                Only used when config.masking.masking_type == "block".
                For span masking (default), this is ignored.

        Returns:
            loss: Scalar loss value
            outputs: Dictionary with metrics and debug info

        Note:
            height/width do NOT change the input format. They only inform the
            BlockMaskGenerator how to interpret the 1D sequence as a 2D grid
            for generating spatially contiguous mask blocks.
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
    # AUTOREGRESSIVE / PLANNING CAPABILITIES
    # =========================================================================

    @torch.no_grad()
    def rollout(
        self,
        byte_ids: torch.Tensor,
        modality: str,
        num_steps: int,
        step_size: int = 1,
        context_window: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive rollout: predict future states iteratively.

        This enables planning by rolling out predicted states:
            state_0 -> predictor -> state_1 (predicted)
            state_0 + state_1 -> predictor -> state_2 (predicted)
            ...

        Note: This operates in EMBEDDING space, not byte space.
        The model predicts abstract representations, not raw bytes.
        This is by design (JEPA predicts representations, not pixels).

        For planning algorithms (CEM, MCTS, etc.), use this to:
        1. Rollout candidate action sequences
        2. Evaluate final state embeddings
        3. Select actions that lead to desired states

        Args:
            byte_ids: Initial state as bytes [batch, seq_len]
            modality: Modality name
            num_steps: Number of steps to roll out
            step_size: Positions to advance per step
            context_window: How many past positions to use as context
                           (None = use all available)

        Returns:
            Predicted embeddings at each step [batch, num_steps, hidden_dim]

        Example:
            # Predict 5 future states
            future_states = model.rollout(byte_ids, "text", num_steps=5)
            # future_states[b, t] = predicted embedding at step t for batch b
        """
        batch_size, seq_len = byte_ids.shape
        device = byte_ids.device
        hidden_dim = self.config.hidden_dim

        # Initialize with encoding of current state
        byte_emb = self.byte_encoder(byte_ids, attention_mask=None)
        current_sequence, _ = self.backbone(byte_emb, modality, attention_mask=None)

        # Store rollout predictions
        rollout_predictions = torch.zeros(
            batch_size, num_steps, hidden_dim, device=device
        )

        current_pos = seq_len  # Start predicting after the input

        for step in range(num_steps):
            # Position to predict
            predict_pos = current_pos + step * step_size

            # Context: use last context_window positions, or all
            if context_window is not None:
                context_start = max(0, current_sequence.shape[1] - context_window)
                context = current_sequence[:, context_start:]
            else:
                context = current_sequence

            # Predict next position
            # Create position tensor for predictor
            target_positions = [
                torch.tensor([predict_pos], device=device)
                for _ in range(batch_size)
            ]

            predictions, _ = self.predictor(
                context,
                target_positions,
                context_mask=None,
            )

            # Store prediction
            rollout_predictions[:, step] = predictions[:, 0]

            # Append prediction to sequence for next iteration
            # This is the autoregressive part: use predicted state as new context
            current_sequence = torch.cat([
                current_sequence,
                predictions,
            ], dim=1)

        return rollout_predictions

    @torch.no_grad()
    def evaluate_trajectory(
        self,
        byte_ids: torch.Tensor,
        modality: str,
        trajectory_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate how well a trajectory of embeddings matches the world model.

        Used for planning: given candidate trajectories, score them by
        how consistent they are with the learned world model.

        Args:
            byte_ids: Initial state [batch, seq_len]
            modality: Modality name
            trajectory_embeddings: Candidate trajectory [batch, traj_len, hidden_dim]

        Returns:
            Energy (lower = more consistent) [batch]
        """
        batch_size = byte_ids.shape[0]
        traj_len = trajectory_embeddings.shape[1]
        device = byte_ids.device

        # Roll out from initial state
        predicted_trajectory = self.rollout(
            byte_ids, modality, num_steps=traj_len
        )

        # Compare predicted vs given trajectory
        # Energy = MSE between predicted and given
        diff = predicted_trajectory - trajectory_embeddings
        energy = (diff ** 2).sum(dim=-1).mean(dim=-1)  # [batch]

        return energy

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
