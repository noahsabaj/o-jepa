"""
JEPA World Model Training Script.

Implements training loop with:
- Warmup-Stable-Decay (WSD) learning rate scheduler
- Mixed precision training
- Gradient accumulation
- Checkpointing
- Single-modality masked prediction (JEPA style)
- EMA target encoder updates
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from .config import ByteJEPAConfig
from .model import JEPAWorldModel
from .progress import ProgressReporter, ModelInfo, TrainingInfo, SimpleReporter


# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================

class WSDScheduler:
    """
    Warmup-Stable-Decay (WSD) Learning Rate Scheduler.

    Three phases:
    1. Warmup: Linear increase from 0 to base_lr
    2. Stable: Constant at base_lr
    3. Decay: Cosine decay to min_lr
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        warmup_ratio: float = 0.01,
        stable_ratio: float = 0.79,
        decay_ratio: float = 0.20,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = base_lr * min_lr_ratio

        # Calculate phase boundaries
        self.warmup_end = int(total_steps * warmup_ratio)
        self.stable_end = int(total_steps * (warmup_ratio + stable_ratio))

        self.current_step = 0

    def get_lr(self) -> float:
        """Calculate learning rate for current step."""
        if self.current_step < self.warmup_end:
            # Warmup phase: linear increase
            progress = self.current_step / max(1, self.warmup_end)
            return self.base_lr * progress

        elif self.current_step < self.stable_end:
            # Stable phase: constant
            return self.base_lr

        else:
            # Decay phase: cosine decay
            decay_steps = self.total_steps - self.stable_end
            progress = (self.current_step - self.stable_end) / max(1, decay_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

    def step(self):
        """Update learning rate and increment step."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_scale' in param_group:
                param_group['lr'] = lr * param_group['lr_scale']
            else:
                param_group['lr'] = lr
        self.current_step += 1

    def state_dict(self) -> Dict:
        """Return scheduler state for checkpointing."""
        return {'current_step': self.current_step}

    def load_state_dict(self, state_dict: Dict):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict['current_step']


# =============================================================================
# OPTIMIZER CREATION
# =============================================================================

def create_optimizer(
    model: JEPAWorldModel,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05,
    betas: Tuple[float, float] = (0.9, 0.95),
    use_muon: bool = True,
) -> torch.optim.Optimizer:
    """
    Create optimizer with Muon/AdamW hybrid.

    Muon (Momentum Orthogonalized by Newton-schulz) is used for 2D weight matrices
    for faster convergence. AdamW is used for 1D params (biases, norms, embeddings).

    Args:
        model: JEPAWorldModel model
        learning_rate: Base learning rate
        weight_decay: Weight decay
        betas: Adam betas
        use_muon: Whether to use Muon optimizer (falls back to AdamW if False or unavailable)

    Returns:
        Configured optimizer
    """
    # Try to import Muon
    muon_available = False
    if use_muon:
        try:
            from pytorch_optimizer import Muon
            muon_available = True
        except ImportError:
            print("Warning: pytorch-optimizer not installed, falling back to AdamW")
            print("Install with: pip install pytorch-optimizer")

    if muon_available:
        # Muon hybrid: use Muon for 2D weights, AdamW for rest
        muon_params = []  # 2D weight matrices (excluding embeddings)
        adamw_params = []  # 1D params, embeddings, norms

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Muon works best on 2D weight matrices, excluding embeddings
            is_2d_weight = param.ndim >= 2
            is_embedding = 'embed' in name.lower()
            is_norm = 'norm' in name.lower()

            if is_2d_weight and not is_embedding and not is_norm:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        param_groups = [
            {
                'params': muon_params,
                'lr': learning_rate,
                'weight_decay': weight_decay,
                'use_muon': True,
                'name': 'muon_weights',
            },
            {
                'params': adamw_params,
                'lr': learning_rate,
                'weight_decay': weight_decay,
                'use_muon': False,
                'name': 'adamw_params',
            },
        ]

        # Filter empty groups
        param_groups = [g for g in param_groups if len(g['params']) > 0]

        optimizer = Muon(
            param_groups,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            backend='newtonschulz5',
        )

        muon_count = len(muon_params)
        adamw_count = len(adamw_params)
        print(f"Using Muon optimizer: {muon_count} params with Muon, {adamw_count} params with AdamW")

        return optimizer

    # Fallback to AdamW with fused=True for CUDA speedup
    predictor_2d_params = []
    predictor_other_params = []
    backbone_params = []
    encoder_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'predictor' in name:
            if param.ndim == 2 and 'norm' not in name:
                predictor_2d_params.append(param)
            else:
                predictor_other_params.append(param)
        elif 'backbone' in name:
            backbone_params.append(param)
        elif 'byte_encoder' in name:
            encoder_params.append(param)
        elif 'decoder' in name:
            decoder_params.append(param)

    param_groups = [
        {
            'params': predictor_2d_params,
            'lr': learning_rate,
            'weight_decay': weight_decay,
            'lr_scale': 1.0,
            'name': 'predictor_2d',
        },
        {
            'params': predictor_other_params,
            'lr': learning_rate,
            'weight_decay': weight_decay,
            'lr_scale': 1.0,
            'name': 'predictor_other',
        },
        {
            'params': backbone_params,
            'lr': learning_rate,
            'weight_decay': weight_decay,
            'lr_scale': 1.0,
            'name': 'backbone',
        },
        {
            'params': encoder_params,
            'lr': learning_rate * 0.5,  # Slightly reduced for encoder
            'weight_decay': weight_decay,
            'lr_scale': 0.5,
            'name': 'byte_encoder',
        },
        {
            'params': decoder_params,
            'lr': learning_rate,
            'weight_decay': weight_decay,
            'lr_scale': 1.0,
            'name': 'decoders',
        },
    ]

    # Filter empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]

    # Use fused AdamW for CUDA speedup when available
    use_fused = torch.cuda.is_available()

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
        fused=use_fused,
    )

    if use_fused:
        print("Using fused AdamW optimizer (CUDA-optimized)")

    return optimizer


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 1000
    total_steps: int = 100000
    min_lr_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_mixed_precision: bool = True
    eval_every_steps: int = 1000
    save_every_steps: int = 5000
    log_every_steps: int = 100


@dataclass
class TrainingState:
    """Mutable training state."""
    step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    total_samples: int = 0


def train_step_bytes(
    model: JEPAWorldModel,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: WSDScheduler,
    scaler: Optional[GradScaler],
    config: TrainingConfig,
    state: TrainingState,
    accumulation_step: int,
    source_modality: str = "vision",
    target_modality: str = "text",
) -> Dict[str, float]:
    """
    Single training step with byte-level data.

    Args:
        model: JEPAWorldModel model
        batch: Data batch with modality bytes and masks
        optimizer: Optimizer
        scheduler: LR scheduler
        scaler: Gradient scaler for mixed precision
        config: Training config
        state: Training state
        accumulation_step: Current accumulation step
        source_modality: Source modality name
        target_modality: Target modality name

    Returns:
        Metrics dictionary
    """
    device = next(model.parameters()).device
    use_amp = scaler is not None

    # Get source and target from batch
    # Handles both paired collator format and basic format
    if "source_bytes" in batch:
        # PairedByteCollator format
        source_bytes = batch["source_bytes"].to(device)
        source_mask = batch["source_mask"].to(device)
        target_bytes = batch["target_bytes"].to(device)
        target_mask = batch["target_mask"].to(device)
        source_mod = batch.get("source_modality", source_modality)
        target_mod = batch.get("target_modality", target_modality)
    else:
        # Basic format: {modality}_bytes or just {modality}
        source_key = f"{source_modality}_bytes" if f"{source_modality}_bytes" in batch else source_modality
        target_key = f"{target_modality}_bytes" if f"{target_modality}_bytes" in batch else target_modality
        source_bytes = batch[source_key].to(device)
        target_bytes = batch[target_key].to(device)
        source_mask = batch.get(f"{source_modality}_mask", torch.ones_like(source_bytes, dtype=torch.bool)).to(device)
        target_mask = batch.get(f"{target_modality}_mask", torch.ones_like(target_bytes, dtype=torch.bool)).to(device)
        source_mod = source_modality
        target_mod = target_modality

    # Forward pass with mixed precision
    with autocast('cuda', enabled=use_amp):
        loss, outputs = model(
            source_bytes=source_bytes,
            target_bytes=target_bytes,
            source_modality=source_mod,
            target_modality=target_mod,
            source_mask=source_mask,
            target_mask=target_mask,
        )
        # Scale loss for gradient accumulation
        loss = loss / config.gradient_accumulation_steps

    # Backward pass
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    # Extract metrics from outputs
    raw_metrics = outputs.get("metrics", {})
    metrics = {k: v if isinstance(v, (int, float)) else v for k, v in raw_metrics.items()}

    # Update weights on last accumulation step
    is_update_step = (accumulation_step + 1) == config.gradient_accumulation_steps

    if is_update_step:
        if scaler is not None:
            scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.max_grad_norm
        )
        metrics['grad_norm'] = grad_norm.item()

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad()

        state.step += 1
        metrics['lr'] = scheduler.get_lr()

    # Scale loss back for logging
    metrics['loss'] = loss.item() * config.gradient_accumulation_steps
    state.total_samples += source_bytes.shape[0]

    return metrics


def train_step_bidirectional_bytes(
    model: JEPAWorldModel,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: WSDScheduler,
    scaler: Optional[GradScaler],
    config: TrainingConfig,
    state: TrainingState,
    accumulation_step: int,
    modality_a: str = "vision",
    modality_b: str = "text",
) -> Dict[str, float]:
    """
    Bidirectional training step: A<->B both directions.

    Args:
        model: JEPAWorldModel model
        batch: Data batch with byte tensors
        optimizer: Optimizer
        scheduler: LR scheduler
        scaler: Gradient scaler
        config: Training config
        state: Training state
        accumulation_step: Current accumulation step
        modality_a: First modality
        modality_b: Second modality

    Returns:
        Metrics dictionary
    """
    device = next(model.parameters()).device
    use_amp = scaler is not None

    # Get bytes for both modalities
    if "source_bytes" in batch:
        bytes_a = batch["source_bytes"].to(device)
        mask_a = batch["source_mask"].to(device)
        bytes_b = batch["target_bytes"].to(device)
        mask_b = batch["target_mask"].to(device)
    else:
        key_a = f"{modality_a}_bytes" if f"{modality_a}_bytes" in batch else modality_a
        key_b = f"{modality_b}_bytes" if f"{modality_b}_bytes" in batch else modality_b
        bytes_a = batch[key_a].to(device)
        bytes_b = batch[key_b].to(device)
        mask_a = batch.get(f"{modality_a}_mask", torch.ones_like(bytes_a, dtype=torch.bool)).to(device)
        mask_b = batch.get(f"{modality_b}_mask", torch.ones_like(bytes_b, dtype=torch.bool)).to(device)

    with autocast('cuda', enabled=use_amp):
        loss, outputs = model.forward_bidirectional(
            modality_a_bytes=bytes_a,
            modality_a_name=modality_a,
            modality_b_bytes=bytes_b,
            modality_b_name=modality_b,
            mask_a=mask_a,
            mask_b=mask_b,
        )
        loss = loss / config.gradient_accumulation_steps

    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    # Extract metrics from outputs
    raw_metrics = outputs.get("metrics", {})
    metrics = {k: v if isinstance(v, (int, float)) else v for k, v in raw_metrics.items()}

    is_update_step = (accumulation_step + 1) == config.gradient_accumulation_steps

    if is_update_step:
        if scaler is not None:
            scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.max_grad_norm
        )
        metrics['grad_norm'] = grad_norm.item()

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad()

        state.step += 1
        metrics['lr'] = scheduler.get_lr()

    metrics['loss'] = loss.item() * config.gradient_accumulation_steps
    state.total_samples += bytes_a.shape[0]

    return metrics


def train_step_jepa(
    model: JEPAWorldModel,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: WSDScheduler,
    scaler: Optional[GradScaler],
    config: TrainingConfig,
    state: TrainingState,
    accumulation_step: int,
    modality: str = "text",
) -> Dict[str, float]:
    """
    JEPA World Model training step.

    Single-modality masked prediction:
    1. Mask random regions
    2. Predict masked embeddings from context
    3. Loss = MSE(predictions, target_encoder(masked))
    4. Update EMA target encoder

    Args:
        model: JEPAWorldModel
        batch: Data batch with byte tensors
        optimizer: Optimizer
        scheduler: LR scheduler
        scaler: Gradient scaler
        config: Training config
        state: Training state
        accumulation_step: Current accumulation step
        modality: Modality name

    Returns:
        Metrics dictionary
    """
    device = next(model.parameters()).device
    use_amp = scaler is not None

    # Get bytes from batch (handles multiple formats)
    if "bytes" in batch:
        byte_ids = batch["bytes"].to(device)
    elif f"{modality}_bytes" in batch:
        byte_ids = batch[f"{modality}_bytes"].to(device)
    elif modality in batch:
        byte_ids = batch[modality].to(device)
    else:
        # Try common keys
        for key in ["input_ids", "pixel_values", "audio"]:
            if key in batch:
                byte_ids = batch[key].to(device)
                break
        else:
            raise ValueError(f"Could not find byte data in batch. Keys: {batch.keys()}")

    # Get optional height/width for 2D masking
    height = batch.get("height", None)
    width = batch.get("width", None)

    # Forward pass with mixed precision
    with autocast('cuda', enabled=use_amp):
        loss, outputs = model(byte_ids, modality=modality, height=height, width=width)
        loss = loss / config.gradient_accumulation_steps

    # Backward pass
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    # Extract metrics
    metrics = outputs.get("metrics", {})

    # Update weights on last accumulation step
    is_update_step = (accumulation_step + 1) == config.gradient_accumulation_steps

    if is_update_step:
        if scaler is not None:
            scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.max_grad_norm
        )
        metrics['grad_norm'] = grad_norm.item()

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # CRITICAL: Update EMA target encoder AFTER optimizer step
        model.update_target_encoder()

        scheduler.step()
        optimizer.zero_grad()

        state.step += 1
        metrics['lr'] = scheduler.get_lr()

    metrics['loss'] = loss.item() * config.gradient_accumulation_steps
    state.total_samples += byte_ids.shape[0]

    return metrics


def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing for memory efficiency."""
    # Enable in backbone transformer layers
    if hasattr(model.backbone, 'layers'):
        for layer in model.backbone.layers:
            if hasattr(layer, 'use_checkpoint'):
                layer.use_checkpoint = True

    # Enable in predictor transformer layers
    if hasattr(model.predictor, 'layers'):
        for layer in model.predictor.layers:
            if hasattr(layer, 'use_checkpoint'):
                layer.use_checkpoint = True

    # Enable in byte encoder
    if hasattr(model, 'byte_encoder') and hasattr(model.byte_encoder, 'blocks'):
        for block in model.byte_encoder.blocks:
            if hasattr(block, 'use_checkpoint'):
                block.use_checkpoint = True

    print("Gradient checkpointing enabled")


def save_checkpoint(
    model: JEPAWorldModel,
    optimizer: torch.optim.Optimizer,
    scheduler: WSDScheduler,
    state: TrainingState,
    config: ByteJEPAConfig,
    save_dir: str,
    is_best: bool = False,
):
    """Save training checkpoint."""
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'state': {
            'step': state.step,
            'epoch': state.epoch,
            'best_loss': state.best_loss,
            'total_samples': state.total_samples,
        },
        'config': config,
    }

    latest_path = os.path.join(save_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)

    step_path = os.path.join(save_dir, f'checkpoint_step{state.step}.pt')
    torch.save(checkpoint, step_path)

    if is_best:
        best_path = os.path.join(save_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_path)


def load_checkpoint(
    checkpoint_path: str,
    model: JEPAWorldModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[WSDScheduler] = None,
    device: str = 'cpu',
) -> TrainingState:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    state_dict = checkpoint.get('state', {})
    state = TrainingState(
        step=state_dict.get('step', 0),
        epoch=state_dict.get('epoch', 0),
        best_loss=state_dict.get('best_loss', float('inf')),
        total_samples=state_dict.get('total_samples', 0),
    )

    print(f"Loaded checkpoint from step {state.step}")
    return state


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(
    model: JEPAWorldModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: ByteJEPAConfig,
    training_config: TrainingConfig,
    save_dir: str,
    resume_path: Optional[str] = None,
    device: str = 'cuda',
    use_wandb: bool = False,
    bidirectional: bool = False,  # Default to JEPA mode (not bidirectional)
    source_modality: str = "text",
    target_modality: str = "text",
    reporter: Optional[ProgressReporter] = None,
    jepa_mode: bool = True,  # New: use JEPA world model training
):
    """
    Main training loop for JEPA World Model.

    Two modes:
    1. JEPA mode (jepa_mode=True): Single-modality masked prediction
    2. Legacy mode (jepa_mode=False): Cross-modal contrastive (bidirectional)

    Args:
        model: JEPAWorldModel
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Model configuration
        training_config: Training hyperparameters
        save_dir: Directory for checkpoints
        resume_path: Path to resume from
        device: Device to train on
        use_wandb: Whether to log to Weights & Biases
        bidirectional: Whether to use bidirectional training (legacy mode)
        source_modality: Modality for JEPA training or source for legacy
        target_modality: Target modality for legacy mode
        reporter: Progress reporter for training output
        jepa_mode: If True, use JEPA world model training (masked prediction)
    """
    # Default to SimpleReporter if none provided
    if reporter is None:
        reporter = SimpleReporter()
    model = model.to(device)

    optimizer = create_optimizer(
        model,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        betas=training_config.betas,
    )

    scheduler = WSDScheduler(
        optimizer=optimizer,
        base_lr=training_config.learning_rate,
        warmup_steps=training_config.warmup_steps,
        total_steps=training_config.total_steps,
        min_lr_ratio=training_config.min_lr_ratio,
    )

    use_amp = training_config.use_mixed_precision and device == 'cuda'
    scaler = GradScaler() if use_amp else None

    state = TrainingState()

    if resume_path and os.path.exists(resume_path):
        state = load_checkpoint(resume_path, model, optimizer, scheduler, device)

    if use_wandb:
        try:
            import wandb
            wandb.init(project='byte-jepa', config={
                'model': config.__dict__,
                'training': training_config.__dict__,
            })
        except ImportError:
            use_wandb = False
            print("WandB not installed, skipping logging")

    # Create model and training info for reporter
    model_info = ModelInfo(
        name="JEPAWorldModel",
        total_params=sum(p.numel() for p in model.parameters()),
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        hidden_dim=config.hidden_dim,
        modalities=config.active_modalities,
        byte_encoder_layers=config.byte_encoder.num_layers,
        backbone_layers=config.backbone.num_layers,
        predictor_layers=config.predictor.num_layers,
    )

    training_info = TrainingInfo(
        total_steps=training_config.total_steps,
        batch_size=train_loader.batch_size or 1,
        learning_rate=training_config.learning_rate,
        warmup_steps=training_config.warmup_steps,
        device=device,
        mixed_precision=use_amp,
        bidirectional=bidirectional if not jepa_mode else False,
        source_modality=source_modality,
        target_modality=target_modality if not jepa_mode else source_modality,
    )

    if jepa_mode:
        print(f"JEPA World Model training: modality={source_modality}")
    else:
        print(f"Legacy mode: bidirectional={bidirectional}")

    reporter.on_train_start(model_info, training_info)

    model.train()
    accumulation_step = 0
    running_loss = 0.0
    start_time = time.time()

    while state.step < training_config.total_steps:
        for batch in train_loader:
            if state.step >= training_config.total_steps:
                break

            if jepa_mode:
                # JEPA World Model: single-modality masked prediction
                metrics = train_step_jepa(
                    model=model,
                    batch=batch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=training_config,
                    state=state,
                    accumulation_step=accumulation_step,
                    modality=source_modality,
                )
            elif bidirectional:
                metrics = train_step_bidirectional_bytes(
                    model=model,
                    batch=batch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=training_config,
                    state=state,
                    accumulation_step=accumulation_step,
                    modality_a=source_modality,
                    modality_b=target_modality,
                )
            else:
                metrics = train_step_bytes(
                    model=model,
                    batch=batch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=training_config,
                    state=state,
                    accumulation_step=accumulation_step,
                    source_modality=source_modality,
                    target_modality=target_modality,
                )

            accumulation_step = (accumulation_step + 1) % training_config.gradient_accumulation_steps
            running_loss += metrics['loss']

            # Logging
            if state.step % training_config.log_every_steps == 0 and accumulation_step == 0:
                elapsed = time.time() - start_time
                samples_per_sec = state.total_samples / elapsed if elapsed > 0 else 0

                avg_loss = running_loss / training_config.log_every_steps
                running_loss = 0.0

                # Build metrics dict based on training mode
                log_metrics = {
                    'loss': metrics['loss'],
                    'avg_loss': avg_loss,
                    'lr': metrics.get('lr', 0),
                    'grad_norm': metrics.get('grad_norm', 0),
                    'samples_per_sec': samples_per_sec,
                    'epoch': state.epoch,
                    'total_samples': state.total_samples,
                    'total_steps': training_config.total_steps,
                }

                if jepa_mode:
                    # JEPA World Model metrics
                    log_metrics.update({
                        'pred_loss': metrics.get('pred_loss'),
                        'var_loss': metrics.get('var_loss'),
                        'std_mean': metrics.get('std_mean'),
                        'cosine_sim': metrics.get('cosine_sim'),
                        'ema_decay': metrics.get('ema_decay'),
                        'num_targets': metrics.get('num_targets'),
                    })
                else:
                    # Legacy contrastive metrics
                    log_metrics.update({
                        'infonce': metrics.get('infonce'),
                        'vicreg': metrics.get('vicreg'),
                        'temperature': metrics.get('temperature'),
                        'infonce_acc_p2t': metrics.get('infonce_acc_p2t'),
                        'infonce_acc_t2p': metrics.get('infonce_acc_t2p'),
                        'vicreg_var_loss': metrics.get('vicreg_var_loss'),
                        'vicreg_cov_loss': metrics.get('vicreg_cov_loss'),
                        'vicreg_std_mean': metrics.get('vicreg_std_mean'),
                    })

                reporter.on_step(step=state.step, metrics=log_metrics)

                if use_wandb:
                    import wandb
                    wandb.log({
                        'loss': avg_loss,
                        'lr': metrics.get('lr', 0),
                        'grad_norm': metrics.get('grad_norm', 0),
                        'samples_per_sec': samples_per_sec,
                        'step': state.step,
                    })

            # Evaluation (only on accumulation boundaries, skip step 0)
            if (val_loader is not None
                and state.step > 0
                and state.step % training_config.eval_every_steps == 0
                and accumulation_step == 0):
                val_loss, retrieval_metrics = evaluate(
                    model, val_loader, device,
                    source_modality=source_modality,
                    target_modality=target_modality,
                )

                is_best = val_loss < state.best_loss
                if is_best:
                    state.best_loss = val_loss

                reporter.on_eval(state.step, val_loss, metrics=retrieval_metrics)

                if use_wandb:
                    import wandb
                    wandb_metrics = {'val_loss': val_loss, 'step': state.step}
                    if retrieval_metrics:
                        wandb_metrics.update(retrieval_metrics)
                    wandb.log(wandb_metrics)

                model.train()

            # Checkpointing (only on accumulation boundaries, skip step 0)
            if (state.step > 0
                and state.step % training_config.save_every_steps == 0
                and accumulation_step == 0):
                is_best = val_loader is not None and state.best_loss == running_loss
                save_checkpoint(
                    model, optimizer, scheduler, state, config, save_dir,
                    is_best=is_best
                )
                checkpoint_path = os.path.join(save_dir, f'checkpoint_step{state.step}.pt')
                reporter.on_save(state.step, checkpoint_path, is_best=is_best)

        state.epoch += 1

    # Final save
    save_checkpoint(model, optimizer, scheduler, state, config, save_dir)
    final_path = os.path.join(save_dir, 'checkpoint_latest.pt')
    reporter.on_save(state.step, final_path)
    reporter.on_train_end(state.step, {'loss': avg_loss, 'best_val_loss': state.best_loss})

    if use_wandb:
        import wandb
        wandb.finish()


def evaluate(
    model: JEPAWorldModel,
    val_loader: DataLoader,
    device: str = 'cuda',
    max_batches: Optional[int] = None,
    source_modality: str = "vision",
    target_modality: str = "text",
    compute_retrieval: bool = True,
) -> Tuple[float, Optional[Dict[str, float]]]:
    """
    Evaluate model on validation set.

    Args:
        model: JEPAWorldModel model
        val_loader: Validation data loader
        device: Device
        max_batches: Maximum batches to evaluate
        source_modality: Source modality
        target_modality: Target modality
        compute_retrieval: Whether to compute retrieval metrics (R@K)

    Returns:
        Tuple of (average validation loss, retrieval metrics dict or None)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Collect embeddings for retrieval metrics
    all_pred_embeds = []
    all_target_embeds = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches and i >= max_batches:
                break

            # Handle different batch formats
            if "source_bytes" in batch:
                source_bytes = batch["source_bytes"].to(device)
                source_mask = batch["source_mask"].to(device)
                target_bytes = batch["target_bytes"].to(device)
                target_mask = batch["target_mask"].to(device)
                source_mod = batch.get("source_modality", source_modality)
                target_mod = batch.get("target_modality", target_modality)
            else:
                source_key = f"{source_modality}_bytes" if f"{source_modality}_bytes" in batch else source_modality
                target_key = f"{target_modality}_bytes" if f"{target_modality}_bytes" in batch else target_modality
                source_bytes = batch[source_key].to(device)
                target_bytes = batch[target_key].to(device)
                source_mask = batch.get(f"{source_modality}_mask", torch.ones_like(source_bytes, dtype=torch.bool)).to(device)
                target_mask = batch.get(f"{target_modality}_mask", torch.ones_like(target_bytes, dtype=torch.bool)).to(device)
                source_mod = source_modality
                target_mod = target_modality

            loss, outputs = model(
                source_bytes=source_bytes,
                target_bytes=target_bytes,
                source_modality=source_mod,
                target_modality=target_mod,
                source_mask=source_mask,
                target_mask=target_mask,
                return_embeddings=compute_retrieval,
            )

            total_loss += loss.item()
            num_batches += 1

            # Collect embeddings for retrieval metrics
            if compute_retrieval and 'pred_embed' in outputs and 'target_embed' in outputs:
                all_pred_embeds.append(outputs['pred_embed'])
                all_target_embeds.append(outputs['target_embed'])

    avg_loss = total_loss / max(1, num_batches)

    # Compute retrieval metrics if we have embeddings
    retrieval_metrics = None
    if compute_retrieval and all_pred_embeds and all_target_embeds:
        # Concatenate all embeddings
        pred_embeds = torch.cat(all_pred_embeds, dim=0)
        target_embeds = torch.cat(all_target_embeds, dim=0)
        # Compute R@1, R@5, R@10
        retrieval_metrics = compute_retrieval_metrics(pred_embeds, target_embeds)

    return avg_loss, retrieval_metrics


if __name__ == "__main__":  # pragma: no cover
    # Test training components
    print("Testing Byte-level O-JEPA Training Components")
    print("=" * 50)

    from .config import get_tiny_config
    from .data import SyntheticByteDataset, get_collator

    # Create tiny model
    config = get_tiny_config()
    model = JEPAWorldModel(config)

    # Test optimizer creation
    training_config = TrainingConfig(
        learning_rate=1e-4,
        total_steps=100,
        warmup_steps=10,
    )

    optimizer = create_optimizer(
        model,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    print(f"Optimizer param groups: {len(optimizer.param_groups)}")
    for i, group in enumerate(optimizer.param_groups):
        print(f"  Group {i} ({group.get('name', 'unnamed')}): {len(group['params'])} params, lr={group['lr']:.2e}")

    # Test scheduler
    scheduler = WSDScheduler(
        optimizer=optimizer,
        base_lr=training_config.learning_rate,
        warmup_steps=10,
        total_steps=100,
    )

    print(f"\nLR schedule test:")
    for step in [0, 5, 10, 50, 80, 90, 99]:
        scheduler.current_step = step
        lr = scheduler.get_lr()
        print(f"  Step {step}: LR = {lr:.6f}")

    # Test with synthetic data
    print("\nTesting training step with synthetic data...")
    dataset = SyntheticByteDataset(
        num_samples=8,
        modalities=["vision", "text"],
        vision_seq_len=config.vision_seq_len,
        text_seq_len=config.text_max_seq_len,
    )
    collator = get_collator(mode="paired", source_modality="vision", target_modality="text")
    loader = DataLoader(dataset, batch_size=2, collate_fn=collator)

    model = model.to('cpu')
    batch = next(iter(loader))

    state = TrainingState()
    metrics = train_step_bytes(
        model=model,
        batch=batch,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        config=training_config,
        state=state,
        accumulation_step=0,
        source_modality="vision",
        target_modality="text",
    )

    print(f"Training step completed!")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Step: {state.step}")

    print("\nTraining components test passed!")
