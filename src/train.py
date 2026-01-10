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
from typing import Dict, Optional, Any, Tuple, List, Union
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from .config import ByteJEPAConfig, TrainingConfig
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
# COMBINED OPTIMIZER (Muon + AdamW)
# =============================================================================

class CombinedOptimizer(torch.optim.Optimizer):
    """
    Combined optimizer that wraps Muon and AdamW for different parameter groups.

    Native PyTorch Muon (2.9+) only accepts strictly 2D parameters, so we need
    separate optimizers for 2D weights (Muon) and other params (AdamW).
    This wrapper makes them behave as a single optimizer.
    """

    def __init__(
        self,
        muon_optimizer: torch.optim.Optimizer,
        adamw_optimizer: torch.optim.Optimizer,
    ):
        # We don't call super().__init__ because we're wrapping optimizers
        # instead of managing params directly
        self.muon_optimizer = muon_optimizer
        self.adamw_optimizer = adamw_optimizer

        # Combined param_groups for compatibility (read-only view)
        self._param_groups = []
        for g in muon_optimizer.param_groups:
            g_copy = dict(g)
            g_copy['optimizer'] = 'muon'
            self._param_groups.append(g_copy)
        for g in adamw_optimizer.param_groups:
            g_copy = dict(g)
            g_copy['optimizer'] = 'adamw'
            self._param_groups.append(g_copy)

    @property
    def param_groups(self):
        """Return combined param groups."""
        return self._param_groups

    @property
    def state(self):
        """Return combined state dict (for compatibility)."""
        combined = {}
        combined.update(self.muon_optimizer.state)
        combined.update(self.adamw_optimizer.state)
        return combined

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients in both optimizers."""
        self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        self.adamw_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        """Step both optimizers."""
        loss = None
        if closure is not None:
            loss = closure()
        self.muon_optimizer.step()
        self.adamw_optimizer.step()
        return loss

    def state_dict(self):
        """Return combined state dict."""
        return {
            'muon': self.muon_optimizer.state_dict(),
            'adamw': self.adamw_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state dict into both optimizers."""
        self.muon_optimizer.load_state_dict(state_dict['muon'])
        self.adamw_optimizer.load_state_dict(state_dict['adamw'])


# =============================================================================
# OPTIMIZER CREATION
# =============================================================================

def create_optimizer(
    model: JEPAWorldModel,
    config: Optional[TrainingConfig] = None,
    # Legacy params for backward compatibility
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05,
    betas: Tuple[float, float] = (0.9, 0.95),
    use_muon: bool = True,
) -> torch.optim.Optimizer:
    """
    Create optimizer with Muon/AdamW hybrid.

    Muon (Momentum Orthogonalized by Newton-schulz) is used for 2D weight matrices
    for faster convergence with higher learning rates. AdamW is used for 1D params,
    embeddings, and norms with lower learning rates.

    For native PyTorch Muon (2.9+), we create a CombinedOptimizer that wraps
    separate Muon and AdamW optimizers since native Muon only accepts 2D params.

    Args:
        model: JEPAWorldModel model
        config: TrainingConfig with full optimizer settings (preferred)
        learning_rate: Legacy - base learning rate (used if config not provided)
        weight_decay: Legacy - weight decay (used if config not provided)
        betas: Legacy - Adam betas (used if config not provided)
        use_muon: Legacy - whether to use Muon (used if config not provided)

    Returns:
        Configured optimizer (may be CombinedOptimizer for native Muon)
    """
    # Use config if provided, otherwise fall back to legacy params
    if config is not None:
        use_muon = config.use_muon
        muon_lr = config.muon_lr
        adamw_lr = config.adamw_lr
        muon_momentum = config.muon_momentum
        muon_nesterov = config.muon_nesterov
        muon_ns_steps = config.muon_ns_steps
        muon_weight_decay = config.muon_weight_decay
        adamw_weight_decay = config.weight_decay
        betas = config.betas
        base_lr = config.learning_rate  # For scheduler reference
        min_tokens = config.min_tokens_per_batch
    else:
        # Legacy mode: use same LR for both (old behavior)
        muon_lr = learning_rate
        adamw_lr = learning_rate
        muon_momentum = 0.95
        muon_nesterov = True
        muon_ns_steps = 5
        muon_weight_decay = 0.0
        adamw_weight_decay = weight_decay
        base_lr = learning_rate
        min_tokens = 65536

    # Try to import Muon (native PyTorch 2.9+ first, then pytorch_optimizer)
    muon_available = False
    muon_class = None
    muon_backend = None

    if use_muon:
        # Try native PyTorch Muon first (available in PyTorch 2.9+)
        try:
            from torch.optim import Muon as NativeMuon
            muon_class = NativeMuon
            muon_available = True
            muon_backend = "native"
            print("Using native torch.optim.Muon (PyTorch 2.9+)")
        except ImportError:
            # Fall back to pytorch_optimizer
            try:
                from pytorch_optimizer import Muon as POMuon
                muon_class = POMuon
                muon_available = True
                muon_backend = "pytorch_optimizer"
                print("Using pytorch_optimizer.Muon")
            except ImportError:
                print("Warning: Muon not available (need PyTorch 2.9+ or pytorch-optimizer)")
                print("Falling back to AdamW")

    if muon_available:
        # Separate parameters for Muon vs AdamW
        muon_params = []  # Strictly 2D weight matrices (excluding embeddings/norms/heads)
        adamw_params = []  # Everything else: 1D, 3D+, embeddings, norms, heads

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Muon works best on strictly 2D weight matrices, excluding special params
            # Native PyTorch Muon ONLY accepts 2D params (ndim == 2 exactly)
            is_2d_weight = param.ndim == 2
            is_embedding = 'embed' in name.lower()
            is_norm = 'norm' in name.lower()
            is_head = 'head' in name.lower()  # Output heads excluded

            if is_2d_weight and not is_embedding and not is_norm and not is_head:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        muon_count = len(muon_params)
        adamw_count = len(adamw_params)
        print(
            f"Muon optimizer: {muon_count} params with Muon (lr={muon_lr:.2e}), "
            f"{adamw_count} params with AdamW (lr={adamw_lr:.2e})"
        )

        # Create optimizer based on backend
        if muon_backend == "native":
            # Native PyTorch Muon requires separate optimizers for 2D and non-2D params
            # Build Muon param group
            muon_param_groups = [
                {
                    'params': muon_params,
                    'lr': muon_lr,
                    'weight_decay': muon_weight_decay,
                    'name': 'muon_weights',
                    'lr_scale': muon_lr / base_lr if base_lr > 0 else 1.0,
                },
            ]

            # Build AdamW param group
            adamw_param_groups = [
                {
                    'params': adamw_params,
                    'lr': adamw_lr,
                    'weight_decay': adamw_weight_decay,
                    'name': 'adamw_params',
                    'lr_scale': adamw_lr / base_lr if base_lr > 0 else 1.0,
                },
            ]

            # Filter empty groups
            muon_param_groups = [g for g in muon_param_groups if len(g['params']) > 0]
            adamw_param_groups = [g for g in adamw_param_groups if len(g['params']) > 0]

            # Create separate optimizers
            if muon_param_groups and adamw_param_groups:
                muon_opt = muon_class(
                    muon_param_groups,
                    lr=muon_lr,
                    momentum=muon_momentum,
                    nesterov=muon_nesterov,
                    ns_steps=muon_ns_steps,
                    weight_decay=muon_weight_decay,
                )
                adamw_opt = torch.optim.AdamW(
                    adamw_param_groups,
                    lr=adamw_lr,
                    betas=betas,
                    weight_decay=adamw_weight_decay,
                    fused=torch.cuda.is_available(),
                )
                optimizer = CombinedOptimizer(muon_opt, adamw_opt)
            elif muon_param_groups:
                # Only Muon params (unlikely)
                optimizer = muon_class(
                    muon_param_groups,
                    lr=muon_lr,
                    momentum=muon_momentum,
                    nesterov=muon_nesterov,
                    ns_steps=muon_ns_steps,
                    weight_decay=muon_weight_decay,
                )
            else:
                # Only AdamW params (unlikely)
                optimizer = torch.optim.AdamW(
                    adamw_param_groups,
                    lr=adamw_lr,
                    betas=betas,
                    weight_decay=adamw_weight_decay,
                    fused=torch.cuda.is_available(),
                )
        else:
            # pytorch_optimizer Muon handles mixed param groups directly
            param_groups = [
                {
                    'params': muon_params,
                    'lr': muon_lr,
                    'weight_decay': muon_weight_decay,
                    'use_muon': True,
                    'name': 'muon_weights',
                    'lr_scale': muon_lr / base_lr if base_lr > 0 else 1.0,
                },
                {
                    'params': adamw_params,
                    'lr': adamw_lr,
                    'weight_decay': adamw_weight_decay,
                    'use_muon': False,
                    'name': 'adamw_params',
                    'lr_scale': adamw_lr / base_lr if base_lr > 0 else 1.0,
                },
            ]
            param_groups = [g for g in param_groups if len(g['params']) > 0]

            optimizer = muon_class(
                param_groups,
                lr=muon_lr,
                betas=betas,
                weight_decay=muon_weight_decay,
                backend='newtonschulz5',
            )

        # Warn about batch size if config provided
        if config is not None:
            effective = config.effective_batch_size
            if effective < 64:  # Very rough heuristic
                warnings.warn(
                    f"Muon works best with large batches. Current effective batch: {effective}. "
                    f"Consider increasing batch_size or gradient_accumulation_steps."
                )

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

    # Use adamw_lr for all groups in fallback mode
    param_groups = [
        {
            'params': predictor_2d_params,
            'lr': adamw_lr,
            'weight_decay': adamw_weight_decay,
            'lr_scale': adamw_lr / base_lr if base_lr > 0 else 1.0,
            'name': 'predictor_2d',
        },
        {
            'params': predictor_other_params,
            'lr': adamw_lr,
            'weight_decay': adamw_weight_decay,
            'lr_scale': adamw_lr / base_lr if base_lr > 0 else 1.0,
            'name': 'predictor_other',
        },
        {
            'params': backbone_params,
            'lr': adamw_lr,
            'weight_decay': adamw_weight_decay,
            'lr_scale': adamw_lr / base_lr if base_lr > 0 else 1.0,
            'name': 'backbone',
        },
        {
            'params': encoder_params,
            'lr': adamw_lr * 0.5,  # Slightly reduced for encoder
            'weight_decay': adamw_weight_decay,
            'lr_scale': (adamw_lr * 0.5) / base_lr if base_lr > 0 else 0.5,
            'name': 'byte_encoder',
        },
        {
            'params': decoder_params,
            'lr': adamw_lr,
            'weight_decay': adamw_weight_decay,
            'lr_scale': adamw_lr / base_lr if base_lr > 0 else 1.0,
            'name': 'decoders',
        },
    ]

    # Filter empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]

    # Use fused AdamW for CUDA speedup when available
    use_fused = torch.cuda.is_available()

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=adamw_lr,
        betas=betas,
        weight_decay=adamw_weight_decay,
        fused=use_fused,
    )

    if use_fused:
        print("Using fused AdamW optimizer (CUDA-optimized)")

    return optimizer


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

@dataclass
class TrainingState:
    """Mutable training state."""
    step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    total_samples: int = 0




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
    modality: str = "text",
    reporter: Optional[ProgressReporter] = None,
):
    """
    Main training loop for JEPA World Model.

    Uses single-modality masked prediction (JEPA style):
    - Mask portions of input sequence
    - Predict masked embeddings from context
    - EMA target encoder provides stable targets

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
        modality: Modality to train on ("text", "vision", "audio")
        reporter: Progress reporter for training output
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
        modality=modality,
    )

    print(f"JEPA World Model training: modality={modality}")

    reporter.on_train_start(model_info, training_info)

    model.train()
    accumulation_step = 0
    running_loss = 0.0
    start_time = time.time()

    while state.step < training_config.total_steps:
        for batch in train_loader:
            if state.step >= training_config.total_steps:
                break

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
                modality=modality,
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

                # JEPA World Model metrics
                log_metrics.update({
                    'pred_loss': metrics.get('pred_loss'),
                    'var_loss': metrics.get('var_loss'),
                    'std_mean': metrics.get('std_mean'),
                    'cosine_sim': metrics.get('cosine_sim'),
                    'ema_decay': metrics.get('ema_decay'),
                    'num_targets': metrics.get('num_targets'),
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
                    source_modality=modality,
                    target_modality=modality,
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


def compute_retrieval_metrics(
    pred_embeds: torch.Tensor,
    target_embeds: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute retrieval metrics (R@1, R@5, R@10).

    For each prediction embedding, find the nearest target embedding
    and check if it matches the ground truth.

    Args:
        pred_embeds: Prediction embeddings [N, D]
        target_embeds: Target embeddings [N, D]

    Returns:
        Dictionary with R@1, R@5, R@10 metrics
    """
    # Normalize embeddings
    pred_embeds = torch.nn.functional.normalize(pred_embeds, dim=-1)
    target_embeds = torch.nn.functional.normalize(target_embeds, dim=-1)

    # Compute cosine similarity matrix [N, N]
    similarity = torch.matmul(pred_embeds, target_embeds.T)

    # For each prediction, rank targets by similarity
    _, indices = similarity.topk(k=min(10, similarity.shape[1]), dim=1)

    # Ground truth is diagonal (pred[i] should match target[i])
    ground_truth = torch.arange(len(pred_embeds), device=pred_embeds.device)

    # Compute R@k
    r_at_1 = (indices[:, 0] == ground_truth).float().mean().item()
    r_at_5 = (indices[:, :5] == ground_truth.unsqueeze(1)).any(dim=1).float().mean().item()
    r_at_10 = (indices == ground_truth.unsqueeze(1)).any(dim=1).float().mean().item()

    return {
        'R@1': r_at_1,
        'R@5': r_at_5,
        'R@10': r_at_10,
    }


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
    metrics = train_step_jepa(
        model=model,
        batch=batch,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        config=training_config,
        state=state,
        accumulation_step=0,
        modality="vision",
    )

    print(f"Training step completed!")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Step: {state.step}")

    print("\nTraining components test passed!")
