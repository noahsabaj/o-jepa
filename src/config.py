"""
JEPA World Model Configuration System

Configuration for byte-level JEPA world model following LeCun's vision.
Supports 3 modalities: vision (raw RGB bytes), text (UTF-8 bytes), audio (PCM bytes).

Key components:
- Encoder: Processes visible context
- Target Encoder: EMA copy, produces prediction targets
- Predictor: Predicts masked region embeddings
- Masking: Generates context/target splits
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any


# =============================================================================
# ARCHITECTURE CONSTANTS
# =============================================================================
# MLP dimension alignment for tensor core efficiency on modern GPUs.
# - BACKBONE_MLP_ALIGNMENT (256): Optimal for backbone's larger hidden dims
# - PREDICTOR_MLP_ALIGNMENT (64): Predictor uses narrower dims, smaller alignment

BACKBONE_MLP_ALIGNMENT = 256  # Backbone MLP hidden dim rounded to this
PREDICTOR_MLP_ALIGNMENT = 64  # Predictor MLP hidden dim rounded to this

# SwiGLU MLP scaling factor.
# Standard FFN: input -> 4x expansion -> output (2 projections: up + down)
# SwiGLU FFN:   input -> gate & up -> element-wise multiply -> output (3 projections)
#
# To match parameter count with standard FFN at 4x expansion:
#   Standard: 2 * (dim * 4*dim) = 8 * dim^2
#   SwiGLU:   3 * (dim * hidden) = 8 * dim^2  =>  hidden = 8/3 * dim = 2.67 * dim
#
# So SwiGLU uses 2/3 of the standard 4x expansion: 4 * 2/3 = 2.67x
# Reference: "GLU Variants Improve Transformer" (Shazeer, 2020)
SWIGLU_HIDDEN_RATIO = 2 / 3


# =============================================================================
# BYTE ENCODER CONFIGURATION
# =============================================================================

@dataclass
class ByteEncoderConfig:
    """Configuration for unified byte encoder with hierarchical multi-scale processing."""
    vocab_size: int = 256  # All possible byte values (0-255)
    hidden_dim: int = 512
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_bias: bool = False
    max_seq_len: int = 8192  # Maximum byte sequence length

    # Hierarchical multi-scale encoding (Loom-inspired)
    # Processes bytes at multiple scales with learned content-dependent gating
    hierarchical_scales: Tuple[int, ...] = (4, 16, 64)  # Kernel sizes in bytes
    hierarchical_gating: str = "softmax"  # Gating type: "softmax", "sigmoid", "linear"

    @property
    def mlp_dim(self) -> int:
        return int(self.hidden_dim * self.mlp_ratio)


# =============================================================================
# BACKBONE CONFIGURATION
# =============================================================================

@dataclass
class BackboneConfig:
    """Configuration for shared transformer backbone."""
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_bias: bool = False
    use_gradient_checkpointing: bool = True

    # Modalities (each gets a learned modality token)
    modalities: Tuple[str, ...] = ("vision", "text", "audio")

    @property
    def mlp_dim(self) -> int:
        return int(self.hidden_dim * self.mlp_ratio)


# =============================================================================
# PREDICTOR CONFIGURATION
# =============================================================================

@dataclass
class PredictorConfig:
    """
    Configuration for JEPA predictor (cross-attention based).

    Design Decision: Narrow Predictor
    ==================================
    The predictor is intentionally narrower than the encoder (default 0.5x).
    This forces the encoder to learn better representations since the
    predictor has limited capacity to "memorize" or "cheat."

    The predictor_ratio controls this:
    - 0.5 (default): Predictor is half the width of encoder
    - 0.33: Very narrow predictor (aggressive bottleneck)
    - 1.0: Same width as encoder (no bottleneck)

    Yann LeCun's I-JEPA uses a narrow predictor for this reason.
    """
    hidden_dim: int = 512  # Input/context dimension (matches encoder)
    num_layers: int = 4    # Number of predictor blocks
    num_heads: int = 8
    mlp_ratio: float = 4.0
    output_dim: int = 512  # Output dimension (matches target encoder)
    dropout: float = 0.0
    use_bias: bool = False
    use_gradient_checkpointing: bool = True
    max_seq_len: int = 8192  # For position embeddings

    # Predictor width relative to encoder (0.5 = half width, 1.0 = same width)
    predictor_ratio: float = 0.5

    # Y4: Optional query self-attention (I-JEPA doesn't use self-attention among queries)
    # True (default): Queries attend to each other before cross-attention
    # False: Pure I-JEPA style - queries only attend to context
    use_query_self_attention: bool = True

    @property
    def mlp_dim(self) -> int:
        return int(self.hidden_dim * self.mlp_ratio)

    @property
    def predictor_dim(self) -> int:
        """Internal predictor dimension (narrower than encoder by predictor_ratio)."""
        return int(self.hidden_dim * self.predictor_ratio)


# =============================================================================
# MASKING CONFIGURATION
# =============================================================================

@dataclass
class MaskingConfig:
    """Configuration for JEPA masking strategy."""
    # Target blocks (what we predict)
    num_target_blocks: int = 4
    target_scale_min: float = 0.15  # Each block is 15-40% of sequence
    target_scale_max: float = 0.40
    target_aspect_ratio: float = 1.0  # For 2D: width/height ratio

    # Context (what we see)
    context_scale_min: float = 0.85  # See 85% of input (after masking)
    context_scale_max: float = 1.0

    # Masking type
    masking_type: str = "span"  # "block" for 2D, "span" for 1D

    # Overlap control
    allow_target_overlap: bool = False

    # Y3: Hierarchical masking (curriculum learning)
    # When enabled, samples from both small and large scale ranges
    # This forces the model to learn predictions at multiple granularities
    use_hierarchical_masking: bool = False
    small_scale_range: Tuple[float, float] = (0.05, 0.15)  # Fine-grained targets
    large_scale_range: Tuple[float, float] = (0.40, 0.70)  # Coarse targets
    hierarchical_small_prob: float = 0.5  # Probability of sampling small scales


# =============================================================================
# EMA CONFIGURATION
# =============================================================================

@dataclass
class EMAConfig:
    """Configuration for Exponential Moving Average target encoder."""
    # EMA decay schedule
    ema_decay_initial: float = 0.996   # Initial decay (more updates to target)
    ema_decay_final: float = 0.9999   # Final decay (fewer updates, more stable)
    ema_warmup_steps: int = 10000    # Steps to reach final decay

    # Update frequency
    update_every: int = 1  # Update EMA every N optimizer steps


# =============================================================================
# LOSS CONFIGURATION
# =============================================================================

@dataclass
class LossConfig:
    """Configuration for JEPA loss function (non-contrastive)."""
    # Core loss type
    loss_type: str = "smooth_l1"  # "mse" or "smooth_l1"
    beta: float = 1.0  # For smooth_l1

    # Variance regularization (prevents collapse)
    use_variance_loss: bool = True
    variance_weight: float = 0.04
    # Matches natural std of neural network outputs (~1.0 with standard init);
    # yields ~0.03 loss for healthy embeddings, ~0.99 for collapsed embeddings
    variance_target: float = 1.0

    # Y2: VICReg-style redundancy loss (decorrelates features)
    # Prevents feature dimensions from becoming redundant/correlated
    # Minimizes off-diagonal covariance matrix elements
    use_redundancy_loss: bool = False  # Disabled by default for backwards compat
    redundancy_weight: float = 0.01

    # Feature normalization
    normalize_predictions: bool = False
    normalize_targets: bool = False




# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimizer - General
    learning_rate: float = 1e-4  # Base LR (used as reference for scheduler)
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)

    # Optimizer - Muon-specific
    # Muon (Momentum Orthogonalized by Newton-schulz) uses higher LR than AdamW
    # and works best on 2D weight matrices (excluding embeddings/norms)
    use_muon: bool = True  # Whether to use Muon for 2D weight params
    muon_lr: float = 1.5e-2  # Muon learning rate (10-20x higher than AdamW)
    adamw_lr: float = 5e-4  # AdamW learning rate for embeddings/norms/1D params
    muon_momentum: float = 0.95  # Muon momentum factor
    muon_nesterov: bool = True  # Use Nesterov momentum in Muon
    muon_ns_steps: int = 5  # Newton-Schulz iteration steps
    muon_weight_decay: float = 0.0  # Muon weight decay (often 0 works best)
    min_tokens_per_batch: int = 65536  # Warn if below this (Muon prefers large batches)

    # Scheduler (Warmup-Stable-Decay)
    warmup_ratio: float = 0.01  # 1% warmup
    stable_ratio: float = 0.79  # 79% stable
    decay_ratio: float = 0.20  # 20% decay
    min_lr_ratio: float = 0.0

    # Training
    total_steps: int = 100000
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Checkpointing
    save_every_steps: int = 5000
    eval_every_steps: int = 1000

    # Mixed precision
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def warmup_steps(self) -> int:
        return int(self.total_steps * self.warmup_ratio)


# =============================================================================
# DATA CONFIGURATION
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for byte-level data."""
    # Modality-specific sequence lengths (for "start small")
    vision_seq_len: int = 3072    # 32×32×3 RGB bytes
    text_max_seq_len: int = 1024  # ~1KB text
    audio_max_seq_len: int = 8000 # 0.5 sec @ 16kHz (16-bit = 2 bytes per sample)

    # Image settings
    image_size: Tuple[int, int] = (32, 32)

    # Audio settings
    audio_sample_rate: int = 16000

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

@dataclass
class ByteJEPAConfig:
    """Master configuration for JEPA World Model."""
    # Core components
    byte_encoder: ByteEncoderConfig = field(default_factory=ByteEncoderConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)

    # JEPA-specific (World Model components)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)

    # Loss and training
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Active modalities
    active_modalities: Tuple[str, ...] = ("vision", "text", "audio")

    def __post_init__(self):
        """Validate configuration consistency."""
        # Ensure byte encoder matches backbone hidden dim
        if self.byte_encoder.hidden_dim != self.backbone.hidden_dim:
            raise ValueError(
                f"ByteEncoder hidden_dim ({self.byte_encoder.hidden_dim}) must match "
                f"backbone hidden_dim ({self.backbone.hidden_dim})"
            )

        # Ensure predictor matches backbone
        if self.predictor.hidden_dim != self.backbone.hidden_dim:
            raise ValueError(
                f"Predictor hidden_dim ({self.predictor.hidden_dim}) must match "
                f"backbone hidden_dim ({self.backbone.hidden_dim})"
            )

    @property
    def hidden_dim(self) -> int:
        """Unified hidden dimension across all components."""
        return self.backbone.hidden_dim

    def get_seq_len(self, modality: str) -> int:
        """Get sequence length for a specific modality."""
        seq_lens = {
            "vision": self.data.vision_seq_len,
            "text": self.data.text_max_seq_len,
            "audio": self.data.audio_max_seq_len,
        }
        if modality not in seq_lens:
            raise ValueError(f"Unknown modality: {modality}")
        return seq_lens[modality]

    def estimate_parameters(self) -> int:
        """Estimate total parameter count."""
        d = self.hidden_dim

        # Byte encoder
        byte_encoder_params = (
            256 * d +  # Byte embedding
            self.byte_encoder.max_seq_len * d +  # Position embedding
            self.byte_encoder.num_layers * (4 * d * d + 2 * d * self.byte_encoder.mlp_dim)
        )

        # Backbone
        backbone_params = (
            len(self.backbone.modalities) * d +  # Modality tokens
            self.backbone.num_layers * (4 * d * d + 2 * d * self.backbone.mlp_dim)
        )

        # Predictor
        predictor_params = self.predictor.num_layers * (
            4 * d * d + 2 * d * self.predictor.mlp_dim
        )

        return byte_encoder_params + backbone_params + predictor_params

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ByteJEPAConfig":
        """Create config from dictionary."""
        return cls(
            byte_encoder=ByteEncoderConfig(**d.get("byte_encoder", {})),
            backbone=BackboneConfig(**d.get("backbone", {})),
            predictor=PredictorConfig(**d.get("predictor", {})),
            masking=MaskingConfig(**d.get("masking", {})),
            ema=EMAConfig(**d.get("ema", {})),
            loss=LossConfig(**d.get("loss", {})),
            training=TrainingConfig(**d.get("training", {})),
            data=DataConfig(**d.get("data", {})),
            active_modalities=tuple(d.get("active_modalities", ("vision", "text", "audio"))),
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def get_default_config() -> ByteJEPAConfig:
    """Get default ByteJEPA configuration."""
    return ByteJEPAConfig()


def get_tiny_config() -> ByteJEPAConfig:
    """Get tiny ByteJEPA configuration for testing."""
    return ByteJEPAConfig(
        byte_encoder=ByteEncoderConfig(
            hidden_dim=256,
            num_layers=1,
            num_heads=4,
            max_seq_len=4096,
        ),
        backbone=BackboneConfig(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            use_gradient_checkpointing=False,
        ),
        predictor=PredictorConfig(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            output_dim=256,
            max_seq_len=4096,
        ),
        masking=MaskingConfig(
            num_target_blocks=2,
            target_scale_min=0.1,
            target_scale_max=0.3,
        ),
        ema=EMAConfig(
            ema_warmup_steps=100,
        ),
        training=TrainingConfig(
            total_steps=1000,
            batch_size=2,
            gradient_accumulation_steps=2,
            use_gradient_checkpointing=False,
        ),
        data=DataConfig(
            vision_seq_len=768,   # 16×16×3
            text_max_seq_len=256,
            audio_max_seq_len=2000,
            image_size=(16, 16),
        ),
    )


def get_small_config() -> ByteJEPAConfig:
    """Get small ByteJEPA configuration for quick experiments."""
    return ByteJEPAConfig(
        byte_encoder=ByteEncoderConfig(
            hidden_dim=384,
            num_layers=2,
            num_heads=6,
        ),
        backbone=BackboneConfig(
            hidden_dim=384,
            num_layers=4,
            num_heads=6,
        ),
        predictor=PredictorConfig(
            hidden_dim=384,
            num_layers=3,
            num_heads=6,
            output_dim=384,
        ),
    )


if __name__ == "__main__":  # pragma: no cover
    # Test configuration
    print("ByteJEPA Configuration")
    print("=" * 50)

    config = get_default_config()
    print(f"Hidden dimension: {config.hidden_dim}")
    print(f"Active modalities: {config.active_modalities}")
    print(f"Byte encoder layers: {config.byte_encoder.num_layers}")
    print(f"Backbone layers: {config.backbone.num_layers}")
    print(f"Predictor layers: {config.predictor.num_layers}")
    print(f"Vision seq length: {config.data.vision_seq_len} bytes")
    print(f"Text max seq length: {config.data.text_max_seq_len} bytes")
    print(f"Audio max seq length: {config.data.audio_max_seq_len} bytes")
    print(f"Estimated parameters: {config.estimate_parameters():,}")

    print(f"\nTiny Configuration")
    print("=" * 50)
    tiny = get_tiny_config()
    print(f"Hidden dimension: {tiny.hidden_dim}")
    print(f"Estimated parameters: {tiny.estimate_parameters():,}")
