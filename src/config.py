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
import yaml
from pathlib import Path


# =============================================================================
# ARCHITECTURE CONSTANTS
# =============================================================================
# MLP dimension alignment for tensor core efficiency on modern GPUs.
# - BACKBONE_MLP_ALIGNMENT (256): Optimal for backbone's larger hidden dims
# - PREDICTOR_MLP_ALIGNMENT (64): Predictor uses narrower dims, smaller alignment

BACKBONE_MLP_ALIGNMENT = 256  # Backbone MLP hidden dim rounded to this
PREDICTOR_MLP_ALIGNMENT = 64  # Predictor MLP hidden dim rounded to this


# =============================================================================
# BYTE ENCODER CONFIGURATION
# =============================================================================

@dataclass
class ByteEncoderConfig:
    """Configuration for unified byte encoder."""
    vocab_size: int = 256  # All possible byte values (0-255)
    hidden_dim: int = 512
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_bias: bool = False
    max_seq_len: int = 8192  # Maximum byte sequence length

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
    """Configuration for JEPA predictor (cross-attention based)."""
    hidden_dim: int = 512  # Input/context dimension (matches encoder)
    num_layers: int = 4    # Number of predictor blocks
    num_heads: int = 8
    mlp_ratio: float = 4.0
    output_dim: int = 512  # Output dimension (matches target encoder)
    dropout: float = 0.0
    use_bias: bool = False
    use_gradient_checkpointing: bool = True
    max_seq_len: int = 8192  # For position embeddings

    @property
    def mlp_dim(self) -> int:
        return int(self.hidden_dim * self.mlp_ratio)

    @property
    def predictor_dim(self) -> int:
        """Internal predictor dimension (narrower than encoder)."""
        return self.hidden_dim // 2


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


# =============================================================================
# EMA CONFIGURATION
# =============================================================================

@dataclass
class EMAConfig:
    """Configuration for Exponential Moving Average target encoder."""
    # EMA decay schedule
    ema_decay_start: float = 0.996   # Initial decay (more updates)
    ema_decay_end: float = 0.9999    # Final decay (fewer updates)
    ema_warmup_steps: int = 10000    # Steps to reach final decay

    # Update frequency
    update_every: int = 1  # Update EMA every N optimizer steps


# =============================================================================
# DECODER CONFIGURATIONS
# =============================================================================

@dataclass
class DecoderConfig:
    """Base configuration for modality decoders."""
    input_dim: int = 512
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.0


@dataclass
class TextDecoderConfig(DecoderConfig):
    """Configuration for text (byte) decoder."""
    vocab_size: int = 256  # Output byte vocabulary
    max_output_len: int = 1024


@dataclass
class ImageDecoderConfig(DecoderConfig):
    """Configuration for image decoder."""
    output_size: Tuple[int, int] = (32, 32)  # Output image size
    output_channels: int = 3  # RGB

    @property
    def output_bytes(self) -> int:
        return self.output_size[0] * self.output_size[1] * self.output_channels


@dataclass
class AudioDecoderConfig(DecoderConfig):
    """Configuration for audio decoder."""
    output_length: int = 8000  # PCM bytes output length
    sample_rate: int = 16000


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
    variance_target: float = 1.0

    # Feature normalization
    normalize_predictions: bool = False
    normalize_targets: bool = False




# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)

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

    # Decoders (optional, for generation tasks)
    text_decoder: TextDecoderConfig = field(default_factory=TextDecoderConfig)
    image_decoder: ImageDecoderConfig = field(default_factory=ImageDecoderConfig)
    audio_decoder: AudioDecoderConfig = field(default_factory=AudioDecoderConfig)

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

        # Ensure decoders have correct input dim
        for decoder in [self.text_decoder, self.image_decoder, self.audio_decoder]:
            if decoder.input_dim != self.predictor.output_dim:
                decoder.input_dim = self.predictor.output_dim

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

        # Decoders (lightweight)
        decoder_params = 3 * (d * 256 + 256 * 256)

        return byte_encoder_params + backbone_params + predictor_params + decoder_params

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
            text_decoder=TextDecoderConfig(**d.get("text_decoder", {})),
            image_decoder=ImageDecoderConfig(**d.get("image_decoder", {})),
            audio_decoder=AudioDecoderConfig(**d.get("audio_decoder", {})),
            loss=LossConfig(**d.get("loss", {})),
            training=TrainingConfig(**d.get("training", {})),
            data=DataConfig(**d.get("data", {})),
            active_modalities=tuple(d.get("active_modalities", ("vision", "text", "audio"))),
        )

    def save(self, path: Path) -> None:
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, path: Path) -> "ByteJEPAConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)


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
        text_decoder=TextDecoderConfig(input_dim=256, hidden_dim=128),
        image_decoder=ImageDecoderConfig(input_dim=256, hidden_dim=128),
        audio_decoder=AudioDecoderConfig(input_dim=256, hidden_dim=128),
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
        text_decoder=TextDecoderConfig(input_dim=384, hidden_dim=192),
        image_decoder=ImageDecoderConfig(input_dim=384, hidden_dim=192),
        audio_decoder=AudioDecoderConfig(input_dim=384, hidden_dim=192),
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
