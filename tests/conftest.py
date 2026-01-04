"""
Byte-level O-JEPA Test Configuration and Fixtures.

Provides shared fixtures for testing byte-level O-JEPA components.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add o-jepa src to path
ojepa_root = Path(__file__).parent.parent
sys.path.insert(0, str(ojepa_root))

from src.config import (
    ByteJEPAConfig,
    ByteEncoderConfig,
    BackboneConfig,
    PredictorConfig,
    TextDecoderConfig,
    ImageDecoderConfig,
    AudioDecoderConfig,
    DataConfig,
    TrainingConfig,
    get_tiny_config,
    get_default_config,
)


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def tiny_config() -> ByteJEPAConfig:
    """Tiny configuration for fast testing."""
    return get_tiny_config()


@pytest.fixture
def default_config() -> ByteJEPAConfig:
    """Default configuration."""
    return get_default_config()


@pytest.fixture
def vision_only_config() -> ByteJEPAConfig:
    """Config with only vision modality."""
    return ByteJEPAConfig(
        byte_encoder=ByteEncoderConfig(
            hidden_dim=128,
            num_layers=1,
            num_heads=4,
        ),
        backbone=BackboneConfig(
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            use_gradient_checkpointing=False,
        ),
        predictor=PredictorConfig(
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            output_dim=128,
        ),
        text_decoder=TextDecoderConfig(input_dim=128, hidden_dim=64),
        image_decoder=ImageDecoderConfig(input_dim=128, hidden_dim=64),
        audio_decoder=AudioDecoderConfig(input_dim=128, hidden_dim=64),
        data=DataConfig(
            vision_seq_len=768,  # 16x16x3
            text_max_seq_len=256,
            audio_max_seq_len=1000,
            image_size=(16, 16),
        ),
        active_modalities=("vision",),
    )


@pytest.fixture
def vision_text_config() -> ByteJEPAConfig:
    """Config for vision-text training."""
    return ByteJEPAConfig(
        byte_encoder=ByteEncoderConfig(
            hidden_dim=128,
            num_layers=1,
            num_heads=4,
        ),
        backbone=BackboneConfig(
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            use_gradient_checkpointing=False,
        ),
        predictor=PredictorConfig(
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            output_dim=128,
        ),
        text_decoder=TextDecoderConfig(input_dim=128, hidden_dim=64),
        image_decoder=ImageDecoderConfig(input_dim=128, hidden_dim=64),
        audio_decoder=AudioDecoderConfig(input_dim=128, hidden_dim=64),
        data=DataConfig(
            vision_seq_len=768,
            text_max_seq_len=256,
            audio_max_seq_len=1000,
            image_size=(16, 16),
        ),
        active_modalities=("vision", "text"),
    )


# =============================================================================
# DATA FIXTURES
# =============================================================================

@pytest.fixture
def batch_size() -> int:
    """Default batch size for testing."""
    return 2


@pytest.fixture
def hidden_dim(tiny_config: ByteJEPAConfig) -> int:
    """Default hidden dimension from tiny_config."""
    return tiny_config.backbone.hidden_dim


@pytest.fixture
def dummy_vision_bytes(batch_size: int, tiny_config: ByteJEPAConfig) -> torch.Tensor:
    """Dummy vision bytes [batch, vision_seq_len]."""
    return torch.randint(0, 256, (batch_size, tiny_config.data.vision_seq_len), dtype=torch.long)


@pytest.fixture
def dummy_text_bytes(batch_size: int, tiny_config: ByteJEPAConfig) -> torch.Tensor:
    """Dummy text bytes [batch, text_max_seq_len]."""
    return torch.randint(0, 256, (batch_size, tiny_config.data.text_max_seq_len), dtype=torch.long)


@pytest.fixture
def dummy_audio_bytes(batch_size: int, tiny_config: ByteJEPAConfig) -> torch.Tensor:
    """Dummy audio bytes [batch, audio_max_seq_len]."""
    return torch.randint(0, 256, (batch_size, tiny_config.data.audio_max_seq_len), dtype=torch.long)


@pytest.fixture
def dummy_vision_mask(batch_size: int, tiny_config: ByteJEPAConfig) -> torch.Tensor:
    """Dummy vision mask [batch, vision_seq_len]."""
    return torch.ones(batch_size, tiny_config.data.vision_seq_len, dtype=torch.bool)


@pytest.fixture
def dummy_text_mask(batch_size: int, tiny_config: ByteJEPAConfig) -> torch.Tensor:
    """Dummy text mask [batch, text_max_seq_len]."""
    # Variable length text
    mask = torch.zeros(batch_size, tiny_config.data.text_max_seq_len, dtype=torch.bool)
    for i in range(batch_size):
        length = torch.randint(50, tiny_config.data.text_max_seq_len, (1,)).item()
        mask[i, :length] = True
    return mask


@pytest.fixture
def dummy_embeddings(batch_size: int, hidden_dim: int) -> torch.Tensor:
    """Dummy normalized embeddings [batch, hidden_dim]."""
    embed = torch.randn(batch_size, hidden_dim)
    return torch.nn.functional.normalize(embed, p=2, dim=-1)


@pytest.fixture
def dummy_sequence_embeddings(batch_size: int, hidden_dim: int) -> torch.Tensor:
    """Dummy sequence embeddings [batch, seq_len, hidden_dim]."""
    return torch.randn(batch_size, 64, hidden_dim)


# =============================================================================
# COMPONENT FIXTURES
# =============================================================================

@pytest.fixture
def byte_encoder(tiny_config: ByteJEPAConfig):
    """Create a tiny byte encoder."""
    from src.byte_encoder import ByteEncoder
    return ByteEncoder(tiny_config.byte_encoder)


@pytest.fixture
def backbone(tiny_config: ByteJEPAConfig):
    """Create a tiny backbone."""
    from src.backbone import SharedBackbone
    return SharedBackbone(tiny_config.backbone)


@pytest.fixture
def predictor(tiny_config: ByteJEPAConfig):
    """Create a tiny predictor."""
    from src.predictor import JEPAPredictor
    return JEPAPredictor(tiny_config.predictor)


@pytest.fixture
def loss_fn():
    """Create loss function."""
    from src.loss import JEPALoss, JEPALossConfig
    config = JEPALossConfig()
    return JEPALoss(config)


@pytest.fixture
def byte_jepa_model(tiny_config: ByteJEPAConfig):
    """Create a tiny ByteJEPA model."""
    from src.model import JEPAWorldModel
    return JEPAWorldModel(tiny_config)


# =============================================================================
# DECODER FIXTURES
# =============================================================================

@pytest.fixture
def text_decoder(tiny_config: ByteJEPAConfig):
    """Create a tiny text decoder."""
    from src.decoders import TextDecoder, TextDecoderConfig
    config = TextDecoderConfig(
        input_dim=tiny_config.backbone.hidden_dim,
        hidden_dim=64,
        num_layers=1,
        max_output_len=128,
    )
    return TextDecoder(config)


@pytest.fixture
def image_decoder(tiny_config: ByteJEPAConfig):
    """Create a tiny image decoder."""
    from src.decoders import ImageDecoder, ImageDecoderConfig
    config = ImageDecoderConfig(
        input_dim=tiny_config.backbone.hidden_dim,
        hidden_dim=64,
        output_size=(16, 16),
        output_channels=3,
    )
    return ImageDecoder(config)


@pytest.fixture
def audio_decoder(tiny_config: ByteJEPAConfig):
    """Create a tiny audio decoder."""
    from src.decoders import AudioDecoder, AudioDecoderConfig
    config = AudioDecoderConfig(
        input_dim=tiny_config.backbone.hidden_dim,
        hidden_dim=64,
        num_layers=1,
        output_length=1000,
    )
    return AudioDecoder(config)


# =============================================================================
# DEVICE FIXTURES
# =============================================================================

@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


# =============================================================================
# BATCH FIXTURES
# =============================================================================

@pytest.fixture
def paired_batch(dummy_vision_bytes, dummy_text_bytes, dummy_vision_mask, dummy_text_mask):
    """Create a paired batch for training."""
    return {
        "source_bytes": dummy_vision_bytes,
        "source_mask": dummy_vision_mask,
        "target_bytes": dummy_text_bytes,
        "target_mask": dummy_text_mask,
        "source_modality": "vision",
        "target_modality": "text",
    }


@pytest.fixture
def multimodal_batch(batch_size: int, tiny_config: ByteJEPAConfig):
    """Create a multi-modal batch."""
    return {
        "vision": torch.randint(0, 256, (batch_size, tiny_config.data.vision_seq_len), dtype=torch.long),
        "vision_mask": torch.ones(batch_size, tiny_config.data.vision_seq_len, dtype=torch.bool),
        "text": torch.randint(0, 256, (batch_size, tiny_config.data.text_max_seq_len), dtype=torch.long),
        "text_mask": torch.ones(batch_size, tiny_config.data.text_max_seq_len, dtype=torch.bool),
    }
