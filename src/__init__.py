"""
JEPA World Model: Byte-Level Joint Embedding Predictive Architecture

Following Yann LeCun's vision: predict abstract representations, not pixels.
The model learns to understand the world by predicting masked region embeddings.
"""

from .config import (
    ByteJEPAConfig,
    ByteEncoderConfig,
    BackboneConfig,
    PredictorConfig,
    MaskingConfig,
    EMAConfig,
    LossConfig,
    TextDecoderConfig,
    ImageDecoderConfig,
    AudioDecoderConfig,
    get_default_config,
    get_tiny_config,
)
from .model import JEPAWorldModel
from .byte_encoder import ByteEncoder
from .backbone import SharedBackbone
from .predictor import JEPAPredictor
from .masking import BlockMaskGenerator, SpanMaskGenerator
from .loss import JEPALoss

__version__ = "0.4.0"  # Cleanup: removed backward compat aliases
__all__ = [
    # Main model
    "JEPAWorldModel",
    # Config
    "ByteJEPAConfig",
    "MaskingConfig",
    "EMAConfig",
    "LossConfig",
    # Components
    "ByteEncoder",
    "SharedBackbone",
    "JEPAPredictor",
    "BlockMaskGenerator",
    "SpanMaskGenerator",
    "JEPALoss",
    # Factory functions
    "get_default_config",
    "get_tiny_config",
]
