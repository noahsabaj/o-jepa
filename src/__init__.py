"""
O-JEPA: Omni-modality Joint Embedding Predictive Architecture

Following Yann LeCun's vision: predict abstract representations, not pixels.
The model learns to understand the world by predicting masked region embeddings.
All modalities pass through the world model. Qwen speaks.
"""

from .config import (
    ByteJEPAConfig,
    ByteEncoderConfig,
    BackboneConfig,
    PredictorConfig,
    MaskingConfig,
    EMAConfig,
    LossConfig,
    LanguageInterfaceConfig,
    get_default_config,
    get_tiny_config,
)
from .model import JEPAWorldModel
from .byte_encoder import ByteEncoder
from .backbone import SharedBackbone
from .predictor import JEPAPredictor
from .masking import BlockMaskGenerator, SpanMaskGenerator
from .loss import JEPALoss
from .language_interface import LanguageInterface, WorldToLanguageProjection

__version__ = "0.5.0"  # Language interface: O-JEPA speaks through Qwen
__all__ = [
    # Main model
    "JEPAWorldModel",
    # Language interface
    "LanguageInterface",
    "WorldToLanguageProjection",
    # Config
    "ByteJEPAConfig",
    "LanguageInterfaceConfig",
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
