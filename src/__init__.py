"""
O-JEPA: Omni-modality Joint Embedding Predictive Architecture

Following Yann LeCun's vision: predict abstract representations, not pixels.
The model learns to understand the world by predicting masked region embeddings.
All modalities pass through the unified byte encoder with hierarchical processing.
"""

from .config import (
    ByteJEPAConfig,
    ByteEncoderConfig,
    BackboneConfig,
    PredictorConfig,
    MaskingConfig,
    EMAConfig,
    LossConfig,
    get_default_config,
    get_tiny_config,
)
from .model import JEPAWorldModel
from .byte_encoder import ByteEncoder, HierarchicalChunking
from .backbone import SharedBackbone
from .predictor import ByteJEPAPredictor
from .masking import BlockMaskGenerator, SpanMaskGenerator
from .loss import JEPALoss

__version__ = "0.6.0"  # Hierarchical byte encoding is THE path
__all__ = [
    # Main model
    "JEPAWorldModel",
    # Config
    "ByteJEPAConfig",
    "ByteEncoderConfig",
    "BackboneConfig",
    "PredictorConfig",
    "MaskingConfig",
    "EMAConfig",
    "LossConfig",
    # Components
    "ByteEncoder",
    "HierarchicalChunking",
    "SharedBackbone",
    "ByteJEPAPredictor",
    "BlockMaskGenerator",
    "SpanMaskGenerator",
    "JEPALoss",
    # Factory functions
    "get_default_config",
    "get_tiny_config",
]
