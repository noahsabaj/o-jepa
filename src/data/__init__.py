"""
Byte-level O-JEPA Data Module.

Provides byte-level data loading for training and evaluation.
"""

# Byte-level datasets
from .byte_dataset import (
    ByteDataset,
    SyntheticByteDataset,
    PairedByteDataset,
)

# LUMA datasets
from .luma_dataset import (
    LUMALocalDataset,
    LUMAMockDataset,
)

# Video+Audio datasets (universal dock)
from .video_dataset import (
    VideoAudioDataset,
    VideoAudioMockDataset,
    get_format_adapter,
)

# Format adapters
from .formats import (
    FormatAdapter,
    ClipInfo,
    GenericVideoAdapter,
    EgoExo4DAdapter,
    FORMAT_REGISTRY,
    register_format,
    get_available_formats,
)

# Byte-level collators
from .collator import (
    ByteCollator,
    PairedByteCollator,
    MultiModalByteCollator,
    get_collator,
)

# Byte-level transforms
from .transforms import (
    ByteNoise,
    ByteMask,
    ByteShift,
    ByteCrop,
    ImageByteFlip,
    ComposeByteTransforms,
    get_byte_vision_transforms,
    get_byte_text_transforms,
    get_byte_audio_transforms,
    get_all_byte_transforms,
)

__all__ = [
    # Byte datasets
    "ByteDataset",
    "SyntheticByteDataset",
    "PairedByteDataset",
    # LUMA datasets
    "LUMALocalDataset",
    "LUMAMockDataset",
    # Video+Audio datasets
    "VideoAudioDataset",
    "VideoAudioMockDataset",
    "get_format_adapter",
    # Format adapters
    "FormatAdapter",
    "ClipInfo",
    "GenericVideoAdapter",
    "EgoExo4DAdapter",
    "FORMAT_REGISTRY",
    "register_format",
    "get_available_formats",
    # Collators
    "ByteCollator",
    "PairedByteCollator",
    "MultiModalByteCollator",
    "get_collator",
    # Transforms
    "ByteNoise",
    "ByteMask",
    "ByteShift",
    "ByteCrop",
    "ImageByteFlip",
    "ComposeByteTransforms",
    "get_byte_vision_transforms",
    "get_byte_text_transforms",
    "get_byte_audio_transforms",
    "get_all_byte_transforms",
]
