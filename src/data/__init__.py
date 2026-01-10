"""
Byte-level O-JEPA Data Module.

Provides byte-level data loading for training and evaluation.

DATA API CONTRACT
=================

All datasets return a dictionary with the following structure:

Required Keys (at least one modality):
    - "text": torch.Tensor[seq_len] - byte IDs (0-255) for text
    - "vision": torch.Tensor[H*W*3] - flattened RGB bytes for vision
    - "audio": torch.Tensor[samples] - audio bytes

Optional Keys:
    - "masks": Dict[str, torch.Tensor] - attention masks per modality
        - masks[modality]: torch.Tensor[seq_len], dtype=bool
        - True = valid token, False = padding

Example dataset output:
    {
        "text": tensor([72, 101, 108, 108, 111]),  # "Hello" in UTF-8
        "vision": tensor([...]),  # H*W*3 RGB bytes
        "masks": {
            "text": tensor([True, True, True, True, True]),
            "vision": tensor([True, True, ...]),
        }
    }

Collator output (after batching):
    {
        "text": tensor([B, max_seq_len]),  # padded batch
        "text_mask": tensor([B, max_seq_len]),  # attention mask
        "vision": tensor([B, H*W*3]),
        "vision_mask": tensor([B, H*W*3]),
    }

Training expects:
    - batch[modality]: torch.Tensor[B, seq_len] - byte IDs
    - batch[modality + "_mask"]: torch.Tensor[B, seq_len] - attention mask (optional)
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

# Byte-level collators
from .collator import (
    ByteCollator,
    PairedByteCollator,
    MultiModalByteCollator,
    get_collator,
)

__all__ = [
    # Byte datasets
    "ByteDataset",
    "SyntheticByteDataset",
    "PairedByteDataset",
    # LUMA datasets
    "LUMALocalDataset",
    "LUMAMockDataset",
    # Collators
    "ByteCollator",
    "PairedByteCollator",
    "MultiModalByteCollator",
    "get_collator",
]
