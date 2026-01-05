"""
Collation functions for byte-level O-JEPA.

Handles batching of variable-length byte sequences.
"""

from typing import Dict, List, Optional
import torch


class ByteCollator:
    """
    Collate byte sequences into batches.

    Handles padding for variable-length sequences within each modality.

    Args:
        pad_value: Value to use for padding (default 0)
        modalities: List of modality names to collate
    """

    def __init__(
        self,
        pad_value: int = 0,
        modalities: List[str] = ["vision", "text", "audio"],
    ):
        self.pad_value = pad_value
        self.modalities = modalities

    def __call__(
        self,
        batch: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.

        Args:
            batch: List of sample dictionaries

        Returns:
            Dictionary with batched tensors and masks
        """
        result = {}

        for modality in self.modalities:
            # Collect all tensors for this modality
            tensors = [s[modality] for s in batch if modality in s]
            if not tensors:
                continue

            # Check if padding is needed (sequences might have different lengths)
            lengths = [len(t) for t in tensors]
            max_len = max(lengths)

            if all(l == max_len for l in lengths):
                # Same length: just stack
                result[modality] = torch.stack(tensors)
            else:
                # Pad to max length
                padded = []
                for t in tensors:
                    if len(t) < max_len:
                        pad = torch.full(
                            (max_len - len(t),),
                            self.pad_value,
                            dtype=t.dtype,
                        )
                        t = torch.cat([t, pad])
                    padded.append(t)
                result[modality] = torch.stack(padded)

            # Create attention mask
            mask_key = f"{modality}_mask"
            if "masks" in batch[0] and modality in batch[0]["masks"]:
                # Use provided masks
                masks = [s["masks"][modality] for s in batch]
                # Pad masks too
                padded_masks = []
                for m in masks:
                    if len(m) < max_len:
                        pad = torch.zeros(max_len - len(m), dtype=m.dtype)
                        m = torch.cat([m, pad])
                    padded_masks.append(m)
                result[mask_key] = torch.stack(padded_masks)
            else:
                # Create mask from lengths
                mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
                for i, length in enumerate(lengths):
                    mask[i, :length] = True
                result[mask_key] = mask

        return result


class PairedByteCollator:
    """
    Collate paired samples (e.g., vision-text pairs).

    Optimized for contrastive learning where we need
    source and target modalities.

    Args:
        source_modality: Name of source modality
        target_modality: Name of target modality
        pad_value: Padding value
    """

    def __init__(
        self,
        source_modality: str = "vision",
        target_modality: str = "text",
        pad_value: int = 0,
    ):
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.pad_value = pad_value

    def __call__(
        self,
        batch: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Collate paired samples.

        Args:
            batch: List of sample dictionaries

        Returns:
            Dictionary with source_bytes, target_bytes, and masks
        """
        result = {}

        # Filter batch to only include items with both modalities
        valid_batch = [
            s for s in batch
            if self.source_modality in s and self.target_modality in s
        ]

        if not valid_batch:
            raise ValueError(
                f"No valid samples in batch with both '{self.source_modality}' "
                f"and '{self.target_modality}' modalities. "
                f"Batch keys: {[list(s.keys()) for s in batch[:3]]}"
            )

        if len(valid_batch) < len(batch):
            # Log warning about filtered samples (only visible if debugging)
            pass  # Some samples were filtered

        batch = valid_batch

        # Collate source modality
        source_tensors = [s[self.source_modality] for s in batch]
        source_lengths = [len(t) for t in source_tensors]
        source_max_len = max(source_lengths)

        source_padded = []
        source_masks = []
        for t in source_tensors:
            mask = torch.ones(source_max_len, dtype=torch.bool)
            if len(t) < source_max_len:
                mask[len(t):] = False
                pad = torch.full((source_max_len - len(t),), self.pad_value, dtype=t.dtype)
                t = torch.cat([t, pad])
            source_padded.append(t)
            source_masks.append(mask)

        result["source_bytes"] = torch.stack(source_padded)
        result["source_mask"] = torch.stack(source_masks)
        result["source_modality"] = self.source_modality

        # Collate target modality
        target_tensors = [s[self.target_modality] for s in batch]
        target_lengths = [len(t) for t in target_tensors]
        target_max_len = max(target_lengths)

        target_padded = []
        target_masks = []
        for t in target_tensors:
            mask = torch.ones(target_max_len, dtype=torch.bool)
            if len(t) < target_max_len:
                mask[len(t):] = False
                pad = torch.full((target_max_len - len(t),), self.pad_value, dtype=t.dtype)
                t = torch.cat([t, pad])
            target_padded.append(t)
            target_masks.append(mask)

        result["target_bytes"] = torch.stack(target_padded)
        result["target_mask"] = torch.stack(target_masks)
        result["target_modality"] = self.target_modality

        return result


class MultiModalByteCollator:
    """
    Collate for multi-modal training with all three modalities.

    Args:
        modalities: Tuple of modality names
        pad_value: Padding value
    """

    def __init__(
        self,
        modalities: tuple = ("vision", "text", "audio"),
        pad_value: int = 0,
    ):
        self.modalities = modalities
        self.pad_value = pad_value

    def __call__(
        self,
        batch: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Collate samples with all modalities.

        Args:
            batch: List of sample dictionaries

        Returns:
            Dictionary with all modalities batched
        """
        result = {}

        for modality in self.modalities:
            tensors = [s.get(modality) for s in batch if modality in s]
            if not tensors:
                continue

            # Find max length
            lengths = [len(t) for t in tensors]
            max_len = max(lengths)

            # Pad and stack
            padded = []
            masks = []
            for t in tensors:
                mask = torch.ones(max_len, dtype=torch.bool)
                if len(t) < max_len:
                    mask[len(t):] = False
                    pad = torch.full((max_len - len(t),), self.pad_value, dtype=t.dtype)
                    t = torch.cat([t, pad])
                padded.append(t)
                masks.append(mask)

            result[f"{modality}_bytes"] = torch.stack(padded)
            result[f"{modality}_mask"] = torch.stack(masks)

        return result


def get_collator(
    mode: str = "paired",
    source_modality: str = "vision",
    target_modality: str = "text",
    modalities: tuple = ("vision", "text", "audio"),
    pad_value: int = 0,
):
    """
    Factory function for collators.

    Args:
        mode: "paired" for paired data, "multi" for multi-modal, "basic" for basic
        source_modality: Source modality for paired mode
        target_modality: Target modality for paired mode
        modalities: Tuple of modalities for basic/multi modes
        pad_value: Padding value

    Returns:
        Appropriate collator instance
    """
    if mode == "paired":
        return PairedByteCollator(source_modality, target_modality, pad_value)
    elif mode == "multi":
        return MultiModalByteCollator(modalities, pad_value)
    else:
        return ByteCollator(pad_value, list(modalities))
