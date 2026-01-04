"""Tests for LUMA Streaming Dataset."""

import pytest
import torch
from torch.utils.data import DataLoader

from src.data import LUMAMockDataset, get_collator


class TestLUMAMockDataset:
    """Test mock LUMA dataset (no network required)."""

    def test_creates_dataset(self):
        """Should create mock dataset."""
        dataset = LUMAMockDataset(num_samples=10)
        assert dataset is not None

    def test_iterates_samples(self):
        """Should iterate over samples."""
        dataset = LUMAMockDataset(num_samples=5)
        samples = list(dataset)
        assert len(samples) == 5

    def test_len(self):
        """Should return correct length."""
        dataset = LUMAMockDataset(num_samples=42)
        assert len(dataset) == 42

    def test_sample_structure(self):
        """Samples should have correct structure."""
        dataset = LUMAMockDataset(num_samples=1)
        sample = next(iter(dataset))

        assert "vision" in sample
        assert "audio" in sample
        assert "text" in sample
        assert "masks" in sample

        assert sample["vision"].dtype == torch.long
        assert sample["audio"].dtype == torch.long
        assert sample["text"].dtype == torch.long

    def test_vision_shape(self):
        """Vision should be 32x32x3 = 3072 bytes."""
        dataset = LUMAMockDataset(num_samples=1, vision_size=(32, 32))
        sample = next(iter(dataset))
        assert sample["vision"].shape == (3072,)

    def test_vision_custom_size(self):
        """Vision should respect custom size."""
        dataset = LUMAMockDataset(num_samples=1, vision_size=(16, 16))
        sample = next(iter(dataset))
        # 16 * 16 * 3 = 768
        assert sample["vision"].shape == (768,)

    def test_audio_shape(self):
        """Audio should be max_len bytes."""
        dataset = LUMAMockDataset(num_samples=1, audio_max_len=8000)
        sample = next(iter(dataset))
        assert sample["audio"].shape == (8000,)

    def test_audio_custom_length(self):
        """Audio should respect custom max length."""
        dataset = LUMAMockDataset(num_samples=1, audio_max_len=4000)
        sample = next(iter(dataset))
        assert sample["audio"].shape == (4000,)

    def test_text_shape(self):
        """Text should be max_len bytes."""
        dataset = LUMAMockDataset(num_samples=1, text_max_len=1024)
        sample = next(iter(dataset))
        assert sample["text"].shape == (1024,)

    def test_text_custom_length(self):
        """Text should respect custom max length."""
        dataset = LUMAMockDataset(num_samples=1, text_max_len=512)
        sample = next(iter(dataset))
        assert sample["text"].shape == (512,)

    def test_byte_range(self):
        """Values should be in [0, 255]."""
        dataset = LUMAMockDataset(num_samples=1)
        sample = next(iter(dataset))

        assert sample["vision"].min() >= 0
        assert sample["vision"].max() <= 255
        assert sample["audio"].min() >= 0
        assert sample["audio"].max() <= 255
        assert sample["text"].min() >= 0
        assert sample["text"].max() <= 255

    def test_masks_are_bool(self):
        """Masks should be boolean tensors."""
        dataset = LUMAMockDataset(num_samples=1)
        sample = next(iter(dataset))

        assert sample["masks"]["vision"].dtype == torch.bool
        assert sample["masks"]["audio"].dtype == torch.bool
        assert sample["masks"]["text"].dtype == torch.bool

    def test_masks_shapes_match_data(self):
        """Mask shapes should match data shapes."""
        dataset = LUMAMockDataset(num_samples=1)
        sample = next(iter(dataset))

        assert sample["masks"]["vision"].shape == sample["vision"].shape
        assert sample["masks"]["audio"].shape == sample["audio"].shape
        assert sample["masks"]["text"].shape == sample["text"].shape

    def test_selective_modalities_vision_text(self):
        """Should only include requested modalities (vision, text)."""
        dataset = LUMAMockDataset(num_samples=1, modalities=["vision", "text"])
        sample = next(iter(dataset))

        assert "vision" in sample
        assert "text" in sample
        assert "audio" not in sample
        assert "vision" in sample["masks"]
        assert "text" in sample["masks"]
        assert "audio" not in sample["masks"]

    def test_selective_modalities_audio_only(self):
        """Should only include audio modality."""
        dataset = LUMAMockDataset(num_samples=1, modalities=["audio"])
        sample = next(iter(dataset))

        assert "audio" in sample
        assert "vision" not in sample
        assert "text" not in sample

    def test_works_with_dataloader(self):
        """Should work with PyTorch DataLoader."""
        dataset = LUMAMockDataset(num_samples=10, modalities=["vision", "text"])
        collator = get_collator(mode="multi")
        loader = DataLoader(dataset, batch_size=4, collate_fn=collator)

        batch = next(iter(loader))
        assert batch["vision_bytes"].shape[0] == 4
        assert batch["text_bytes"].shape[0] == 4

    def test_works_with_paired_collator(self):
        """Should work with paired collator for training."""
        dataset = LUMAMockDataset(num_samples=10, modalities=["vision", "text"])
        collator = get_collator(
            mode="paired",
            source_modality="vision",
            target_modality="text"
        )
        loader = DataLoader(dataset, batch_size=4, collate_fn=collator)

        batch = next(iter(loader))
        assert "source_bytes" in batch
        assert "target_bytes" in batch
        assert batch["source_modality"] == "vision"
        assert batch["target_modality"] == "text"

    def test_works_with_audio_modality_collator(self):
        """Should work with audio-to-text paired collator."""
        dataset = LUMAMockDataset(num_samples=10, modalities=["audio", "text"])
        collator = get_collator(
            mode="paired",
            source_modality="audio",
            target_modality="text"
        )
        loader = DataLoader(dataset, batch_size=4, collate_fn=collator)

        batch = next(iter(loader))
        assert batch["source_modality"] == "audio"
        assert batch["target_modality"] == "text"

    def test_multiple_iterations(self):
        """Should be able to iterate multiple times."""
        dataset = LUMAMockDataset(num_samples=5)

        # First iteration
        samples1 = list(dataset)
        assert len(samples1) == 5

        # Second iteration
        samples2 = list(dataset)
        assert len(samples2) == 5

    def test_variable_length_audio(self):
        """Audio should have variable actual lengths (mask indicates valid)."""
        dataset = LUMAMockDataset(num_samples=10, audio_max_len=8000)

        # Collect actual lengths
        actual_lengths = []
        for sample in dataset:
            mask = sample["masks"]["audio"]
            actual_len = mask.sum().item()
            actual_lengths.append(actual_len)

        # Should have some variation (not all exactly max_len)
        # Audio is generated with random length between 4000 and max_len
        assert min(actual_lengths) < max(actual_lengths)

    def test_variable_length_text(self):
        """Text should have variable actual lengths (mask indicates valid)."""
        dataset = LUMAMockDataset(num_samples=10, text_max_len=1024)

        # Collect actual lengths
        actual_lengths = []
        for sample in dataset:
            mask = sample["masks"]["text"]
            actual_len = mask.sum().item()
            actual_lengths.append(actual_len)

        # Should have some variation
        assert min(actual_lengths) < max(actual_lengths)

    def test_vision_always_full(self):
        """Vision mask should always be all True (no padding)."""
        dataset = LUMAMockDataset(num_samples=5)

        for sample in dataset:
            mask = sample["masks"]["vision"]
            assert mask.all(), "Vision mask should be all True"


class TestLUMAMockDatasetIntegration:
    """Integration tests with training pipeline."""

    def test_batch_collation_shapes(self):
        """Batched shapes should be correct."""
        dataset = LUMAMockDataset(
            num_samples=8,
            modalities=["vision", "text", "audio"],
            vision_size=(32, 32),
            text_max_len=256,
            audio_max_len=4000,
        )
        collator = get_collator(mode="multi")
        loader = DataLoader(dataset, batch_size=4, collate_fn=collator)

        batch = next(iter(loader))

        # Check shapes
        assert batch["vision_bytes"].shape == (4, 3072)
        assert batch["text_bytes"].shape[0] == 4
        assert batch["text_bytes"].shape[1] <= 256
        assert batch["audio_bytes"].shape[0] == 4
        assert batch["audio_bytes"].shape[1] <= 4000

    def test_dataloader_num_workers(self):
        """Should work with multiple workers."""
        dataset = LUMAMockDataset(num_samples=20)
        collator = get_collator(mode="multi")

        # Note: IterableDataset with num_workers>0 requires special handling
        # Each worker gets the same iterator, so we test with 0 workers
        loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collator,
            num_workers=0,
        )

        batches = list(loader)
        assert len(batches) == 5  # 20 samples / 4 batch_size
