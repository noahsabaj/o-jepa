"""Tests for byte-level data loading and transforms."""

import pytest
import torch
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader

from src.data import (
    SyntheticByteDataset,
    ByteCollator,
    PairedByteCollator,
    MultiModalByteCollator,
    get_collator,
    ByteNoise,
    ByteMask,
    ByteShift,
    ByteCrop,
    ImageByteFlip,
    ComposeByteTransforms,
    get_byte_vision_transforms,
    get_byte_text_transforms,
    get_byte_audio_transforms,
)
from src.data.byte_dataset import ByteDataset, PairedByteDataset


class TestSyntheticByteDataset:
    """Tests for SyntheticByteDataset."""

    def test_length(self):
        """Test dataset length."""
        dataset = SyntheticByteDataset(num_samples=100)
        assert len(dataset) == 100

    def test_item_structure(self):
        """Test item structure."""
        dataset = SyntheticByteDataset(
            num_samples=10,
            modalities=["vision", "text"],
        )
        item = dataset[0]

        assert "vision" in item
        assert "text" in item
        assert "masks" in item

    def test_byte_range(self):
        """Test bytes are in valid range."""
        dataset = SyntheticByteDataset(num_samples=10)
        item = dataset[0]

        for modality in ["vision", "text", "audio"]:
            if modality in item:
                assert item[modality].min() >= 0
                assert item[modality].max() <= 255

    def test_sequence_lengths(self):
        """Test sequence lengths match config."""
        vision_len = 768
        text_len = 256
        audio_len = 2000

        dataset = SyntheticByteDataset(
            num_samples=10,
            vision_seq_len=vision_len,
            text_seq_len=text_len,
            audio_seq_len=audio_len,
        )
        item = dataset[0]

        assert item["vision"].shape[0] == vision_len
        assert item["text"].shape[0] == text_len
        assert item["audio"].shape[0] == audio_len

    def test_masks_all_ones(self):
        """Test that masks are all ones for synthetic data."""
        dataset = SyntheticByteDataset(num_samples=10)
        item = dataset[0]

        for modality in ["vision", "text", "audio"]:
            if modality in item["masks"]:
                assert item["masks"][modality].all()

    def test_modalities_selection(self):
        """Test modality selection."""
        dataset = SyntheticByteDataset(
            num_samples=10,
            modalities=["vision"],
        )
        item = dataset[0]

        assert "vision" in item
        # Only vision should be in the main dict
        assert len([k for k in item.keys() if k != "masks"]) == 1

    def test_dtype_is_long(self):
        """Test that byte tensors are long dtype."""
        dataset = SyntheticByteDataset(num_samples=10)
        item = dataset[0]

        for modality in ["vision", "text", "audio"]:
            if modality in item:
                assert item[modality].dtype == torch.long


class TestByteCollator:
    """Tests for ByteCollator."""

    def test_basic_collation(self):
        """Test basic collation."""
        collator = ByteCollator(modalities=["vision", "text"])

        batch = [
            {"vision": torch.randint(0, 256, (100,), dtype=torch.long)},
            {"vision": torch.randint(0, 256, (100,), dtype=torch.long)},
        ]

        result = collator(batch)

        assert "vision" in result
        assert result["vision"].shape[0] == 2

    def test_padding(self):
        """Test padding for variable lengths."""
        collator = ByteCollator(modalities=["text"])

        batch = [
            {"text": torch.randint(0, 256, (50,), dtype=torch.long)},
            {"text": torch.randint(0, 256, (100,), dtype=torch.long)},
        ]

        result = collator(batch)

        assert result["text"].shape == (2, 100)  # Padded to max length

    def test_mask_creation(self):
        """Test mask creation for padded sequences."""
        collator = ByteCollator(modalities=["text"])

        batch = [
            {"text": torch.randint(0, 256, (50,), dtype=torch.long)},
            {"text": torch.randint(0, 256, (100,), dtype=torch.long)},
        ]

        result = collator(batch)

        assert "text_mask" in result
        # First sample should have 50 True, rest False
        assert result["text_mask"][0, :50].all()
        assert not result["text_mask"][0, 50:].any()

    def test_custom_pad_value(self):
        """Test custom padding value."""
        collator = ByteCollator(modalities=["text"], pad_value=255)

        batch = [
            {"text": torch.zeros(50, dtype=torch.long)},
            {"text": torch.zeros(100, dtype=torch.long)},
        ]

        result = collator(batch)

        # Padding should be 255
        assert (result["text"][0, 50:] == 255).all()


class TestPairedByteCollator:
    """Tests for PairedByteCollator."""

    def test_paired_collation(self):
        """Test paired collation."""
        collator = PairedByteCollator(
            source_modality="vision",
            target_modality="text",
        )

        batch = [
            {
                "vision": torch.randint(0, 256, (768,), dtype=torch.long),
                "text": torch.randint(0, 256, (256,), dtype=torch.long),
            },
            {
                "vision": torch.randint(0, 256, (768,), dtype=torch.long),
                "text": torch.randint(0, 256, (256,), dtype=torch.long),
            },
        ]

        result = collator(batch)

        assert "source_bytes" in result
        assert "target_bytes" in result
        assert "source_mask" in result
        assert "target_mask" in result

    def test_modality_names(self):
        """Test modality names in result."""
        collator = PairedByteCollator(
            source_modality="vision",
            target_modality="text",
        )

        batch = [
            {
                "vision": torch.randint(0, 256, (768,), dtype=torch.long),
                "text": torch.randint(0, 256, (256,), dtype=torch.long),
            },
        ]

        result = collator(batch)

        assert result["source_modality"] == "vision"
        assert result["target_modality"] == "text"

    def test_variable_lengths(self):
        """Test padding with variable lengths."""
        collator = PairedByteCollator(
            source_modality="vision",
            target_modality="text",
        )

        batch = [
            {
                "vision": torch.randint(0, 256, (768,), dtype=torch.long),
                "text": torch.randint(0, 256, (100,), dtype=torch.long),
            },
            {
                "vision": torch.randint(0, 256, (768,), dtype=torch.long),
                "text": torch.randint(0, 256, (200,), dtype=torch.long),
            },
        ]

        result = collator(batch)

        # Target should be padded to max length (200)
        assert result["target_bytes"].shape == (2, 200)
        # Masks should reflect actual lengths
        assert result["target_mask"][0, :100].all()
        assert not result["target_mask"][0, 100:].any()

    def test_filters_missing_modalities(self):
        """Test that samples missing required modalities are filtered."""
        collator = PairedByteCollator(
            source_modality="vision",
            target_modality="text",
        )

        # Mix of valid and invalid samples
        batch = [
            {
                "vision": torch.randint(0, 256, (768,), dtype=torch.long),
                "text": torch.randint(0, 256, (256,), dtype=torch.long),
            },
            {
                # Missing vision
                "text": torch.randint(0, 256, (256,), dtype=torch.long),
            },
            {
                "vision": torch.randint(0, 256, (768,), dtype=torch.long),
                "text": torch.randint(0, 256, (256,), dtype=torch.long),
            },
        ]

        result = collator(batch)

        # Only 2 valid samples should be in result
        assert result["source_bytes"].shape[0] == 2
        assert result["target_bytes"].shape[0] == 2

    def test_raises_on_all_invalid_samples(self):
        """Test that ValueError is raised when no valid samples exist."""
        collator = PairedByteCollator(
            source_modality="vision",
            target_modality="text",
        )

        # All samples missing at least one required modality
        batch = [
            {
                # Missing vision
                "text": torch.randint(0, 256, (256,), dtype=torch.long),
            },
            {
                # Missing text
                "vision": torch.randint(0, 256, (768,), dtype=torch.long),
            },
        ]

        with pytest.raises(ValueError, match="No valid samples"):
            collator(batch)


class TestMultiModalByteCollator:
    """Tests for MultiModalByteCollator."""

    def test_multimodal_collation(self):
        """Test multi-modal collation."""
        collator = MultiModalByteCollator(modalities=("vision", "text", "audio"))

        batch = [
            {
                "vision": torch.randint(0, 256, (768,), dtype=torch.long),
                "text": torch.randint(0, 256, (256,), dtype=torch.long),
                "audio": torch.randint(0, 256, (2000,), dtype=torch.long),
            },
            {
                "vision": torch.randint(0, 256, (768,), dtype=torch.long),
                "text": torch.randint(0, 256, (256,), dtype=torch.long),
                "audio": torch.randint(0, 256, (2000,), dtype=torch.long),
            },
        ]

        result = collator(batch)

        assert "vision_bytes" in result
        assert "text_bytes" in result
        assert "audio_bytes" in result
        assert "vision_mask" in result
        assert "text_mask" in result
        assert "audio_mask" in result


class TestGetCollator:
    """Tests for get_collator factory."""

    def test_paired_mode(self):
        """Test paired mode."""
        collator = get_collator(mode="paired")
        assert isinstance(collator, PairedByteCollator)

    def test_multi_mode(self):
        """Test multi mode."""
        collator = get_collator(mode="multi")
        assert isinstance(collator, MultiModalByteCollator)

    def test_basic_mode(self):
        """Test basic mode."""
        collator = get_collator(mode="basic")
        assert isinstance(collator, ByteCollator)


class TestByteTransforms:
    """Tests for byte transforms."""

    def test_byte_noise(self):
        """Test ByteNoise transform."""
        transform = ByteNoise(noise_prob=0.5, noise_strength=10)

        x = torch.randint(0, 256, (100,), dtype=torch.long)
        out = transform(x.clone())

        # Should still be in valid range
        assert out.min() >= 0
        assert out.max() <= 255

    def test_byte_noise_zero_prob(self):
        """Test ByteNoise with zero probability."""
        transform = ByteNoise(noise_prob=0.0, noise_strength=10)

        x = torch.randint(0, 256, (100,), dtype=torch.long)
        out = transform(x.clone())

        # Should be unchanged
        assert torch.equal(out, x)

    def test_byte_mask(self):
        """Test ByteMask transform."""
        transform = ByteMask(mask_prob=1.0, mask_value=0)

        x = torch.randint(1, 256, (100,), dtype=torch.long)
        masked, mask = transform(x.clone())

        # All values should be masked
        assert (masked == 0).all()

    def test_byte_mask_zero_prob(self):
        """Test ByteMask with zero probability."""
        transform = ByteMask(mask_prob=0.0, mask_value=0)

        x = torch.randint(1, 256, (100,), dtype=torch.long)
        masked, mask = transform(x.clone())

        # Should be unchanged
        assert torch.equal(masked, x)

    def test_byte_shift(self):
        """Test ByteShift transform."""
        transform = ByteShift(max_shift=10)

        x = torch.randint(0, 256, (100,), dtype=torch.long)
        out = transform(x.clone())

        assert out.shape == x.shape
        assert out.min() >= 0
        assert out.max() <= 255

    def test_byte_crop(self):
        """Test ByteCrop transform."""
        transform = ByteCrop(crop_size=50, pad_value=0)

        x = torch.randint(0, 256, (100,), dtype=torch.long)
        out = transform(x.clone())

        # Output should be exactly crop_size
        assert out.shape[0] == 50

    def test_image_byte_flip_horizontal(self):
        """Test ImageByteFlip horizontal flip."""
        transform = ImageByteFlip(height=4, width=4, channels=3, p=1.0)

        # Create a simple pattern
        x = torch.arange(4 * 4 * 3, dtype=torch.long)
        out = transform(x.clone())

        assert out.shape == x.shape
        # Check flip happened (values should be different)
        assert not torch.equal(out, x)

    def test_image_byte_flip_no_flip(self):
        """Test ImageByteFlip with zero probability."""
        transform = ImageByteFlip(height=4, width=4, channels=3, p=0.0)

        x = torch.arange(4 * 4 * 3, dtype=torch.long)
        out = transform(x.clone())

        assert out.shape == x.shape
        # No flip should happen
        assert torch.equal(out, x)

    def test_compose(self):
        """Test ComposeByteTransforms."""
        transforms = ComposeByteTransforms([
            ByteNoise(noise_prob=0.1),
            ByteShift(max_shift=5),
        ])

        x = torch.randint(0, 256, (100,), dtype=torch.long)
        out = transforms(x.clone())

        assert out.shape == x.shape
        assert out.min() >= 0
        assert out.max() <= 255


class TestTransformFactories:
    """Tests for transform factory functions."""

    def test_get_byte_vision_transforms(self):
        """Test vision transforms factory."""
        transforms = get_byte_vision_transforms(
            height=32,
            width=32,
            training=True,
        )

        x = torch.randint(0, 256, (32 * 32 * 3,), dtype=torch.long)
        out = transforms(x.clone())

        assert out.shape == x.shape
        assert out.min() >= 0
        assert out.max() <= 255

    def test_get_byte_text_transforms(self):
        """Test text transforms factory."""
        transforms = get_byte_text_transforms(training=True)

        x = torch.randint(0, 256, (256,), dtype=torch.long)
        out = transforms(x.clone())

        assert out.min() >= 0
        assert out.max() <= 255

    def test_get_byte_audio_transforms(self):
        """Test audio transforms factory."""
        transforms = get_byte_audio_transforms(training=True)

        x = torch.randint(0, 256, (2000,), dtype=torch.long)
        out = transforms(x.clone())

        assert out.min() >= 0
        assert out.max() <= 255


class TestDataLoader:
    """Tests for DataLoader integration."""

    def test_synthetic_with_dataloader(self):
        """Test SyntheticByteDataset with DataLoader."""
        dataset = SyntheticByteDataset(
            num_samples=100,
            modalities=["vision", "text"],
        )
        collator = get_collator(
            mode="paired",
            source_modality="vision",
            target_modality="text",
        )
        loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collator,
        )

        batch = next(iter(loader))

        assert batch["source_bytes"].shape[0] == 4
        assert batch["target_bytes"].shape[0] == 4

    def test_multimodal_with_dataloader(self):
        """Test with MultiModalByteCollator."""
        dataset = SyntheticByteDataset(
            num_samples=100,
            modalities=["vision", "text", "audio"],
        )
        collator = get_collator(mode="multi")
        loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collator,
        )

        batch = next(iter(loader))

        assert "vision_bytes" in batch
        assert "text_bytes" in batch
        assert "audio_bytes" in batch

    def test_dataloader_iteration(self):
        """Test full iteration through DataLoader."""
        dataset = SyntheticByteDataset(
            num_samples=20,
            modalities=["vision", "text"],
        )
        collator = get_collator(mode="paired")
        loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collator,
        )

        batches = list(loader)
        assert len(batches) == 5  # 20 / 4

    def test_drop_last(self):
        """Test drop_last behavior."""
        dataset = SyntheticByteDataset(
            num_samples=22,
            modalities=["vision", "text"],
        )
        collator = get_collator(mode="paired")
        loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collator,
            drop_last=True,
        )

        batches = list(loader)
        assert len(batches) == 5  # 22 / 4 = 5, remainder dropped


class TestByteDataset:
    """Tests for ByteDataset."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary data directory with test files."""
        # Create images directory
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        return tmp_path

    def test_init_empty_dir(self, tmp_path):
        """Test initialization with empty directory."""
        dataset = ByteDataset(data_dir=tmp_path, modalities=["vision"])
        assert len(dataset) == 0

    def test_init_with_manifest(self, tmp_path):
        """Test initialization with manifest file."""
        manifest = tmp_path / "manifest.txt"
        manifest.write_text("image_path:test1.jpg\ttext:hello\n"
                           "image_path:test2.jpg\ttext:world\n")

        dataset = ByteDataset(data_dir=tmp_path, modalities=["vision", "text"])
        assert len(dataset) == 2

    def test_load_samples_auto_discover(self, tmp_path):
        """Test auto-discovery of image files."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # Create fake image files (just touch them)
        (images_dir / "test1.jpg").touch()
        (images_dir / "test2.png").touch()

        dataset = ByteDataset(data_dir=tmp_path, modalities=["vision"])
        assert len(dataset) == 2

    def test_vision_seq_len(self, tmp_path):
        """Test vision sequence length calculation."""
        dataset = ByteDataset(
            data_dir=tmp_path,
            vision_size=(16, 16),
            modalities=["vision"],
        )
        assert dataset.vision_seq_len == 16 * 16 * 3

    def test_getitem_empty_sample(self, tmp_path):
        """Test getitem with no matching data."""
        manifest = tmp_path / "manifest.txt"
        manifest.write_text("foo:bar\n")

        dataset = ByteDataset(data_dir=tmp_path, modalities=["vision"])
        item = dataset[0]
        assert "masks" in item

    def test_getitem_with_text(self, tmp_path):
        """Test getitem with text data."""
        manifest = tmp_path / "manifest.txt"
        manifest.write_text("text:Hello World\n")

        dataset = ByteDataset(data_dir=tmp_path, modalities=["text"])
        item = dataset[0]
        assert "text" in item
        assert "masks" in item
        assert item["text"].dtype == torch.long

    def test_load_text_bytes(self, tmp_path):
        """Test _load_text_bytes method."""
        dataset = ByteDataset(
            data_dir=tmp_path,
            modalities=["text"],
            text_max_len=20,
        )

        text = "Hello!"
        bytes_tensor, mask = dataset._load_text_bytes(text)

        assert bytes_tensor.shape[0] == 20
        assert mask[:len(text.encode("utf-8"))].all()
        assert not mask[len(text.encode("utf-8")):].any()

    def test_load_text_bytes_truncation(self, tmp_path):
        """Test text truncation when exceeding max length."""
        dataset = ByteDataset(
            data_dir=tmp_path,
            modalities=["text"],
            text_max_len=5,
        )

        text = "Hello World!"
        bytes_tensor, mask = dataset._load_text_bytes(text)

        assert bytes_tensor.shape[0] == 5
        assert mask.sum() == 5

    def test_load_image_bytes_requires_pil(self, tmp_path):
        """Test image loading requires PIL."""
        import src.data.byte_dataset as byte_dataset_module
        original = byte_dataset_module.PIL_AVAILABLE
        try:
            byte_dataset_module.PIL_AVAILABLE = False
            dataset = ByteDataset(data_dir=tmp_path, modalities=["vision"])
            with pytest.raises(ImportError, match="PIL"):
                dataset._load_image_bytes("fake.jpg")
        finally:
            byte_dataset_module.PIL_AVAILABLE = original

    def test_load_audio_bytes_requires_torchaudio(self, tmp_path):
        """Test audio loading requires torchaudio."""
        import src.data.byte_dataset as byte_dataset_module
        original = byte_dataset_module.TORCHAUDIO_AVAILABLE
        try:
            byte_dataset_module.TORCHAUDIO_AVAILABLE = False
            dataset = ByteDataset(data_dir=tmp_path, modalities=["audio"])
            with pytest.raises(ImportError, match="torchaudio"):
                dataset._load_audio_bytes("fake.wav")
        finally:
            byte_dataset_module.TORCHAUDIO_AVAILABLE = original

    def test_load_image_bytes_with_pil(self, tmp_path):
        """Test image loading with PIL."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        # Create a test image
        img = Image.new("RGB", (32, 32), color=(255, 0, 0))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        dataset = ByteDataset(
            data_dir=tmp_path,
            vision_size=(32, 32),
            modalities=["vision"],
        )

        bytes_tensor, mask = dataset._load_image_bytes(str(img_path))

        assert bytes_tensor.shape[0] == 32 * 32 * 3
        assert bytes_tensor.dtype == torch.long
        assert mask.all()


class TestPairedByteDataset:
    """Tests for PairedByteDataset."""

    def test_init_empty_dir(self, tmp_path):
        """Test initialization with empty directory."""
        dataset = PairedByteDataset(data_dir=tmp_path)
        assert len(dataset) == 0

    def test_init_with_captions(self, tmp_path):
        """Test initialization with captions file."""
        (tmp_path / "images").mkdir()
        captions = tmp_path / "captions.txt"
        captions.write_text("image1.jpg\tA cat on a mat\n"
                           "image2.jpg\tA dog in fog\n")

        dataset = PairedByteDataset(data_dir=tmp_path)
        assert len(dataset) == 2

    def test_vision_seq_len(self, tmp_path):
        """Test vision sequence length calculation."""
        dataset = PairedByteDataset(
            data_dir=tmp_path,
            vision_size=(16, 16),
        )
        assert dataset.vision_seq_len == 16 * 16 * 3

    def test_getitem_fallback_to_random(self, tmp_path):
        """Test getitem falls back to random data when image missing."""
        (tmp_path / "images").mkdir()
        captions = tmp_path / "captions.txt"
        captions.write_text("nonexistent.jpg\tTest caption\n")

        dataset = PairedByteDataset(data_dir=tmp_path)
        item = dataset[0]

        assert "vision" in item
        assert "text" in item
        assert "masks" in item
        assert item["vision"].shape[0] == dataset.vision_seq_len

    def test_getitem_with_real_image(self, tmp_path):
        """Test getitem with real image file."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        (tmp_path / "images").mkdir()

        # Create a test image
        img = Image.new("RGB", (32, 32), color=(0, 255, 0))
        img.save(tmp_path / "images" / "test.png")

        captions = tmp_path / "captions.txt"
        captions.write_text("test.png\tA green square\n")

        dataset = PairedByteDataset(
            data_dir=tmp_path,
            vision_size=(32, 32),
        )
        item = dataset[0]

        assert "vision" in item
        assert item["vision"].shape[0] == 32 * 32 * 3
        assert item["masks"]["vision"].all()

    def test_text_encoding(self, tmp_path):
        """Test text is properly encoded to UTF-8 bytes."""
        (tmp_path / "images").mkdir()
        captions = tmp_path / "captions.txt"
        captions.write_text("img.jpg\tHello World\n")

        dataset = PairedByteDataset(
            data_dir=tmp_path,
            text_max_len=20,
        )
        item = dataset[0]

        assert item["text"].shape[0] == 20
        text_len = len("Hello World".encode("utf-8"))
        assert item["masks"]["text"][:text_len].all()

    def test_load_pairs_empty_lines(self, tmp_path):
        """Test load_pairs handles empty lines."""
        (tmp_path / "images").mkdir()
        captions = tmp_path / "captions.txt"
        captions.write_text("img1.jpg\tCaption 1\n\n\nimg2.jpg\tCaption 2\n")

        dataset = PairedByteDataset(data_dir=tmp_path)
        # Only lines with proper format should be loaded
        assert len(dataset) == 2

    def test_load_pairs_malformed_lines(self, tmp_path):
        """Test load_pairs handles malformed lines."""
        (tmp_path / "images").mkdir()
        captions = tmp_path / "captions.txt"
        captions.write_text("img1.jpg\tCaption 1\n"
                           "malformed_no_tab\n"
                           "img2.jpg\tCaption 2\n")

        dataset = PairedByteDataset(data_dir=tmp_path)
        # Only properly formatted lines should be loaded
        assert len(dataset) == 2
