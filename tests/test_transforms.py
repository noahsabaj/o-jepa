"""Tests for data transforms."""

import pytest
import torch
import numpy as np

from src.data.transforms import (
    # Core transforms
    Compose,
    ToTensor,
    Normalize,
    Resize,
    RandomCrop,
    CenterCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ColorJitter,
    GaussianBlur,
    RandomResizedCrop,
    # Audio transforms
    AudioNormalize,
    AudioRandomCrop,
    AudioTimeStretch,
    AudioAddNoise,
    # Video transforms
    VideoRandomCrop,
    VideoTemporalCrop,
    VideoRandomHorizontalFlip,
    # Factories
    get_vision_transforms,
    get_audio_transforms,
    get_video_transforms,
    get_depth_transforms,
    get_thermal_transforms,
    get_all_byte_transforms,
    # Byte transforms
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


class TestCompose:
    """Tests for Compose transform."""

    def test_compose_single(self):
        """Test compose with single transform."""
        transform = Compose([lambda x: x * 2])
        x = torch.ones(3, 4, 4)
        result = transform(x)
        assert torch.allclose(result, x * 2)

    def test_compose_multiple(self):
        """Test compose with multiple transforms."""
        transform = Compose([lambda x: x + 1, lambda x: x * 2])
        x = torch.ones(3, 4, 4)
        result = transform(x)
        assert torch.allclose(result, (x + 1) * 2)

    def test_compose_empty(self):
        """Test compose with no transforms."""
        transform = Compose([])
        x = torch.ones(3, 4, 4)
        result = transform(x)
        assert torch.equal(result, x)


class TestToTensor:
    """Tests for ToTensor transform."""

    def test_from_numpy(self):
        """Test conversion from numpy array."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        transform = ToTensor()
        result = transform(arr)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 2)

    def test_tensor_passthrough(self):
        """Test tensor passthrough."""
        x = torch.tensor([[1, 2], [3, 4]])
        transform = ToTensor()
        result = transform(x)
        assert torch.equal(result, x)


class TestNormalize:
    """Tests for Normalize transform."""

    def test_normalize_basic(self):
        """Test basic normalization."""
        transform = Normalize(mean=[0.5], std=[0.5])
        x = torch.ones(1, 4, 4) * 0.5
        result = transform(x)
        assert torch.allclose(result, torch.zeros(1, 4, 4), atol=1e-6)

    def test_normalize_rgb(self):
        """Test RGB normalization."""
        transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        x = torch.rand(3, 32, 32)
        result = transform(x)
        assert result.shape == x.shape


class TestResize:
    """Tests for Resize transform."""

    def test_resize_3d(self):
        """Test resize with 3D tensor."""
        transform = Resize((16, 16))
        x = torch.rand(3, 32, 32)
        result = transform(x)
        assert result.shape == (3, 16, 16)

    def test_resize_4d(self):
        """Test resize with 4D tensor."""
        transform = Resize((16, 16))
        x = torch.rand(2, 3, 32, 32)
        result = transform(x)
        assert result.shape == (2, 3, 16, 16)


class TestRandomCrop:
    """Tests for RandomCrop transform."""

    def test_random_crop_basic(self):
        """Test basic random crop."""
        transform = RandomCrop((16, 16))
        x = torch.rand(3, 32, 32)
        result = transform(x)
        assert result.shape == (3, 16, 16)

    def test_random_crop_with_padding(self):
        """Test random crop with padding."""
        transform = RandomCrop((16, 16), padding=4)
        x = torch.rand(3, 32, 32)
        result = transform(x)
        assert result.shape == (3, 16, 16)

    def test_random_crop_same_size(self):
        """Test random crop when size equals input."""
        transform = RandomCrop((32, 32))
        x = torch.rand(3, 32, 32)
        result = transform(x)
        assert result.shape == (3, 32, 32)


class TestCenterCrop:
    """Tests for CenterCrop transform."""

    def test_center_crop_basic(self):
        """Test basic center crop."""
        transform = CenterCrop((16, 16))
        x = torch.rand(3, 32, 32)
        result = transform(x)
        assert result.shape == (3, 16, 16)

    def test_center_crop_rectangular(self):
        """Test center crop with rectangular output."""
        transform = CenterCrop((16, 24))
        x = torch.rand(3, 32, 32)
        result = transform(x)
        assert result.shape == (3, 16, 24)


class TestRandomHorizontalFlip:
    """Tests for RandomHorizontalFlip transform."""

    def test_flip_always(self):
        """Test flip with p=1.0."""
        transform = RandomHorizontalFlip(p=1.0)
        x = torch.arange(12).view(1, 3, 4).float()
        result = transform(x)
        assert result.shape == x.shape
        # Check that it's actually flipped
        assert torch.equal(result, x.flip(-1))

    def test_flip_never(self):
        """Test flip with p=0.0."""
        transform = RandomHorizontalFlip(p=0.0)
        x = torch.arange(12).view(1, 3, 4).float()
        result = transform(x)
        assert torch.equal(result, x)


class TestRandomVerticalFlip:
    """Tests for RandomVerticalFlip transform."""

    def test_flip_always(self):
        """Test flip with p=1.0."""
        transform = RandomVerticalFlip(p=1.0)
        x = torch.arange(12).view(1, 3, 4).float()
        result = transform(x)
        assert result.shape == x.shape
        assert torch.equal(result, x.flip(-2))

    def test_flip_never(self):
        """Test flip with p=0.0."""
        transform = RandomVerticalFlip(p=0.0)
        x = torch.arange(12).view(1, 3, 4).float()
        result = transform(x)
        assert torch.equal(result, x)


class TestColorJitter:
    """Tests for ColorJitter transform."""

    def test_color_jitter_basic(self):
        """Test basic color jitter."""
        transform = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        x = torch.rand(3, 32, 32)
        result = transform(x)
        assert result.shape == x.shape
        # Check values are clamped
        assert result.min() >= 0
        assert result.max() <= 1

    def test_color_jitter_no_changes(self):
        """Test color jitter with all zeros."""
        transform = ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
        x = torch.rand(3, 32, 32)
        result = transform(x)
        assert result.shape == x.shape

    def test_color_jitter_grayscale(self):
        """Test color jitter with single channel (no saturation)."""
        transform = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        x = torch.rand(1, 32, 32)  # Grayscale
        result = transform(x)
        assert result.shape == x.shape


class TestGaussianBlur:
    """Tests for GaussianBlur transform."""

    def test_gaussian_blur_basic(self):
        """Test basic Gaussian blur."""
        transform = GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        x = torch.rand(3, 32, 32)
        result = transform(x)
        assert result.shape == x.shape

    def test_gaussian_blur_single_channel(self):
        """Test Gaussian blur with single channel."""
        transform = GaussianBlur(kernel_size=3, sigma=(0.5, 1.0))
        x = torch.rand(1, 32, 32)
        result = transform(x)
        assert result.shape == x.shape


class TestRandomResizedCrop:
    """Tests for RandomResizedCrop transform."""

    def test_random_resized_crop_basic(self):
        """Test basic random resized crop."""
        transform = RandomResizedCrop((16, 16), scale=(0.5, 1.0))
        x = torch.rand(3, 32, 32)
        result = transform(x)
        assert result.shape == (3, 16, 16)

    def test_random_resized_crop_fallback(self):
        """Test fallback to center crop."""
        # Use extreme ratio that should trigger fallback
        transform = RandomResizedCrop((16, 16), scale=(0.01, 0.02), ratio=(0.01, 0.02))
        x = torch.rand(3, 32, 32)
        result = transform(x)
        assert result.shape == (3, 16, 16)


class TestAudioNormalize:
    """Tests for AudioNormalize transform."""

    def test_audio_normalize_basic(self):
        """Test basic audio normalization."""
        transform = AudioNormalize()
        x = torch.randn(1, 16000) * 0.5
        result = transform(x)
        # Max should be 1.0 after normalization
        assert result.abs().max() <= 1.0 + 1e-6

    def test_audio_normalize_zero(self):
        """Test normalization with zero signal."""
        transform = AudioNormalize()
        x = torch.zeros(1, 16000)
        result = transform(x)
        assert torch.equal(result, x)


class TestAudioRandomCrop:
    """Tests for AudioRandomCrop transform."""

    def test_audio_crop_longer(self):
        """Test cropping longer audio."""
        transform = AudioRandomCrop(length=8000)
        x = torch.rand(1, 16000)
        result = transform(x)
        assert result.shape == (1, 8000)

    def test_audio_crop_shorter(self):
        """Test cropping shorter audio (padding)."""
        transform = AudioRandomCrop(length=16000)
        x = torch.rand(1, 8000)
        result = transform(x)
        assert result.shape == (1, 16000)


class TestAudioTimeStretch:
    """Tests for AudioTimeStretch transform."""

    def test_audio_time_stretch_1d(self):
        """Test time stretch with 1D input."""
        transform = AudioTimeStretch(rate_range=(0.9, 1.1))
        x = torch.rand(16000)
        result = transform(x)
        # Length should change based on rate
        assert result.ndim == 1

    def test_audio_time_stretch_2d(self):
        """Test time stretch with 2D input."""
        transform = AudioTimeStretch(rate_range=(0.9, 1.1))
        x = torch.rand(1, 16000)
        result = transform(x)
        assert result.ndim == 2

    def test_audio_time_stretch_3d(self):
        """Test time stretch with 3D input (passthrough)."""
        transform = AudioTimeStretch(rate_range=(0.9, 1.1))
        x = torch.rand(1, 1, 16000)
        result = transform(x)
        assert result.shape == x.shape


class TestAudioAddNoise:
    """Tests for AudioAddNoise transform."""

    def test_audio_add_noise_basic(self):
        """Test basic noise addition."""
        transform = AudioAddNoise(snr_range=(10, 30))
        x = torch.randn(1, 16000)
        result = transform(x)
        assert result.shape == x.shape
        # Should be different from original
        assert not torch.equal(result, x)


class TestVideoRandomCrop:
    """Tests for VideoRandomCrop transform."""

    def test_video_random_crop_basic(self):
        """Test basic video random crop."""
        transform = VideoRandomCrop((16, 16))
        x = torch.rand(3, 8, 32, 32)  # [C, T, H, W]
        result = transform(x)
        assert result.shape == (3, 8, 16, 16)

    def test_video_random_crop_same_size(self):
        """Test video crop when size matches."""
        transform = VideoRandomCrop((32, 32))
        x = torch.rand(3, 8, 32, 32)
        result = transform(x)
        assert result.shape == (3, 8, 32, 32)

    def test_video_random_crop_tchw(self):
        """Test video crop with [T, C, H, W] format."""
        transform = VideoRandomCrop((16, 16))
        x = torch.rand(8, 3, 32, 32)  # [T, C, H, W]
        result = transform(x)
        assert result.shape[-2:] == (16, 16)


class TestVideoTemporalCrop:
    """Tests for VideoTemporalCrop transform."""

    def test_temporal_crop_basic(self):
        """Test basic temporal crop."""
        transform = VideoTemporalCrop(num_frames=4)
        x = torch.rand(3, 8, 32, 32)
        result = transform(x)
        assert result.shape == (3, 4, 32, 32)

    def test_temporal_crop_short(self):
        """Test temporal crop with short video (repeats frames)."""
        transform = VideoTemporalCrop(num_frames=8)
        x = torch.rand(3, 4, 32, 32)
        result = transform(x)
        assert result.shape == (3, 8, 32, 32)


class TestVideoRandomHorizontalFlip:
    """Tests for VideoRandomHorizontalFlip transform."""

    def test_video_flip_always(self):
        """Test video flip with p=1.0."""
        transform = VideoRandomHorizontalFlip(p=1.0)
        x = torch.arange(24).view(1, 2, 3, 4).float()
        result = transform(x)
        assert result.shape == x.shape
        assert torch.equal(result, x.flip(-1))

    def test_video_flip_never(self):
        """Test video flip with p=0.0."""
        transform = VideoRandomHorizontalFlip(p=0.0)
        x = torch.arange(24).view(1, 2, 3, 4).float()
        result = transform(x)
        assert torch.equal(result, x)


class TestTransformFactories:
    """Tests for transform factory functions."""

    def test_get_vision_transforms_training(self):
        """Test vision transforms for training."""
        transform = get_vision_transforms(image_size=32, is_training=True)
        x = torch.rand(3, 64, 64)
        result = transform(x)
        assert result.shape == (3, 32, 32)

    def test_get_vision_transforms_eval(self):
        """Test vision transforms for evaluation."""
        transform = get_vision_transforms(image_size=32, is_training=False)
        x = torch.rand(3, 64, 64)
        result = transform(x)
        assert result.shape == (3, 32, 32)

    def test_get_audio_transforms_training(self):
        """Test audio transforms for training."""
        transform = get_audio_transforms(sample_length=8000, is_training=True)
        x = torch.rand(1, 16000)
        result = transform(x)
        assert result.shape[-1] == 8000

    def test_get_audio_transforms_eval(self):
        """Test audio transforms for evaluation."""
        transform = get_audio_transforms(sample_length=8000, is_training=False)
        x = torch.rand(1, 16000)
        result = transform(x)
        assert result.shape[-1] == 8000

    def test_get_video_transforms_training(self):
        """Test video transforms for training."""
        transform = get_video_transforms(image_size=32, num_frames=4, is_training=True)
        x = torch.rand(3, 8, 64, 64)
        result = transform(x)
        assert result.shape == (3, 4, 32, 32)

    def test_get_video_transforms_eval(self):
        """Test video transforms for evaluation."""
        transform = get_video_transforms(image_size=32, num_frames=4, is_training=False)
        x = torch.rand(3, 8, 64, 64)
        result = transform(x)
        assert result.shape[1] == 4

    def test_get_depth_transforms_training(self):
        """Test depth transforms for training."""
        transform = get_depth_transforms(image_size=32, is_training=True)
        x = torch.rand(1, 64, 64)
        result = transform(x)
        assert result.shape == (1, 32, 32)

    def test_get_depth_transforms_eval(self):
        """Test depth transforms for evaluation."""
        transform = get_depth_transforms(image_size=32, is_training=False)
        x = torch.rand(1, 64, 64)
        result = transform(x)
        assert result.shape == (1, 32, 32)

    def test_get_thermal_transforms(self):
        """Test thermal transforms (same as depth)."""
        transform = get_thermal_transforms(image_size=32, is_training=True)
        x = torch.rand(1, 64, 64)
        result = transform(x)
        assert result.shape == (1, 32, 32)

    def test_get_all_byte_transforms_training(self):
        """Test get_all_byte_transforms for training."""
        transforms = get_all_byte_transforms(height=32, width=32, training=True)
        assert "vision" in transforms
        assert "text" in transforms
        assert "audio" in transforms
        assert transforms["vision"] is not None

    def test_get_all_byte_transforms_eval(self):
        """Test get_all_byte_transforms for evaluation."""
        transforms = get_all_byte_transforms(height=32, width=32, training=False)
        assert transforms["vision"] is None
        assert transforms["text"] is None
        assert transforms["audio"] is None


class TestByteTransformsExtended:
    """Extended tests for byte transforms."""

    def test_byte_noise_flip_type(self):
        """Test ByteNoise with flip type."""
        transform = ByteNoise(noise_prob=1.0, noise_type="flip")
        x = torch.randint(0, 256, (100,), dtype=torch.long)
        result = transform(x)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_byte_crop_padding(self):
        """Test ByteCrop with padding needed."""
        transform = ByteCrop(crop_size=100, pad_value=255)
        x = torch.randint(0, 256, (50,), dtype=torch.long)
        result = transform(x)
        assert result.shape[0] == 100
        assert (result[50:] == 255).all()

    def test_compose_byte_transforms_tuple_handling(self):
        """Test ComposeByteTransforms handles tuple returns."""
        transforms = ComposeByteTransforms([
            ByteMask(mask_prob=0.5, mask_value=0),  # Returns tuple
            ByteNoise(noise_prob=0.1),  # Returns tensor
        ])
        x = torch.randint(0, 256, (100,), dtype=torch.long)
        result = transforms(x)
        assert result.shape == x.shape

    def test_get_byte_vision_transforms_eval(self):
        """Test byte vision transforms returns None for eval."""
        transform = get_byte_vision_transforms(height=32, width=32, training=False)
        assert transform is None

    def test_get_byte_text_transforms_eval(self):
        """Test byte text transforms returns None for eval."""
        transform = get_byte_text_transforms(training=False)
        assert transform is None

    def test_get_byte_audio_transforms_eval(self):
        """Test byte audio transforms returns None for eval."""
        transform = get_byte_audio_transforms(training=False)
        assert transform is None
