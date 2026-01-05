"""
O-JEPA Data Transforms

Provides augmentation pipelines for each modality.
"""

from typing import Callable, Optional, Tuple, List
import torch
import torch.nn.functional as F


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class ToTensor:
    """Convert numpy array or PIL Image to tensor."""

    def __call__(self, x):
        if hasattr(x, 'numpy'):
            return torch.from_numpy(x.numpy())
        elif isinstance(x, torch.Tensor):
            return x
        else:
            # Assume numpy array
            return torch.from_numpy(x)


class Normalize:
    """Normalize tensor with mean and std."""

    def __init__(self, mean: List[float], std: List[float]):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype)
        return (x - mean) / std


class Resize:
    """Resize image to target size."""

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
            x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=False)
            return x.squeeze(0)
        return F.interpolate(x, size=self.size, mode='bilinear', align_corners=False)


class RandomCrop:
    """Random crop with optional padding."""

    def __init__(self, size: Tuple[int, int], padding: int = 0):
        self.size = size
        self.padding = padding

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = F.pad(x, [self.padding] * 4, mode='reflect')

        _, h, w = x.shape
        new_h, new_w = self.size

        if h == new_h and w == new_w:
            return x

        top = torch.randint(0, h - new_h + 1, (1,)).item()
        left = torch.randint(0, w - new_w + 1, (1,)).item()

        return x[:, top:top + new_h, left:left + new_w]


class CenterCrop:
    """Center crop to target size."""

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w = x.shape
        new_h, new_w = self.size

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        return x[:, top:top + new_h, left:left + new_w]


class RandomHorizontalFlip:
    """Random horizontal flip."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return x.flip(-1)
        return x


class RandomVerticalFlip:
    """Random vertical flip."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return x.flip(-2)
        return x


class ColorJitter:
    """Random color jittering."""

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Brightness
        if self.brightness > 0:
            factor = 1 + (torch.rand(1).item() * 2 - 1) * self.brightness
            x = x * factor

        # Contrast
        if self.contrast > 0:
            factor = 1 + (torch.rand(1).item() * 2 - 1) * self.contrast
            mean = x.mean(dim=(-2, -1), keepdim=True)
            x = mean + factor * (x - mean)

        # Saturation (simplified - works on RGB)
        if self.saturation > 0 and x.shape[0] == 3:
            factor = 1 + (torch.rand(1).item() * 2 - 1) * self.saturation
            gray = x.mean(dim=0, keepdim=True)
            x = gray + factor * (x - gray)

        return x.clamp(0, 1)


class GaussianBlur:
    """Apply Gaussian blur."""

    def __init__(self, kernel_size: int = 5, sigma: Tuple[float, float] = (0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sigma = torch.empty(1).uniform_(self.sigma[0], self.sigma[1]).item()

        # Create Gaussian kernel
        kernel_size = self.kernel_size
        coords = torch.arange(kernel_size).float() - kernel_size // 2
        kernel_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_2d = kernel_1d.view(-1, 1) @ kernel_1d.view(1, -1)
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        kernel_2d = kernel_2d.repeat(x.shape[0], 1, 1, 1)

        # Apply blur
        padding = kernel_size // 2
        x = x.unsqueeze(0)
        x = F.pad(x, [padding] * 4, mode='reflect')
        x = F.conv2d(x, kernel_2d.to(x.device, x.dtype), groups=x.shape[1])
        return x.squeeze(0)


class RandomResizedCrop:
    """Random resized crop for vision data."""

    def __init__(
        self,
        size: Tuple[int, int],
        scale: Tuple[float, float] = (0.8, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.333),
    ):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.shape
        area = h * w

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.empty(1).uniform_(self.ratio[0], self.ratio[1]).item()

            new_w = int(round((target_area * aspect_ratio) ** 0.5))
            new_h = int(round((target_area / aspect_ratio) ** 0.5))

            if 0 < new_w <= w and 0 < new_h <= h:
                i = torch.randint(0, h - new_h + 1, (1,)).item()
                j = torch.randint(0, w - new_w + 1, (1,)).item()

                x = x[:, i:i + new_h, j:j + new_w]
                x = x.unsqueeze(0)
                x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=False)
                return x.squeeze(0)

        # Fallback to center crop
        crop = CenterCrop((min(h, w), min(h, w)))
        resize = Resize(self.size)
        return resize(crop(x))


# Audio-specific transforms
class AudioNormalize:
    """Normalize audio waveform."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        max_val = x.abs().max()
        if max_val > 0:
            x = x / max_val
        return x


class AudioRandomCrop:
    """Random crop audio to fixed length."""

    def __init__(self, length: int):
        self.length = length

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] >= self.length:
            start = torch.randint(0, x.shape[-1] - self.length + 1, (1,)).item()
            return x[..., start:start + self.length]
        else:
            # Pad if too short
            pad_amount = self.length - x.shape[-1]
            return F.pad(x, (0, pad_amount))


class AudioTimeStretch:
    """Time stretch augmentation (simplified)."""

    def __init__(self, rate_range: Tuple[float, float] = (0.9, 1.1)):
        self.rate_range = rate_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        rate = torch.empty(1).uniform_(self.rate_range[0], self.rate_range[1]).item()
        new_length = int(x.shape[-1] / rate)

        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
            x = F.interpolate(x, size=new_length, mode='linear', align_corners=False)
            return x.squeeze(0).squeeze(0)
        elif x.ndim == 2:
            x = x.unsqueeze(0)
            x = F.interpolate(x, size=new_length, mode='linear', align_corners=False)
            return x.squeeze(0)
        return x


class AudioAddNoise:
    """Add random noise to audio."""

    def __init__(self, snr_range: Tuple[float, float] = (10, 30)):
        self.snr_range = snr_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        snr_db = torch.empty(1).uniform_(self.snr_range[0], self.snr_range[1]).item()

        signal_power = x.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(x) * noise_power.sqrt()

        return x + noise


# Video-specific transforms
class VideoRandomCrop:
    """Random spatial crop for video."""

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [C, T, H, W] or [T, C, H, W]
        if x.shape[1] == 3:  # [T, C, H, W]
            x = x.permute(1, 0, 2, 3)  # -> [C, T, H, W]

        _, t, h, w = x.shape
        new_h, new_w = self.size

        if h == new_h and w == new_w:
            return x

        top = torch.randint(0, h - new_h + 1, (1,)).item()
        left = torch.randint(0, w - new_w + 1, (1,)).item()

        return x[:, :, top:top + new_h, left:left + new_w]


class VideoTemporalCrop:
    """Random temporal crop for video."""

    def __init__(self, num_frames: int):
        self.num_frames = num_frames

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [C, T, H, W]
        t = x.shape[1]

        if t >= self.num_frames:
            start = torch.randint(0, t - self.num_frames + 1, (1,)).item()
            return x[:, start:start + self.num_frames]
        else:
            # Repeat frames if too short
            indices = torch.linspace(0, t - 1, self.num_frames).long()
            return x[:, indices]


class VideoRandomHorizontalFlip:
    """Random horizontal flip for video."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [C, T, H, W]
        if torch.rand(1).item() < self.p:
            return x.flip(-1)
        return x


# Transform factories
def get_vision_transforms(
    image_size: int = 224,
    is_training: bool = True,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> Compose:
    """Get vision transforms for training or evaluation."""

    if is_training:
        transforms = [
            RandomResizedCrop((image_size, image_size), scale=(0.5, 1.0)),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            Normalize(list(mean), list(std)),
        ]
    else:
        transforms = [
            Resize((256, 256)),
            CenterCrop((image_size, image_size)),
            Normalize(list(mean), list(std)),
        ]

    return Compose(transforms)


def get_audio_transforms(
    sample_length: int = 32000,
    is_training: bool = True,
) -> Compose:
    """Get audio transforms for training or evaluation."""

    if is_training:
        transforms = [
            AudioNormalize(),
            AudioTimeStretch(rate_range=(0.9, 1.1)),
            AudioRandomCrop(sample_length),
            AudioAddNoise(snr_range=(15, 40)),
        ]
    else:
        transforms = [
            AudioNormalize(),
            AudioRandomCrop(sample_length),
        ]

    return Compose(transforms)


def get_video_transforms(
    image_size: int = 224,
    num_frames: int = 8,
    is_training: bool = True,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> Compose:
    """Get video transforms for training or evaluation."""

    if is_training:
        transforms = [
            VideoTemporalCrop(num_frames),
            VideoRandomCrop((image_size, image_size)),
            VideoRandomHorizontalFlip(p=0.5),
        ]
    else:
        transforms = [
            VideoTemporalCrop(num_frames),
        ]

    return Compose(transforms)


def get_depth_transforms(
    image_size: int = 224,
    is_training: bool = True,
) -> Compose:
    """Get depth transforms for training or evaluation."""

    if is_training:
        transforms = [
            RandomResizedCrop((image_size, image_size), scale=(0.5, 1.0)),
            RandomHorizontalFlip(p=0.5),
        ]
    else:
        transforms = [
            Resize((256, 256)),
            CenterCrop((image_size, image_size)),
        ]

    return Compose(transforms)


def get_thermal_transforms(
    image_size: int = 224,
    is_training: bool = True,
) -> Compose:
    """Get thermal transforms for training or evaluation."""
    # Same as depth
    return get_depth_transforms(image_size, is_training)


# =============================================================================
# BYTE-LEVEL TRANSFORMS
# =============================================================================

class ByteNoise:
    """
    Add noise to byte sequences.

    Randomly flips bits or adds/subtracts small values.

    Args:
        noise_prob: Probability of modifying each byte
        noise_type: "flip" for bit flipping, "additive" for value changes
        noise_strength: For additive noise, max value to add/subtract
    """

    def __init__(
        self,
        noise_prob: float = 0.05,
        noise_type: str = "additive",
        noise_strength: int = 10,
    ):
        self.noise_prob = noise_prob
        self.noise_type = noise_type
        self.noise_strength = noise_strength

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply noise to byte tensor.

        Args:
            x: Byte tensor [seq_len] with values in [0, 255]

        Returns:
            Noised tensor [seq_len]
        """
        mask = torch.rand_like(x.float()) < self.noise_prob

        if self.noise_type == "flip":
            # Flip random bits
            flip_bits = torch.randint(0, 8, x.shape, device=x.device)
            flip_mask = (1 << flip_bits)
            noised = x ^ (mask.long() * flip_mask)
        else:
            # Additive noise
            noise = torch.randint(
                -self.noise_strength, self.noise_strength + 1,
                x.shape, device=x.device, dtype=x.dtype
            )
            noised = (x + mask.long() * noise).clamp(0, 255)

        return noised


class ByteMask:
    """
    Mask out portions of byte sequences.

    Replaces masked positions with a special byte value.

    Args:
        mask_prob: Probability of masking each byte
        mask_value: Value to use for masked positions (default 0)
    """

    def __init__(
        self,
        mask_prob: float = 0.15,
        mask_value: int = 0,
    ):
        self.mask_prob = mask_prob
        self.mask_value = mask_value

    def __call__(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mask byte tensor.

        Args:
            x: Byte tensor [seq_len]

        Returns:
            Tuple of (masked tensor, mask indicating modified positions)
        """
        mask = torch.rand_like(x.float()) < self.mask_prob
        masked = x.clone()
        masked[mask] = self.mask_value
        return masked, mask


class ByteShift:
    """
    Circular shift byte sequence.

    Useful for data augmentation.

    Args:
        max_shift: Maximum shift amount (absolute value)
    """

    def __init__(self, max_shift: int = 100):
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply circular shift to byte tensor.

        Args:
            x: Byte tensor [seq_len]

        Returns:
            Shifted tensor [seq_len]
        """
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        return torch.roll(x, shifts=shift)


class ByteCrop:
    """
    Random crop of byte sequence.

    Args:
        crop_size: Size of crop
        pad_value: Value for padding if sequence too short
    """

    def __init__(
        self,
        crop_size: int,
        pad_value: int = 0,
    ):
        self.crop_size = crop_size
        self.pad_value = pad_value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Crop byte tensor.

        Args:
            x: Byte tensor [seq_len]

        Returns:
            Cropped tensor [crop_size]
        """
        seq_len = len(x)

        if seq_len <= self.crop_size:
            # Pad if too short
            padded = torch.full((self.crop_size,), self.pad_value, dtype=x.dtype)
            padded[:seq_len] = x
            return padded
        else:
            # Random crop
            start = torch.randint(0, seq_len - self.crop_size, (1,)).item()
            return x[start:start + self.crop_size]


class ImageByteFlip:
    """
    Horizontal flip for image bytes.

    Operates on flattened RGB bytes assuming row-major order.

    Args:
        height: Image height
        width: Image width
        channels: Number of channels (3 for RGB)
        p: Probability of flipping
    """

    def __init__(
        self,
        height: int = 32,
        width: int = 32,
        channels: int = 3,
        p: float = 0.5,
    ):
        self.height = height
        self.width = width
        self.channels = channels
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply horizontal flip to image bytes.

        Args:
            x: Flattened image bytes [H*W*C]

        Returns:
            Flipped (or not) bytes [H*W*C]
        """
        if torch.rand(1).item() > self.p:
            return x

        # Reshape to [H, W, C]
        img = x.view(self.height, self.width, self.channels)
        # Flip horizontally
        flipped = img.flip(1)
        # Flatten back
        return flipped.reshape(-1)


class ComposeByteTransforms:
    """
    Compose multiple byte transforms.

    Args:
        transforms: List of transform objects
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            result = t(x)
            # Handle transforms that return tuples
            if isinstance(result, tuple):
                x = result[0]
            else:
                x = result
        return x


def get_byte_vision_transforms(
    height: int = 32,
    width: int = 32,
    training: bool = True,
) -> Optional[ComposeByteTransforms]:
    """
    Get standard transforms for vision bytes.

    Args:
        height: Image height
        width: Image width
        training: Whether in training mode

    Returns:
        Compose of transforms (or None for validation)
    """
    if not training:
        return None

    return ComposeByteTransforms([
        ImageByteFlip(height, width, 3, p=0.5),
        ByteNoise(noise_prob=0.02, noise_strength=5),
    ])


def get_byte_text_transforms(training: bool = True) -> Optional[ComposeByteTransforms]:
    """
    Get standard transforms for text bytes.

    Args:
        training: Whether in training mode

    Returns:
        Compose of transforms (or None for validation)
    """
    if not training:
        return None

    return ComposeByteTransforms([
        ByteNoise(noise_prob=0.01, noise_strength=0, noise_type="flip"),
    ])


def get_byte_audio_transforms(training: bool = True) -> Optional[ComposeByteTransforms]:
    """
    Get standard transforms for audio bytes.

    Args:
        training: Whether in training mode

    Returns:
        Compose of transforms (or None for validation)
    """
    if not training:
        return None

    return ComposeByteTransforms([
        ByteNoise(noise_prob=0.02, noise_strength=3),
    ])


def get_all_byte_transforms(
    height: int = 32,
    width: int = 32,
    training: bool = True,
) -> dict:
    """
    Get all byte transforms for all modalities.

    Args:
        height: Image height
        width: Image width
        training: Whether in training mode

    Returns:
        Dictionary mapping modality names to transforms
    """
    return {
        "vision": get_byte_vision_transforms(height, width, training),
        "text": get_byte_text_transforms(training),
        "audio": get_byte_audio_transforms(training),
    }
