"""
Byte-level datasets for O-JEPA.

Provides datasets that load any modality as raw bytes (0-255).
"""

from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset

# Optional imports for real data loading
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except (ImportError, OSError):
    # OSError can happen when torchaudio lib fails to load
    TORCHAUDIO_AVAILABLE = False
    torchaudio = None


class ByteDataset(Dataset):
    """
    Unified dataset that loads any modality as raw bytes.

    Supports:
    - Vision: RGB images as flattened bytes
    - Text: UTF-8 encoded bytes
    - Audio: PCM audio samples as bytes

    Args:
        data_dir: Root directory containing data
        modalities: List of modalities to load
        vision_size: Target image size (H, W)
        text_max_len: Maximum text length in bytes
        audio_max_len: Maximum audio length in bytes
        audio_sample_rate: Target sample rate for audio
    """

    def __init__(
        self,
        data_dir: Path,
        modalities: List[str] = ["vision", "text", "audio"],
        vision_size: Tuple[int, int] = (32, 32),
        text_max_len: int = 1024,
        audio_max_len: int = 8000,
        audio_sample_rate: int = 16000,
    ):
        self.data_dir = Path(data_dir)
        self.modalities = modalities
        self.vision_size = vision_size
        self.text_max_len = text_max_len
        self.audio_max_len = audio_max_len
        self.audio_sample_rate = audio_sample_rate

        # Vision sequence length
        self.vision_seq_len = vision_size[0] * vision_size[1] * 3

        # Load sample list
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        """Load list of samples from data directory."""
        samples = []

        # Try to load from manifest file
        manifest_path = self.data_dir / "manifest.txt"
        if manifest_path.exists():
            with open(manifest_path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    sample = {}
                    for part in parts:
                        key, value = part.split(":")
                        sample[key] = value
                    samples.append(sample)
        else:
            # Auto-discover files
            if "vision" in self.modalities:
                image_dir = self.data_dir / "images"
                if image_dir.exists():
                    for img_path in sorted(image_dir.glob("*.jpg")) + \
                                   sorted(image_dir.glob("*.png")):
                        samples.append({"image_path": str(img_path)})

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a sample as byte tensors."""
        sample = self.samples[idx]
        result = {}
        masks = {}

        for modality in self.modalities:
            if modality == "vision" and "image_path" in sample:
                bytes_tensor, mask = self._load_image_bytes(sample["image_path"])
                result["vision"] = bytes_tensor
                masks["vision"] = mask

            elif modality == "text" and "text" in sample:
                bytes_tensor, mask = self._load_text_bytes(sample["text"])
                result["text"] = bytes_tensor
                masks["text"] = mask

            elif modality == "audio" and "audio_path" in sample:
                bytes_tensor, mask = self._load_audio_bytes(sample["audio_path"])
                result["audio"] = bytes_tensor
                masks["audio"] = mask

        result["masks"] = masks
        return result

    def _load_image_bytes(
        self,
        path: Union[str, Path],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load image as raw RGB bytes.

        Args:
            path: Path to image file

        Returns:
            Tuple of (bytes tensor [H*W*3], mask tensor [H*W*3])
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL required for image loading")

        img = Image.open(path).convert("RGB")
        img = img.resize(self.vision_size)

        # Convert to bytes: [H, W, 3] -> [H*W*3]
        bytes_array = np.array(img, dtype=np.uint8).flatten()
        bytes_tensor = torch.from_numpy(bytes_array).long()

        # Mask: all ones for images (no padding)
        mask = torch.ones(len(bytes_tensor), dtype=torch.bool)

        return bytes_tensor, mask

    def _load_text_bytes(
        self,
        text: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load text as UTF-8 bytes.

        Args:
            text: Text string

        Returns:
            Tuple of (bytes tensor [max_len], mask tensor [max_len])
        """
        text_bytes = text.encode("utf-8")[:self.text_max_len]
        actual_len = len(text_bytes)

        # Create padded array
        padded = np.zeros(self.text_max_len, dtype=np.uint8)
        padded[:actual_len] = list(text_bytes)
        bytes_tensor = torch.from_numpy(padded).long()

        # Mask: 1 for actual bytes, 0 for padding
        mask = torch.zeros(self.text_max_len, dtype=torch.bool)
        mask[:actual_len] = True

        return bytes_tensor, mask

    def _load_audio_bytes(
        self,
        path: Union[str, Path],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load audio as raw PCM bytes (8-bit unsigned).

        Args:
            path: Path to audio file

        Returns:
            Tuple of (bytes tensor [max_len], mask tensor [max_len])
        """
        if not TORCHAUDIO_AVAILABLE:
            raise ImportError("torchaudio required for audio loading")

        # Load audio
        waveform, sr = torchaudio.load(path)

        # Resample if needed
        if sr != self.audio_sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.audio_sample_rate)(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Convert to 8-bit unsigned PCM bytes
        # waveform is in [-1, 1], convert to [0, 255]
        pcm = ((waveform[0] + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)
        pcm_bytes = pcm.numpy()

        # Truncate or pad
        actual_len = min(len(pcm_bytes), self.audio_max_len)
        padded = np.zeros(self.audio_max_len, dtype=np.uint8)
        padded[:actual_len] = pcm_bytes[:actual_len]
        bytes_tensor = torch.from_numpy(padded).long()

        # Mask
        mask = torch.zeros(self.audio_max_len, dtype=torch.bool)
        mask[:actual_len] = True

        return bytes_tensor, mask


class SyntheticByteDataset(Dataset):
    """
    Generate random byte data for testing.

    Useful for testing the training pipeline without real data.

    Args:
        num_samples: Number of samples to generate
        modalities: List of modalities to include
        vision_seq_len: Vision sequence length (H*W*3)
        text_seq_len: Text sequence length
        audio_seq_len: Audio sequence length
    """

    def __init__(
        self,
        num_samples: int = 1000,
        modalities: List[str] = ["vision", "text", "audio"],
        vision_seq_len: int = 3072,  # 32×32×3
        text_seq_len: int = 1024,
        audio_seq_len: int = 8000,
    ):
        self.num_samples = num_samples
        self.modalities = modalities
        self.seq_lens = {
            "vision": vision_seq_len,
            "text": text_seq_len,
            "audio": audio_seq_len,
        }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate random byte data."""
        result = {}
        masks = {}

        for mod in self.modalities:
            seq_len = self.seq_lens[mod]

            # Random bytes
            bytes_tensor = torch.randint(0, 256, (seq_len,), dtype=torch.long)
            result[mod] = bytes_tensor

            # All valid for synthetic data
            masks[mod] = torch.ones(seq_len, dtype=torch.bool)

        result["masks"] = masks
        return result


class PairedByteDataset(Dataset):
    """
    Dataset for paired multimodal data (e.g., image-caption pairs).

    Expects data in format:
    - images/: Directory of images
    - captions.txt: Tab-separated image_name and caption

    Args:
        data_dir: Root directory
        vision_size: Target image size
        text_max_len: Maximum caption length in bytes
    """

    def __init__(
        self,
        data_dir: Path,
        vision_size: Tuple[int, int] = (32, 32),
        text_max_len: int = 1024,
    ):
        self.data_dir = Path(data_dir)
        self.vision_size = vision_size
        self.text_max_len = text_max_len
        self.vision_seq_len = vision_size[0] * vision_size[1] * 3

        # Load pairs
        self.pairs = self._load_pairs()

    def _load_pairs(self) -> List[Dict]:
        """Load image-caption pairs."""
        pairs = []
        captions_path = self.data_dir / "captions.txt"

        if captions_path.exists():
            with open(captions_path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        pairs.append({
                            "image_path": self.data_dir / "images" / parts[0],
                            "caption": parts[1],
                        })
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a paired sample."""
        pair = self.pairs[idx]
        result = {}
        masks = {}

        # Load image
        if PIL_AVAILABLE and Path(pair["image_path"]).exists():
            img = Image.open(pair["image_path"]).convert("RGB")
            img = img.resize(self.vision_size)
            bytes_array = np.array(img, dtype=np.uint8).flatten()
            result["vision"] = torch.from_numpy(bytes_array).long()
            masks["vision"] = torch.ones(len(bytes_array), dtype=torch.bool)
        else:
            # Fallback to random data
            result["vision"] = torch.randint(0, 256, (self.vision_seq_len,), dtype=torch.long)
            masks["vision"] = torch.ones(self.vision_seq_len, dtype=torch.bool)

        # Load caption
        text_bytes = pair["caption"].encode("utf-8")[:self.text_max_len]
        actual_len = len(text_bytes)
        padded = np.zeros(self.text_max_len, dtype=np.uint8)
        padded[:actual_len] = list(text_bytes)
        result["text"] = torch.from_numpy(padded).long()
        mask = torch.zeros(self.text_max_len, dtype=torch.bool)
        mask[:actual_len] = True
        masks["text"] = mask

        result["masks"] = masks
        return result
