"""
Universal Video+Audio Dataset for O-JEPA.

A flexible, pluggable dataset system that works with any video+audio dataset.
Designed following Steve Jobs' simplicity principle and Yann LeCun's
world model philosophy.

Usage:
    # Auto-detect format
    dataset = VideoAudioDataset("/path/to/videos")

    # Explicit format
    dataset = VideoAudioDataset("/path/to/ego_exo4d", format="ego_exo4d")

    # Multi-view
    dataset = VideoAudioDataset("/path/to/data", views=["ego", "exo"])
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

import torch
from torch.utils.data import Dataset

from .formats.base import FormatAdapter, ClipInfo

logger = logging.getLogger(__name__)


def get_format_adapter(
    data_dir: Union[str, Path],
    format: str = "auto",
    frame_size: Tuple[int, int] = (224, 224),
    fps: int = 8,
    audio_sample_rate: int = 16000,
) -> FormatAdapter:
    """
    Get the appropriate format adapter for a dataset.

    Args:
        data_dir: Path to dataset directory
        format: Format name ("auto", "generic", "ego_exo4d")
        frame_size: Target frame size (H, W)
        fps: Frames per second to extract
        audio_sample_rate: Target audio sample rate

    Returns:
        FormatAdapter instance
    """
    from .formats import FORMAT_REGISTRY

    data_dir = Path(data_dir)

    if format == "auto":
        # Try each registered adapter
        for name, adapter_class in FORMAT_REGISTRY.items():
            if name == "generic":
                continue  # Try generic last
            if adapter_class.detect(data_dir):
                logger.info(f"Auto-detected format: {name}")
                return adapter_class(
                    data_dir=data_dir,
                    frame_size=frame_size,
                    fps=fps,
                    audio_sample_rate=audio_sample_rate,
                )

        # Fallback to generic
        if "generic" in FORMAT_REGISTRY:
            logger.info("Using generic video format")
            return FORMAT_REGISTRY["generic"](
                data_dir=data_dir,
                frame_size=frame_size,
                fps=fps,
                audio_sample_rate=audio_sample_rate,
            )

        raise ValueError(f"No suitable format adapter found for {data_dir}")

    elif format in FORMAT_REGISTRY:
        return FORMAT_REGISTRY[format](
            data_dir=data_dir,
            frame_size=frame_size,
            fps=fps,
            audio_sample_rate=audio_sample_rate,
        )

    else:
        available = list(FORMAT_REGISTRY.keys())
        raise ValueError(f"Unknown format '{format}'. Available: {available}")


class VideoAudioDataset(Dataset):
    """
    Universal dock for video+audio datasets.

    Works with any dataset format through pluggable adapters.
    Returns byte tensors compatible with O-JEPA training.

    Args:
        data_dir: Path to dataset directory
        format: Format name ("auto", "generic", "ego_exo4d")
        clip_duration: Duration of each clip in seconds
        clip_overlap: Overlap between clips (0.0 to 0.9)
        fps: Frames per second to extract
        audio_sample_rate: Target audio sample rate
        frame_size: Target frame size (H, W)
        modalities: List of modalities to load ("vision", "audio")
        views: List of camera views to load

    Example:
        >>> dataset = VideoAudioDataset("/data/videos", clip_duration=2.0)
        >>> sample = dataset[0]
        >>> sample["vision"].shape  # [T*H*W*3]
        >>> sample["audio"].shape   # [samples]
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        format: str = "auto",
        clip_duration: float = 2.0,
        clip_overlap: float = 0.5,
        fps: int = 8,
        audio_sample_rate: int = 16000,
        frame_size: Tuple[int, int] = (224, 224),
        modalities: Optional[List[str]] = None,
        views: Optional[List[str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.format = format
        self.clip_duration = clip_duration
        self.clip_overlap = max(0.0, min(0.9, clip_overlap))
        self.fps = fps
        self.audio_sample_rate = audio_sample_rate
        self.frame_size = frame_size
        self.modalities = modalities or ["vision", "audio"]
        self.views = views or ["default"]

        # Get format adapter
        self.adapter = get_format_adapter(
            data_dir=self.data_dir,
            format=format,
            frame_size=frame_size,
            fps=fps,
            audio_sample_rate=audio_sample_rate,
        )

        # Discover clips
        self.clips = self.adapter.discover_clips(
            clip_duration=clip_duration,
            clip_overlap=clip_overlap,
        )

        # Calculate expected sequence lengths
        self.vision_seq_len = int(clip_duration * fps) * frame_size[0] * frame_size[1] * 3
        self.audio_seq_len = int(clip_duration * audio_sample_rate)

        logger.info(
            f"VideoAudioDataset initialized:\n"
            f"  Directory: {self.data_dir}\n"
            f"  Format: {format}\n"
            f"  Clips: {len(self.clips)}\n"
            f"  Duration: {clip_duration}s, Overlap: {clip_overlap*100:.0f}%\n"
            f"  Vision seq len: {self.vision_seq_len}\n"
            f"  Audio seq len: {self.audio_seq_len}"
        )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a clip as byte tensors.

        Returns:
            Dict with keys:
                - "vision": Frames as bytes [T*H*W*3]
                - "audio": Audio as bytes [samples]
                - "masks": Dict of masks for each modality
                - For multi-view: "vision_ego", "vision_exo", etc.
        """
        clip = self.clips[idx]
        result = {}
        masks = {}

        # Load vision
        if "vision" in self.modalities:
            if len(self.views) == 1:
                # Single view
                view = self.views[0]
                bytes_tensor, mask = self.adapter.load_frames(clip, view)
                result["vision"] = bytes_tensor
                masks["vision"] = mask
            else:
                # Multi-view: separate keys
                for view in self.views:
                    bytes_tensor, mask = self.adapter.load_frames(clip, view)
                    result[f"vision_{view}"] = bytes_tensor
                    masks[f"vision_{view}"] = mask

        # Load audio
        if "audio" in self.modalities:
            bytes_tensor, mask = self.adapter.load_audio(clip, interleave=True)
            result["audio"] = bytes_tensor
            masks["audio"] = mask

        result["masks"] = masks

        return result

    def get_available_views(self) -> List[str]:
        """Get list of available camera views."""
        return self.adapter.get_available_views()

    @property
    def num_clips(self) -> int:
        """Total number of clips."""
        return len(self.clips)


class VideoAudioMockDataset(Dataset):
    """
    Mock dataset for testing without real video files.

    Generates random byte data matching the expected format.
    """

    def __init__(
        self,
        num_samples: int = 100,
        clip_duration: float = 2.0,
        fps: int = 8,
        audio_sample_rate: int = 16000,
        frame_size: Tuple[int, int] = (224, 224),
        modalities: Optional[List[str]] = None,
        views: Optional[List[str]] = None,
        num_audio_channels: int = 1,
    ):
        self.num_samples = num_samples
        self.clip_duration = clip_duration
        self.fps = fps
        self.audio_sample_rate = audio_sample_rate
        self.frame_size = frame_size
        self.modalities = modalities or ["vision", "audio"]
        self.views = views or ["default"]
        self.num_audio_channels = num_audio_channels

        # Calculate sequence lengths
        self.vision_seq_len = int(clip_duration * fps) * frame_size[0] * frame_size[1] * 3
        # Interleaved audio: samples * channels
        self.audio_seq_len = int(clip_duration * audio_sample_rate) * num_audio_channels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = {}
        masks = {}

        # Generate vision
        if "vision" in self.modalities:
            if len(self.views) == 1:
                result["vision"] = torch.randint(0, 256, (self.vision_seq_len,), dtype=torch.long)
                masks["vision"] = torch.ones(self.vision_seq_len, dtype=torch.bool)
            else:
                for view in self.views:
                    result[f"vision_{view}"] = torch.randint(0, 256, (self.vision_seq_len,), dtype=torch.long)
                    masks[f"vision_{view}"] = torch.ones(self.vision_seq_len, dtype=torch.bool)

        # Generate audio
        if "audio" in self.modalities:
            result["audio"] = torch.randint(0, 256, (self.audio_seq_len,), dtype=torch.long)
            masks["audio"] = torch.ones(self.audio_seq_len, dtype=torch.bool)

        result["masks"] = masks

        return result
