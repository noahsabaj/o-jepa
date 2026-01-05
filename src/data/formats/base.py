"""
Base classes for video format adapters.

Defines the FormatAdapter protocol and ClipInfo dataclass that all
format-specific adapters must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch


@dataclass
class ClipInfo:
    """
    Information about a single video clip.

    Attributes:
        video_path: Path to the video file
        start_time: Start time in seconds
        end_time: End time in seconds
        view: Camera view name (e.g., "ego", "exo", "cam01")
        audio_path: Optional separate audio file path
        metadata: Optional additional metadata
    """
    video_path: Path
    start_time: float
    end_time: float
    view: str = "default"
    audio_path: Optional[Path] = None
    metadata: Optional[dict] = None

    @property
    def duration(self) -> float:
        """Clip duration in seconds."""
        return self.end_time - self.start_time


class FormatAdapter(ABC):
    """
    Abstract base class for dataset format adapters.

    Each dataset format (Ego-Exo4D, Kinetics, Something-Something, etc.)
    implements this interface to handle its specific file structure.

    The adapter is responsible for:
    1. Detecting if it can handle a given directory
    2. Discovering all clips in the dataset
    3. Loading video frames as byte tensors
    4. Loading audio as byte tensors
    """

    def __init__(
        self,
        data_dir: Path,
        frame_size: Tuple[int, int] = (224, 224),
        fps: int = 8,
        audio_sample_rate: int = 16000,
    ):
        """
        Initialize the format adapter.

        Args:
            data_dir: Root directory of the dataset
            frame_size: Target frame size (H, W)
            fps: Frames per second to extract
            audio_sample_rate: Target audio sample rate
        """
        self.data_dir = Path(data_dir)
        self.frame_size = frame_size
        self.fps = fps
        self.audio_sample_rate = audio_sample_rate

    @staticmethod
    @abstractmethod
    def detect(data_dir: Path) -> bool:
        """
        Check if this adapter can handle the given directory.

        Args:
            data_dir: Directory to check

        Returns:
            True if this adapter can handle this directory structure
        """
        ...

    @abstractmethod
    def discover_clips(
        self,
        clip_duration: float = 2.0,
        clip_overlap: float = 0.5,
    ) -> List[ClipInfo]:
        """
        Discover all clips in the dataset.

        Args:
            clip_duration: Duration of each clip in seconds
            clip_overlap: Overlap between clips (0.0 to 0.9)

        Returns:
            List of ClipInfo objects
        """
        ...

    @abstractmethod
    def load_frames(
        self,
        clip: ClipInfo,
        view: str = "default",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load video frames for a clip as byte tensor.

        Args:
            clip: Clip information
            view: Camera view to load

        Returns:
            Tuple of (frames bytes [T*H*W*3], mask [T*H*W*3])
        """
        ...

    @abstractmethod
    def load_audio(
        self,
        clip: ClipInfo,
        interleave: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load audio for a clip as byte tensor.

        Args:
            clip: Clip information
            interleave: If True, interleave multi-channel audio

        Returns:
            Tuple of (audio bytes [samples], mask [samples])
        """
        ...

    @abstractmethod
    def get_available_views(self) -> List[str]:
        """
        Get list of available camera views.

        Returns:
            List of view names (e.g., ["ego", "exo"])
        """
        ...

    def get_video_duration(self, video_path: Path) -> float:
        """
        Get duration of a video file in seconds.

        Args:
            video_path: Path to video file

        Returns:
            Duration in seconds
        """
        try:
            import av
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                return float(stream.duration * stream.time_base)
        except Exception:
            # Fallback: try with decord
            try:
                from decord import VideoReader
                vr = VideoReader(str(video_path))
                return len(vr) / vr.get_avg_fps()
            except Exception:
                return 0.0

    def generate_clip_windows(
        self,
        duration: float,
        clip_duration: float,
        clip_overlap: float,
    ) -> List[Tuple[float, float]]:
        """
        Generate clip windows for a video.

        Args:
            duration: Total video duration
            clip_duration: Duration of each clip
            clip_overlap: Overlap between clips (0.0 to 0.9)

        Returns:
            List of (start_time, end_time) tuples
        """
        if duration <= 0 or clip_duration <= 0:
            return []

        clip_overlap = max(0.0, min(0.9, clip_overlap))
        stride = clip_duration * (1.0 - clip_overlap)

        windows = []
        start = 0.0

        while start + clip_duration <= duration:
            windows.append((start, start + clip_duration))
            start += stride

        # Add final partial clip if significant
        if start < duration and (duration - start) > clip_duration * 0.5:
            windows.append((duration - clip_duration, duration))

        return windows
