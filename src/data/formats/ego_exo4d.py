"""
Ego-Exo4D format adapter.

Handles the Ego-Exo4D dataset structure with:
- Synchronized ego (first-person) and exo (third-person) views
- 7-channel spatial audio from GoPros
- Additional modalities: IMU, eye gaze, 3D point clouds

Dataset: https://ego-exo4d-data.org/
Paper: https://arxiv.org/abs/2311.18259

This is a STUB implementation. Full implementation will be added
when Ego-Exo4D dataset access is approved.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import logging

import numpy as np
import torch

from .base import FormatAdapter, ClipInfo

logger = logging.getLogger(__name__)


class EgoExo4DAdapter(FormatAdapter):
    """
    Adapter for Ego-Exo4D dataset.

    Expected directory structure (based on Ego-Exo4D documentation):
        data_dir/
            takes/
                <take_uid>/
                    frame_aligned_videos/
                        ego/
                            <video_file>.mp4
                        exo/
                            cam01/<video_file>.mp4
                            cam02/<video_file>.mp4
                            ...
                    audio/
                        <audio_file>.wav  # 7-channel spatial audio
                    metadata.json
            annotations/
                ...
            takes.json  # Master manifest

    Features:
        - Synchronized ego + exo views
        - 7-channel spatial audio (interleaved for O-JEPA)
        - Optional: IMU, eye gaze as additional modalities
    """

    # Ego-Exo4D specific constants
    MANIFEST_FILE = "takes.json"
    TAKES_DIR = "takes"
    EGO_SUBDIR = "ego"
    EXO_SUBDIR = "exo"
    AUDIO_SUBDIR = "audio"

    def __init__(
        self,
        data_dir: Path,
        frame_size: Tuple[int, int] = (224, 224),
        fps: int = 8,
        audio_sample_rate: int = 16000,
    ):
        super().__init__(data_dir, frame_size, fps, audio_sample_rate)

        # Ego-Exo4D specific state
        self._manifest = None
        self._takes = []

        # Load manifest if available
        self._load_manifest()

    def _load_manifest(self):
        """Load the takes.json manifest file."""
        manifest_path = self.data_dir / self.MANIFEST_FILE

        if manifest_path.exists():
            import json
            with open(manifest_path) as f:
                self._manifest = json.load(f)

            # Parse takes
            if isinstance(self._manifest, list):
                self._takes = self._manifest
            elif isinstance(self._manifest, dict) and "takes" in self._manifest:
                self._takes = self._manifest["takes"]

            logger.info(f"Loaded Ego-Exo4D manifest with {len(self._takes)} takes")

    @staticmethod
    def detect(data_dir: Path) -> bool:
        """
        Detect if this is an Ego-Exo4D dataset.

        Checks for:
        1. takes.json manifest file
        2. takes/ directory structure
        """
        data_dir = Path(data_dir)

        # Check for manifest
        if (data_dir / "takes.json").exists():
            return True

        # Check for takes directory structure
        takes_dir = data_dir / "takes"
        if takes_dir.exists():
            # Check if any subdirectory has ego/exo structure
            for take_dir in takes_dir.iterdir():
                if take_dir.is_dir():
                    if (take_dir / "frame_aligned_videos" / "ego").exists():
                        return True
                    if (take_dir / "ego").exists():
                        return True

        return False

    def discover_clips(
        self,
        clip_duration: float = 2.0,
        clip_overlap: float = 0.5,
    ) -> List[ClipInfo]:
        """
        Discover all clips from Ego-Exo4D takes.

        STUB: Returns empty list until dataset is available.
        When implemented, will:
        1. Parse takes.json for video paths
        2. Generate synchronized ego+exo clips
        3. Link to spatial audio
        """
        clips = []

        if not self._takes:
            logger.warning(
                "Ego-Exo4D manifest not loaded. "
                "This is a stub implementation - full support coming when dataset arrives."
            )
            return clips

        # STUB: Parse takes and generate clips
        # TODO: Implement when dataset is available
        for take in self._takes:
            take_uid = take.get("take_uid", take.get("uid", "unknown"))
            take_dir = self.data_dir / self.TAKES_DIR / take_uid

            if not take_dir.exists():
                continue

            # Find ego video
            ego_dir = take_dir / "frame_aligned_videos" / self.EGO_SUBDIR
            if not ego_dir.exists():
                ego_dir = take_dir / self.EGO_SUBDIR

            if not ego_dir.exists():
                continue

            # Find video files
            video_files = list(ego_dir.glob("*.mp4"))
            if not video_files:
                continue

            video_path = video_files[0]
            duration = self.get_video_duration(video_path)

            if duration <= 0:
                continue

            # Generate clip windows
            windows = self.generate_clip_windows(duration, clip_duration, clip_overlap)

            for start, end in windows:
                clips.append(ClipInfo(
                    video_path=video_path,
                    start_time=start,
                    end_time=end,
                    view="ego",
                    metadata={"take_uid": take_uid},
                ))

        logger.info(f"Discovered {len(clips)} clips from Ego-Exo4D")
        return clips

    def load_frames(
        self,
        clip: ClipInfo,
        view: str = "ego",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load video frames for the specified view.

        STUB: Delegates to generic video loading.
        When fully implemented, will handle ego/exo synchronization.
        """
        # For now, use the same video loading as generic adapter
        from .generic import GenericVideoAdapter

        generic = GenericVideoAdapter(
            data_dir=self.data_dir,
            frame_size=self.frame_size,
            fps=self.fps,
            audio_sample_rate=self.audio_sample_rate,
        )

        # TODO: When implemented, handle view switching
        # If view == "exo", find the corresponding exo video
        if view == "exo" and clip.metadata:
            take_uid = clip.metadata.get("take_uid")
            if take_uid:
                exo_dir = self.data_dir / self.TAKES_DIR / take_uid / "frame_aligned_videos" / self.EXO_SUBDIR
                if exo_dir.exists():
                    # Find first exo camera
                    cam_dirs = sorted([d for d in exo_dir.iterdir() if d.is_dir()])
                    if cam_dirs:
                        exo_videos = list(cam_dirs[0].glob("*.mp4"))
                        if exo_videos:
                            exo_clip = ClipInfo(
                                video_path=exo_videos[0],
                                start_time=clip.start_time,
                                end_time=clip.end_time,
                                view="exo",
                            )
                            return generic.load_frames(exo_clip)

        return generic.load_frames(clip)

    def load_audio(
        self,
        clip: ClipInfo,
        interleave: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load 7-channel spatial audio.

        STUB: Uses generic audio loading.
        When fully implemented, will handle 7-channel spatial audio properly.
        """
        from .generic import GenericVideoAdapter

        generic = GenericVideoAdapter(
            data_dir=self.data_dir,
            frame_size=self.frame_size,
            fps=self.fps,
            audio_sample_rate=self.audio_sample_rate,
        )

        # TODO: When implemented, find and load spatial audio file
        # Ego-Exo4D has separate .wav files with 7-channel audio
        if clip.metadata:
            take_uid = clip.metadata.get("take_uid")
            if take_uid:
                audio_dir = self.data_dir / self.TAKES_DIR / take_uid / self.AUDIO_SUBDIR
                if audio_dir.exists():
                    audio_files = list(audio_dir.glob("*.wav"))
                    if audio_files:
                        audio_clip = ClipInfo(
                            video_path=clip.video_path,
                            start_time=clip.start_time,
                            end_time=clip.end_time,
                            audio_path=audio_files[0],
                        )
                        return generic.load_audio(audio_clip, interleave=interleave)

        return generic.load_audio(clip, interleave=interleave)

    def get_available_views(self) -> List[str]:
        """
        Get available camera views.

        Ego-Exo4D has:
        - "ego": First-person view from head-mounted camera
        - "exo": Third-person views from static cameras
        """
        return ["ego", "exo"]

    def get_exo_cameras(self, take_uid: str) -> List[str]:
        """
        Get list of available exo camera IDs for a take.

        Args:
            take_uid: Take identifier

        Returns:
            List of camera IDs (e.g., ["cam01", "cam02", "cam03"])
        """
        exo_dir = self.data_dir / self.TAKES_DIR / take_uid / "frame_aligned_videos" / self.EXO_SUBDIR

        if not exo_dir.exists():
            return []

        cameras = sorted([d.name for d in exo_dir.iterdir() if d.is_dir()])
        return cameras
