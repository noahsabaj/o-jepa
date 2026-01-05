"""
Generic video format adapter.

Works with any directory containing video files.
Auto-discovers .mp4, .avi, .mkv, .webm, .mov files.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import logging

import numpy as np
import torch

from .base import FormatAdapter, ClipInfo

logger = logging.getLogger(__name__)

# Video file extensions to search for
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".webm", ".mov", ".m4v"}

# Optional imports
try:
    from decord import VideoReader, cpu, gpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    VideoReader = None

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False


class GenericVideoAdapter(FormatAdapter):
    """
    Generic adapter for any folder of video files.

    Discovers all video files in the directory and creates clips
    based on the specified duration and overlap.

    Works as the fallback when no specific format adapter matches.
    """

    @staticmethod
    def detect(data_dir: Path) -> bool:
        """
        Detect if directory contains video files.

        Returns True if any video files are found.
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            return False

        # Check for video files
        for ext in VIDEO_EXTENSIONS:
            if list(data_dir.rglob(f"*{ext}")):
                return True

        return False

    def discover_clips(
        self,
        clip_duration: float = 2.0,
        clip_overlap: float = 0.5,
    ) -> List[ClipInfo]:
        """
        Discover all clips from video files in the directory.

        Scans for video files and generates clip windows for each.
        """
        clips = []

        # Find all video files
        video_files = []
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(self.data_dir.rglob(f"*{ext}"))

        video_files = sorted(video_files)
        logger.info(f"Found {len(video_files)} video files in {self.data_dir}")

        for video_path in video_files:
            duration = self.get_video_duration(video_path)
            if duration <= 0:
                logger.warning(f"Could not get duration for {video_path}")
                continue

            # Generate clip windows
            windows = self.generate_clip_windows(duration, clip_duration, clip_overlap)

            for start, end in windows:
                clips.append(ClipInfo(
                    video_path=video_path,
                    start_time=start,
                    end_time=end,
                    view="default",
                ))

        logger.info(f"Generated {len(clips)} clips")
        return clips

    def load_frames(
        self,
        clip: ClipInfo,
        view: str = "default",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load video frames as byte tensor.

        Uses decord for fast GPU-accelerated decoding if available,
        falls back to PyAV otherwise.
        """
        if DECORD_AVAILABLE:
            return self._load_frames_decord(clip)
        elif AV_AVAILABLE:
            return self._load_frames_av(clip)
        else:
            raise ImportError(
                "No video decoder available. Install decord or av:\n"
                "  pip install decord\n"
                "  pip install av"
            )

    def _load_frames_decord(
        self,
        clip: ClipInfo,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load frames using decord (fast, GPU-accelerated)."""
        try:
            # Use CPU context (GPU requires special setup)
            vr = VideoReader(str(clip.video_path), ctx=cpu(0))
            fps = vr.get_avg_fps()

            # Calculate frame indices
            start_frame = int(clip.start_time * fps)
            end_frame = int(clip.end_time * fps)
            num_frames = int(clip.duration * self.fps)

            # Sample frames uniformly
            frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
            frame_indices = np.clip(frame_indices, 0, len(vr) - 1)

            # Load frames
            frames = vr.get_batch(frame_indices.tolist()).asnumpy()  # [T, H, W, 3]

            # Resize if needed
            if frames.shape[1:3] != self.frame_size:
                frames = self._resize_frames(frames)

            # Flatten to bytes: [T, H, W, 3] -> [T*H*W*3]
            bytes_array = frames.astype(np.uint8).flatten()
            bytes_tensor = torch.from_numpy(bytes_array).long()
            mask = torch.ones(len(bytes_tensor), dtype=torch.bool)

            return bytes_tensor, mask

        except Exception as e:
            logger.warning(f"Decord failed for {clip.video_path}: {e}")
            # Return zeros on error
            expected_len = int(clip.duration * self.fps) * self.frame_size[0] * self.frame_size[1] * 3
            return torch.zeros(expected_len, dtype=torch.long), torch.zeros(expected_len, dtype=torch.bool)

    def _load_frames_av(
        self,
        clip: ClipInfo,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load frames using PyAV."""
        try:
            with av.open(str(clip.video_path)) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate)

                # Seek to start
                start_pts = int(clip.start_time / stream.time_base)
                container.seek(start_pts, stream=stream)

                # Calculate target frames
                num_frames = int(clip.duration * self.fps)
                frame_interval = fps / self.fps

                frames = []
                frame_count = 0
                next_frame_idx = 0

                for frame in container.decode(stream):
                    current_time = float(frame.pts * stream.time_base)

                    if current_time > clip.end_time:
                        break

                    if current_time >= clip.start_time:
                        if frame_count >= next_frame_idx:
                            img = frame.to_ndarray(format="rgb24")
                            frames.append(img)
                            next_frame_idx += frame_interval

                            if len(frames) >= num_frames:
                                break

                        frame_count += 1

                if not frames:
                    raise ValueError("No frames extracted")

                frames = np.stack(frames)  # [T, H, W, 3]

                # Resize if needed
                if frames.shape[1:3] != self.frame_size:
                    frames = self._resize_frames(frames)

                # Flatten to bytes
                bytes_array = frames.astype(np.uint8).flatten()
                bytes_tensor = torch.from_numpy(bytes_array).long()
                mask = torch.ones(len(bytes_tensor), dtype=torch.bool)

                return bytes_tensor, mask

        except Exception as e:
            logger.warning(f"PyAV failed for {clip.video_path}: {e}")
            expected_len = int(clip.duration * self.fps) * self.frame_size[0] * self.frame_size[1] * 3
            return torch.zeros(expected_len, dtype=torch.long), torch.zeros(expected_len, dtype=torch.bool)

    def _resize_frames(self, frames: np.ndarray) -> np.ndarray:
        """Resize frames to target size."""
        try:
            from PIL import Image
            resized = []
            for frame in frames:
                img = Image.fromarray(frame)
                img = img.resize((self.frame_size[1], self.frame_size[0]))  # PIL uses (W, H)
                resized.append(np.array(img))
            return np.stack(resized)
        except ImportError:
            # Fallback: simple nearest-neighbor resize with numpy
            t, h, w, c = frames.shape
            th, tw = self.frame_size
            y_indices = np.linspace(0, h - 1, th, dtype=int)
            x_indices = np.linspace(0, w - 1, tw, dtype=int)
            return frames[:, y_indices][:, :, x_indices]

    def load_audio(
        self,
        clip: ClipInfo,
        interleave: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load audio as byte tensor.

        Audio is extracted from the video file or a separate audio file.
        Multi-channel audio is interleaved to preserve spatial information.
        """
        audio_path = clip.audio_path or clip.video_path

        if AV_AVAILABLE:
            return self._load_audio_av(audio_path, clip.start_time, clip.end_time, interleave)
        else:
            # Try torchaudio as fallback
            try:
                import torchaudio
                return self._load_audio_torchaudio(audio_path, clip.start_time, clip.end_time, interleave)
            except ImportError:
                raise ImportError(
                    "No audio decoder available. Install av or torchaudio:\n"
                    "  pip install av\n"
                    "  pip install torchaudio"
                )

    def _load_audio_av(
        self,
        audio_path: Path,
        start_time: float,
        end_time: float,
        interleave: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load audio using PyAV."""
        try:
            with av.open(str(audio_path)) as container:
                if not container.streams.audio:
                    # No audio stream
                    duration = end_time - start_time
                    expected_samples = int(duration * self.audio_sample_rate)
                    return torch.zeros(expected_samples, dtype=torch.long), torch.zeros(expected_samples, dtype=torch.bool)

                stream = container.streams.audio[0]

                # Seek to start
                start_pts = int(start_time / stream.time_base)
                container.seek(start_pts, stream=stream)

                # Decode audio
                samples = []
                for frame in container.decode(stream):
                    current_time = float(frame.pts * stream.time_base)

                    if current_time > end_time:
                        break

                    if current_time >= start_time:
                        audio_array = frame.to_ndarray()  # [channels, samples]
                        samples.append(audio_array)

                if not samples:
                    duration = end_time - start_time
                    expected_samples = int(duration * self.audio_sample_rate)
                    return torch.zeros(expected_samples, dtype=torch.long), torch.zeros(expected_samples, dtype=torch.bool)

                # Concatenate all samples
                audio = np.concatenate(samples, axis=1)  # [channels, samples]

                # Resample if needed
                if stream.rate != self.audio_sample_rate:
                    audio = self._resample_audio(audio, stream.rate)

                # Convert to bytes
                audio_bytes = self._audio_to_bytes(audio, interleave)

                bytes_tensor = torch.from_numpy(audio_bytes).long()
                mask = torch.ones(len(bytes_tensor), dtype=torch.bool)

                return bytes_tensor, mask

        except Exception as e:
            logger.warning(f"Audio extraction failed for {audio_path}: {e}")
            duration = end_time - start_time
            expected_samples = int(duration * self.audio_sample_rate)
            return torch.zeros(expected_samples, dtype=torch.long), torch.zeros(expected_samples, dtype=torch.bool)

    def _load_audio_torchaudio(
        self,
        audio_path: Path,
        start_time: float,
        end_time: float,
        interleave: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load audio using torchaudio."""
        import torchaudio

        try:
            # Get audio info
            info = torchaudio.info(str(audio_path))
            sr = info.sample_rate

            # Calculate frame offsets
            frame_offset = int(start_time * sr)
            num_frames = int((end_time - start_time) * sr)

            # Load audio segment
            waveform, sr = torchaudio.load(
                str(audio_path),
                frame_offset=frame_offset,
                num_frames=num_frames,
            )

            # Resample if needed
            if sr != self.audio_sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.audio_sample_rate)(waveform)

            audio = waveform.numpy()  # [channels, samples]

            # Convert to bytes
            audio_bytes = self._audio_to_bytes(audio, interleave)

            bytes_tensor = torch.from_numpy(audio_bytes).long()
            mask = torch.ones(len(bytes_tensor), dtype=torch.bool)

            return bytes_tensor, mask

        except Exception as e:
            logger.warning(f"Torchaudio failed for {audio_path}: {e}")
            duration = end_time - start_time
            expected_samples = int(duration * self.audio_sample_rate)
            return torch.zeros(expected_samples, dtype=torch.long), torch.zeros(expected_samples, dtype=torch.bool)

    def _resample_audio(self, audio: np.ndarray, original_sr: int) -> np.ndarray:
        """Simple audio resampling using linear interpolation."""
        channels, samples = audio.shape
        ratio = self.audio_sample_rate / original_sr
        new_samples = int(samples * ratio)

        # Linear interpolation
        old_indices = np.linspace(0, samples - 1, new_samples)
        resampled = np.zeros((channels, new_samples), dtype=audio.dtype)

        for c in range(channels):
            resampled[c] = np.interp(old_indices, np.arange(samples), audio[c])

        return resampled

    def _audio_to_bytes(self, audio: np.ndarray, interleave: bool) -> np.ndarray:
        """
        Convert audio to 8-bit unsigned bytes.

        Args:
            audio: Audio array [channels, samples]
            interleave: If True, interleave channels for spatial preservation

        Returns:
            Byte array
        """
        # Normalize to [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        # Convert to [0, 255]
        audio_uint8 = ((audio + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

        if interleave and audio_uint8.shape[0] > 1:
            # Interleave: [ch1_s1, ch2_s1, ..., chN_s1, ch1_s2, ...]
            channels, samples = audio_uint8.shape
            interleaved = audio_uint8.T.flatten()  # Transpose then flatten
            return interleaved
        else:
            # Flatten directly (mono or concatenate)
            return audio_uint8.flatten()

    def get_available_views(self) -> List[str]:
        """Generic adapter only has default view."""
        return ["default"]
