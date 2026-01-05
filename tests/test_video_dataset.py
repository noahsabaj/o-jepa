"""
Tests for Video+Audio Dataset.

Tests the VideoAudioDataset universal dock and format adapters.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from src.data import (
    VideoAudioDataset,
    VideoAudioMockDataset,
    FormatAdapter,
    ClipInfo,
    GenericVideoAdapter,
    EgoExo4DAdapter,
    FORMAT_REGISTRY,
    get_format_adapter,
    get_available_formats,
    register_format,
)


# =============================================================================
# CLIP INFO TESTS
# =============================================================================

class TestClipInfo:
    """Tests for ClipInfo dataclass."""

    def test_clip_info_creation(self):
        """Test basic ClipInfo creation."""
        clip = ClipInfo(
            video_path=Path("/test/video.mp4"),
            start_time=0.0,
            end_time=2.0,
        )

        assert clip.video_path == Path("/test/video.mp4")
        assert clip.start_time == 0.0
        assert clip.end_time == 2.0
        assert clip.view == "default"
        assert clip.audio_path is None
        assert clip.metadata is None

    def test_clip_info_duration(self):
        """Test duration property."""
        clip = ClipInfo(
            video_path=Path("/test/video.mp4"),
            start_time=1.5,
            end_time=4.0,
        )

        assert clip.duration == 2.5

    def test_clip_info_with_metadata(self):
        """Test ClipInfo with metadata."""
        clip = ClipInfo(
            video_path=Path("/test/video.mp4"),
            start_time=0.0,
            end_time=2.0,
            view="ego",
            metadata={"take_uid": "take_001"},
        )

        assert clip.view == "ego"
        assert clip.metadata["take_uid"] == "take_001"


# =============================================================================
# FORMAT REGISTRY TESTS
# =============================================================================

class TestFormatRegistry:
    """Tests for format registry."""

    def test_available_formats(self):
        """Test get_available_formats returns expected formats."""
        formats = get_available_formats()

        assert "generic" in formats
        assert "ego_exo4d" in formats

    def test_format_registry_contents(self):
        """Test FORMAT_REGISTRY contains expected adapters."""
        assert GenericVideoAdapter == FORMAT_REGISTRY["generic"]
        assert EgoExo4DAdapter == FORMAT_REGISTRY["ego_exo4d"]

    def test_register_format(self):
        """Test registering a new format."""
        class DummyAdapter(FormatAdapter):
            @staticmethod
            def detect(data_dir):
                return False

            def discover_clips(self, clip_duration=2.0, clip_overlap=0.5):
                return []

            def load_frames(self, clip, view="default"):
                return torch.zeros(1), torch.zeros(1, dtype=torch.bool)

            def load_audio(self, clip, interleave=True):
                return torch.zeros(1), torch.zeros(1, dtype=torch.bool)

            def get_available_views(self):
                return ["default"]

        register_format("dummy", DummyAdapter)

        assert "dummy" in FORMAT_REGISTRY
        assert FORMAT_REGISTRY["dummy"] == DummyAdapter

        # Cleanup
        del FORMAT_REGISTRY["dummy"]


# =============================================================================
# GENERIC ADAPTER TESTS
# =============================================================================

class TestGenericVideoAdapter:
    """Tests for GenericVideoAdapter."""

    def test_detect_empty_dir(self, tmp_path):
        """Test detection on empty directory."""
        assert GenericVideoAdapter.detect(tmp_path) is False

    def test_detect_with_video_files(self, tmp_path):
        """Test detection with video files present."""
        # Create dummy video file
        (tmp_path / "video.mp4").touch()

        assert GenericVideoAdapter.detect(tmp_path) is True

    def test_detect_various_extensions(self, tmp_path):
        """Test detection with various video extensions."""
        for ext in [".mp4", ".avi", ".mkv", ".webm", ".mov"]:
            (tmp_path / f"video{ext}").touch()

        assert GenericVideoAdapter.detect(tmp_path) is True

    def test_generate_clip_windows(self, tmp_path):
        """Test clip window generation."""
        adapter = GenericVideoAdapter(data_dir=tmp_path)

        # 10 second video, 2 second clips, 50% overlap
        windows = adapter.generate_clip_windows(
            duration=10.0,
            clip_duration=2.0,
            clip_overlap=0.5,
        )

        # Expected: 0-2, 1-3, 2-4, 3-5, 4-6, 5-7, 6-8, 7-9, 8-10
        assert len(windows) == 9
        assert windows[0] == (0.0, 2.0)
        assert windows[1] == (1.0, 3.0)
        assert windows[-1] == (8.0, 10.0)

    def test_generate_clip_windows_no_overlap(self, tmp_path):
        """Test clip window generation with no overlap."""
        adapter = GenericVideoAdapter(data_dir=tmp_path)

        windows = adapter.generate_clip_windows(
            duration=10.0,
            clip_duration=2.0,
            clip_overlap=0.0,
        )

        # Expected: 0-2, 2-4, 4-6, 6-8, 8-10
        assert len(windows) == 5
        assert windows[0] == (0.0, 2.0)
        assert windows[1] == (2.0, 4.0)

    def test_generate_clip_windows_short_video(self, tmp_path):
        """Test clip windows for video shorter than clip duration."""
        adapter = GenericVideoAdapter(data_dir=tmp_path)

        windows = adapter.generate_clip_windows(
            duration=1.0,
            clip_duration=2.0,
            clip_overlap=0.5,
        )

        assert len(windows) == 0

    def test_get_available_views(self, tmp_path):
        """Test available views for generic adapter."""
        adapter = GenericVideoAdapter(data_dir=tmp_path)

        views = adapter.get_available_views()

        assert views == ["default"]


# =============================================================================
# EGO-EXO4D ADAPTER TESTS
# =============================================================================

class TestEgoExo4DAdapter:
    """Tests for EgoExo4DAdapter."""

    def test_detect_empty_dir(self, tmp_path):
        """Test detection on empty directory."""
        assert EgoExo4DAdapter.detect(tmp_path) is False

    def test_detect_with_manifest(self, tmp_path):
        """Test detection with takes.json manifest."""
        (tmp_path / "takes.json").write_text("[]")

        assert EgoExo4DAdapter.detect(tmp_path) is True

    def test_detect_with_takes_dir(self, tmp_path):
        """Test detection with takes directory structure."""
        takes_dir = tmp_path / "takes" / "take_001" / "frame_aligned_videos" / "ego"
        takes_dir.mkdir(parents=True)

        assert EgoExo4DAdapter.detect(tmp_path) is True

    def test_get_available_views(self, tmp_path):
        """Test available views for Ego-Exo4D."""
        adapter = EgoExo4DAdapter(data_dir=tmp_path)

        views = adapter.get_available_views()

        assert "ego" in views
        assert "exo" in views


# =============================================================================
# VIDEO AUDIO MOCK DATASET TESTS
# =============================================================================

class TestVideoAudioMockDataset:
    """Tests for VideoAudioMockDataset."""

    def test_mock_dataset_length(self):
        """Test mock dataset has correct length."""
        dataset = VideoAudioMockDataset(num_samples=50)

        assert len(dataset) == 50

    def test_mock_dataset_sample_shape(self):
        """Test mock dataset sample shapes."""
        dataset = VideoAudioMockDataset(
            num_samples=10,
            clip_duration=2.0,
            fps=8,
            audio_sample_rate=16000,
            frame_size=(224, 224),
        )

        sample = dataset[0]

        # Vision: T * H * W * 3 = 2*8 * 224 * 224 * 3 = 2,408,448
        expected_vision_len = 2 * 8 * 224 * 224 * 3
        assert sample["vision"].shape == (expected_vision_len,)

        # Audio: duration * sample_rate = 2 * 16000 = 32000
        expected_audio_len = 2 * 16000
        assert sample["audio"].shape == (expected_audio_len,)

    def test_mock_dataset_masks(self):
        """Test mock dataset masks are all True."""
        dataset = VideoAudioMockDataset(num_samples=10)

        sample = dataset[0]

        assert sample["masks"]["vision"].all()
        assert sample["masks"]["audio"].all()

    def test_mock_dataset_multi_view(self):
        """Test mock dataset with multiple views."""
        dataset = VideoAudioMockDataset(
            num_samples=10,
            views=["ego", "exo"],
        )

        sample = dataset[0]

        assert "vision_ego" in sample
        assert "vision_exo" in sample
        assert "vision" not in sample

    def test_mock_dataset_multi_channel_audio(self):
        """Test mock dataset with multi-channel audio."""
        dataset = VideoAudioMockDataset(
            num_samples=10,
            clip_duration=2.0,
            audio_sample_rate=16000,
            num_audio_channels=7,  # Ego-Exo4D has 7 channels
        )

        sample = dataset[0]

        # Audio: duration * sample_rate * channels = 2 * 16000 * 7 = 224000
        expected_audio_len = 2 * 16000 * 7
        assert sample["audio"].shape == (expected_audio_len,)

    def test_mock_dataset_byte_values(self):
        """Test mock dataset values are in byte range."""
        dataset = VideoAudioMockDataset(num_samples=10)

        sample = dataset[0]

        assert sample["vision"].min() >= 0
        assert sample["vision"].max() <= 255
        assert sample["audio"].min() >= 0
        assert sample["audio"].max() <= 255

    def test_mock_dataset_dtype(self):
        """Test mock dataset tensors have correct dtype."""
        dataset = VideoAudioMockDataset(num_samples=10)

        sample = dataset[0]

        assert sample["vision"].dtype == torch.long
        assert sample["audio"].dtype == torch.long


# =============================================================================
# VIDEO AUDIO DATASET TESTS (WITH MOCK ADAPTER)
# =============================================================================

class TestVideoAudioDataset:
    """Tests for VideoAudioDataset."""

    def test_unknown_format_raises(self, tmp_path):
        """Test that unknown format raises error."""
        with pytest.raises(ValueError, match="Unknown format"):
            VideoAudioDataset(tmp_path, format="nonexistent_format")

    def test_auto_format_empty_dir(self, tmp_path):
        """Test auto format on empty directory."""
        # Should fall back to generic, which returns no clips
        dataset = VideoAudioDataset(tmp_path, format="generic")

        assert len(dataset) == 0

    def test_sequence_lengths(self, tmp_path):
        """Test expected sequence lengths are calculated correctly."""
        dataset = VideoAudioDataset(
            tmp_path,
            format="generic",
            clip_duration=2.0,
            fps=8,
            audio_sample_rate=16000,
            frame_size=(224, 224),
        )

        # Vision: T * H * W * 3
        assert dataset.vision_seq_len == 16 * 224 * 224 * 3

        # Audio: duration * sample_rate
        assert dataset.audio_seq_len == 32000


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestVideoDatasetIntegration:
    """Integration tests for video dataset with training pipeline."""

    def test_mock_dataset_with_dataloader(self):
        """Test mock dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        dataset = VideoAudioMockDataset(num_samples=20)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        batch = next(iter(loader))

        assert batch["vision"].shape[0] == 4
        assert batch["audio"].shape[0] == 4

    def test_mock_dataset_output_format(self):
        """Test mock dataset output matches O-JEPA expected format."""
        dataset = VideoAudioMockDataset(num_samples=10)
        sample = dataset[0]

        # Should have modality keys
        assert "vision" in sample or "vision_ego" in sample
        assert "audio" in sample

        # Should have masks dict
        assert "masks" in sample
        assert isinstance(sample["masks"], dict)

    def test_multi_view_separate_keys(self):
        """Test multi-view returns separate keys."""
        dataset = VideoAudioMockDataset(
            num_samples=10,
            views=["ego", "exo"],
        )

        sample = dataset[0]

        # Should have separate keys, not stacked
        assert "vision_ego" in sample
        assert "vision_exo" in sample
        assert sample["vision_ego"].shape == sample["vision_exo"].shape
