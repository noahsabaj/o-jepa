"""LUMA Dataset for ByteJEPA.

Provides dataset loaders for the LUMA multimodal dataset.
LUMA contains image, audio, and text modalities aligned by class label.

Dataset: bezirganyan/LUMA on HuggingFace
Paper: https://arxiv.org/abs/2406.09864

Note on splits:
    - "train": 80% of classes (for training)
    - "test": 20% of classes (use as validation during training)
    The split is by class label, not by sample, to ensure proper evaluation
    on unseen classes.
"""

from typing import Dict, List, Iterator
import logging
import torch
from torch.utils.data import IterableDataset
import numpy as np

# Set up logging for data loading issues
logger = logging.getLogger(__name__)

# Optional imports for image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Optional imports for audio processing
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


class LUMALocalDataset(IterableDataset):
    """
    Local LUMA dataset loader (from downloaded files).

    Loads from locally cloned LUMA repository.
    Expected structure:
        data_dir/
            edm_images.pickle    # Images DataFrame
            text_data.tsv        # Text TSV
            audio/
                datalist.csv     # Audio index
                cv_audio/        # Audio files
                ls_audio/
                sw_audio/

    Args:
        data_dir: Path to LUMA data directory
        split: "train" or "test" (80/20 split by default)
        modalities: List of modalities to load
        vision_size: Target image size (H, W)
        text_max_len: Maximum text length in bytes
        audio_max_len: Maximum audio length in bytes
        audio_sample_rate: Target sample rate for audio
        align_by_class: If True, yield samples aligned by class label
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        modalities: List[str] = None,
        vision_size: tuple = (32, 32),
        text_max_len: int = 1024,
        audio_max_len: int = 8000,
        audio_sample_rate: int = 16000,
        align_by_class: bool = True,
        train_ratio: float = 0.8,
    ):
        from pathlib import Path
        import pandas as pd
        import pickle

        self.data_dir = Path(data_dir)
        self.split = split
        self.modalities = modalities or ["vision", "text", "audio"]
        self.vision_size = vision_size
        self.text_max_len = text_max_len
        self.audio_max_len = audio_max_len
        self.audio_sample_rate = audio_sample_rate
        self.align_by_class = align_by_class
        self.train_ratio = train_ratio

        self.vision_seq_len = vision_size[0] * vision_size[1] * 3

        # Load data indices
        self._images_df = None
        self._text_df = None
        self._audio_df = None
        self._class_to_indices = {}

        self._load_indices()
        self._estimated_length = self._compute_length()

    def _compute_length(self) -> int:
        """Compute estimated number of samples based on class indices."""
        import random

        # Get all classes
        classes = list(self._class_to_indices.keys())

        # Determine split (same logic as __iter__)
        random.seed(42)
        shuffled = classes.copy()
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * self.train_ratio)

        if self.split == "train":
            active_classes = shuffled[:split_idx]
        else:
            active_classes = shuffled[split_idx:]

        # Count samples per class
        total = 0
        for label in active_classes:
            indices = self._class_to_indices.get(label, {})
            counts = []
            if "vision" in self.modalities and indices.get('vision'):
                counts.append(len(indices['vision']))
            if "text" in self.modalities and indices.get('text'):
                counts.append(len(indices['text']))
            if "audio" in self.modalities and indices.get('audio'):
                counts.append(len(indices['audio']))
            if counts:
                total += min(counts) if self.align_by_class else max(counts)

        return total

    def __len__(self) -> int:
        """Return estimated number of samples for progress tracking."""
        return self._estimated_length

    def _load_indices(self):
        """Load data indices from files."""
        import pandas as pd
        import pickle

        # Load images
        if "vision" in self.modalities:
            img_path = self.data_dir / "edm_images.pickle"
            if img_path.exists():
                self._images_df = pickle.load(open(img_path, 'rb'))
                # Build class index
                for idx, row in self._images_df.iterrows():
                    label = row['label']
                    if label not in self._class_to_indices:
                        self._class_to_indices[label] = {'vision': [], 'text': [], 'audio': []}
                    self._class_to_indices[label]['vision'].append(idx)

        # Load text
        if "text" in self.modalities:
            text_path = self.data_dir / "text_data.tsv"
            if text_path.exists():
                self._text_df = pd.read_csv(text_path, sep='\t')
                for idx, row in self._text_df.iterrows():
                    label = row['label']
                    if label not in self._class_to_indices:
                        self._class_to_indices[label] = {'vision': [], 'text': [], 'audio': []}
                    self._class_to_indices[label]['text'].append(idx)

        # Load audio index
        if "audio" in self.modalities:
            audio_path = self.data_dir / "audio" / "datalist.csv"
            if audio_path.exists():
                self._audio_df = pd.read_csv(audio_path)
                for idx, row in self._audio_df.iterrows():
                    label = row['label']
                    if label not in self._class_to_indices:
                        self._class_to_indices[label] = {'vision': [], 'text': [], 'audio': []}
                    self._class_to_indices[label]['audio'].append(idx)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over samples."""
        import random

        # Get all classes
        classes = list(self._class_to_indices.keys())

        # Determine split
        random.seed(42)  # Reproducible split
        random.shuffle(classes)
        split_idx = int(len(classes) * self.train_ratio)

        if self.split == "train":
            active_classes = classes[:split_idx]
        else:
            active_classes = classes[split_idx:]

        # Iterate by class for alignment
        for label in active_classes:
            indices = self._class_to_indices[label]

            # Get minimum count across modalities
            counts = []
            if "vision" in self.modalities and indices['vision']:
                counts.append(len(indices['vision']))
            if "text" in self.modalities and indices['text']:
                counts.append(len(indices['text']))
            if "audio" in self.modalities and indices['audio']:
                counts.append(len(indices['audio']))

            if not counts:
                continue

            num_samples = min(counts) if self.align_by_class else max(counts)

            for i in range(num_samples):
                result = {}
                masks = {}

                # Vision
                if "vision" in self.modalities and indices['vision']:
                    idx = indices['vision'][i % len(indices['vision'])]
                    bytes_tensor, mask = self._load_image(idx)
                    result["vision"] = bytes_tensor
                    masks["vision"] = mask

                # Text
                if "text" in self.modalities and indices['text']:
                    idx = indices['text'][i % len(indices['text'])]
                    bytes_tensor, mask = self._load_text(idx)
                    result["text"] = bytes_tensor
                    masks["text"] = mask

                # Audio
                if "audio" in self.modalities and indices['audio']:
                    idx = indices['audio'][i % len(indices['audio'])]
                    bytes_tensor, mask = self._load_audio(idx)
                    result["audio"] = bytes_tensor
                    masks["audio"] = mask

                result["masks"] = masks

                # Only yield if we have all requested modalities
                # (check for keys excluding "masks")
                result_modalities = set(result.keys()) - {"masks"}
                if result_modalities == set(self.modalities):
                    yield result

    def _load_image(self, idx: int) -> tuple:
        """Load image from pickle DataFrame."""
        row = self._images_df.iloc[idx]
        img_array = np.array(row['image'], dtype=np.uint8)

        # Resize if needed (LUMA is 32x32)
        if img_array.shape[:2] != self.vision_size:
            if PIL_AVAILABLE:
                from PIL import Image
                img = Image.fromarray(img_array)
                img = img.resize(self.vision_size)
                img_array = np.array(img, dtype=np.uint8)

        # Flatten to bytes
        bytes_tensor = torch.from_numpy(img_array.flatten()).long()
        mask = torch.ones(len(bytes_tensor), dtype=torch.bool)

        return bytes_tensor, mask

    def _load_text(self, idx: int) -> tuple:
        """Load text from TSV DataFrame."""
        row = self._text_df.iloc[idx]
        text = str(row['text'])

        text_bytes = text.encode("utf-8")[:self.text_max_len]
        actual_len = len(text_bytes)

        padded = np.zeros(self.text_max_len, dtype=np.uint8)
        padded[:actual_len] = list(text_bytes)
        bytes_tensor = torch.from_numpy(padded).long()

        mask = torch.zeros(self.text_max_len, dtype=torch.bool)
        mask[:actual_len] = True

        return bytes_tensor, mask

    def _load_audio(self, idx: int) -> tuple:
        """Load audio from wav file using soundfile."""
        row = self._audio_df.iloc[idx]
        audio_path = self.data_dir / "audio" / row['path']

        if not SOUNDFILE_AVAILABLE:
            logger.warning("soundfile not available - audio will be zeros. Install with: pip install soundfile")
            padded = torch.zeros(self.audio_max_len, dtype=torch.long)
            mask = torch.zeros(self.audio_max_len, dtype=torch.bool)
            return padded, mask

        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            padded = torch.zeros(self.audio_max_len, dtype=torch.long)
            mask = torch.zeros(self.audio_max_len, dtype=torch.bool)
            return padded, mask

        try:
            # Load with soundfile (works without torchcodec)
            data, sr = sf.read(audio_path)
            waveform = torch.tensor(data, dtype=torch.float32)

            # Convert to mono if stereo
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=-1)

            # Simple resampling if needed (linear interpolation)
            if sr != self.audio_sample_rate:
                ratio = self.audio_sample_rate / sr
                new_len = int(len(waveform) * ratio)
                waveform = torch.nn.functional.interpolate(
                    waveform.unsqueeze(0).unsqueeze(0),
                    size=new_len,
                    mode='linear',
                    align_corners=True
                ).squeeze()

            # Normalize and convert to 8-bit PCM
            max_val = waveform.abs().max()
            if max_val > 0:
                waveform = waveform / max_val

            pcm = ((waveform + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)

            # Truncate or pad
            actual_len = min(len(pcm), self.audio_max_len)
            padded = torch.zeros(self.audio_max_len, dtype=torch.long)
            padded[:actual_len] = pcm[:actual_len].long()

            mask = torch.zeros(self.audio_max_len, dtype=torch.bool)
            mask[:actual_len] = True

            return padded, mask

        except Exception as e:
            logger.warning(f"Error loading audio {audio_path}: {e}")
            padded = torch.zeros(self.audio_max_len, dtype=torch.long)
            mask = torch.zeros(self.audio_max_len, dtype=torch.bool)
            return padded, mask


class LUMAMockDataset(IterableDataset):
    """
    Mock LUMA dataset for testing without network access.

    Generates synthetic data matching LUMA structure.
    Useful for unit testing the data pipeline.

    Args:
        num_samples: Number of samples to generate
        modalities: List of modalities to include
        vision_size: Target image size (H, W)
        text_max_len: Maximum text length in bytes
        audio_max_len: Maximum audio length in bytes

    Example:
        >>> dataset = LUMAMockDataset(num_samples=10, modalities=["vision", "text"])
        >>> samples = list(dataset)
        >>> len(samples)
        10
    """

    def __init__(
        self,
        num_samples: int = 100,
        modalities: List[str] = None,
        vision_size: tuple = (32, 32),
        text_max_len: int = 1024,
        audio_max_len: int = 8000,
    ):
        self.num_samples = num_samples
        self.modalities = modalities or ["vision", "text", "audio"]
        self.vision_size = vision_size
        self.text_max_len = text_max_len
        self.audio_max_len = audio_max_len
        self.vision_seq_len = vision_size[0] * vision_size[1] * 3

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Generate synthetic samples."""
        for _ in range(self.num_samples):
            result = {}
            masks = {}

            if "vision" in self.modalities:
                result["vision"] = torch.randint(
                    0, 256, (self.vision_seq_len,), dtype=torch.long
                )
                masks["vision"] = torch.ones(self.vision_seq_len, dtype=torch.bool)

            if "audio" in self.modalities:
                # Variable length audio (between half max and max)
                min_audio = max(1, self.audio_max_len // 2)
                if min_audio >= self.audio_max_len:
                    audio_len = self.audio_max_len
                else:
                    audio_len = torch.randint(min_audio, self.audio_max_len + 1, (1,)).item()
                audio = torch.zeros(self.audio_max_len, dtype=torch.long)
                audio[:audio_len] = torch.randint(0, 256, (audio_len,))
                result["audio"] = audio
                mask = torch.zeros(self.audio_max_len, dtype=torch.bool)
                mask[:audio_len] = True
                masks["audio"] = mask

            if "text" in self.modalities:
                # Variable length text (between 10% and max)
                min_text = max(1, self.text_max_len // 10)
                if min_text >= self.text_max_len:
                    text_len = self.text_max_len
                else:
                    text_len = torch.randint(min_text, self.text_max_len + 1, (1,)).item()
                text = torch.zeros(self.text_max_len, dtype=torch.long)
                text[:text_len] = torch.randint(32, 127, (text_len,))
                result["text"] = text
                mask = torch.zeros(self.text_max_len, dtype=torch.bool)
                mask[:text_len] = True
                masks["text"] = mask

            result["masks"] = masks
            yield result

    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples
