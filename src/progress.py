"""Simple training progress reporter. That's it."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple


@dataclass
class ModelInfo:
    """Model info for training."""
    name: str
    total_params: int
    trainable_params: int
    hidden_dim: int
    modalities: Tuple[str, ...]
    byte_encoder_layers: int
    backbone_layers: int
    predictor_layers: int


@dataclass
class TrainingInfo:
    """Training config for progress display."""
    total_steps: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    device: str
    mixed_precision: bool
    modality: str


class ProgressReporter:
    """Simple progress reporter. Prints to stdout."""

    def __init__(self):
        self.total_steps = 100000
        self.best_val_loss = float('inf')

    def on_train_start(self, model_info: ModelInfo, training_info: TrainingInfo) -> None:
        self.total_steps = training_info.total_steps
        print(f"\n{'='*60}")
        print(f"ByteJEPA Training")
        print(f"{'='*60}")
        print(f"Parameters: {model_info.total_params/1e6:.1f}M")
        print(f"Modality: {training_info.modality}")
        print(f"Steps: {training_info.total_steps:,} | Batch: {training_info.batch_size}")
        print(f"Device: {training_info.device} | AMP: {training_info.mixed_precision}")
        print(f"{'='*60}\n")

    def on_step(self, step: int, metrics: Dict[str, Any]) -> None:
        loss = metrics.get('loss', metrics.get('avg_loss', 0))
        lr = metrics.get('lr', 0)
        throughput = metrics.get('samples_per_sec', 0)
        print(f"Step {step:>6}/{self.total_steps} | Loss: {loss:.4f} | LR: {lr:.2e} | {throughput:.0f} samples/s")

    def on_eval(self, step: int, val_loss: float, metrics: Optional[Dict[str, Any]] = None) -> None:
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        marker = " *BEST*" if is_best else ""
        print(f"  Eval @ {step}: val_loss={val_loss:.4f}{marker}")

    def on_save(self, step: int, path: str, is_best: bool = False) -> None:
        marker = " (best)" if is_best else ""
        print(f"  Saved checkpoint @ step {step}{marker}")

    def on_train_end(self, final_step: int, final_metrics: Dict[str, Any]) -> None:
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Final step: {final_step:,}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")

    def on_error(self, error: Exception, step: int) -> None:
        print(f"\nError at step {step}: {error}")


# Alias for compatibility
SimpleReporter = ProgressReporter


def create_reporter(name: str = "simple", **kwargs) -> ProgressReporter:
    """Create a progress reporter. Always returns SimpleReporter."""
    return ProgressReporter()
