"""Tests for JEPA Training Utilities."""

import pytest
import torch
from torch.amp import GradScaler

from src.train import (
    WSDScheduler,
    TrainingState,
    train_step_jepa,
    evaluate,
)
from src.train import TrainingConfig as TrainConfig  # Different from config.py
from src.config import TrainingConfig, get_tiny_config
from src.model import JEPAWorldModel


class TestWSDScheduler:
    """Test Warmup-Stable-Decay learning rate scheduler."""

    def test_create_scheduler(self):
        """Should create WSD scheduler."""
        optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.randn(10))], lr=1e-4
        )
        scheduler = WSDScheduler(
            optimizer=optimizer,
            base_lr=1e-4,
            warmup_steps=100,
            total_steps=1000,
        )

        assert scheduler is not None

    def test_warmup_phase(self):
        """Learning rate should increase during warmup."""
        base_lr = 1e-4
        optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.randn(10))], lr=base_lr
        )
        scheduler = WSDScheduler(
            optimizer=optimizer,
            base_lr=base_lr,
            warmup_steps=100,
            total_steps=1000,
            warmup_ratio=0.1,
            stable_ratio=0.7,
            decay_ratio=0.2,
        )

        lrs = []
        for step in range(100):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])

        # Learning rate should increase
        assert lrs[-1] > lrs[0]
        # Should reach approximately base_lr at end of warmup
        assert abs(lrs[-1] - base_lr) < base_lr * 0.2

    def test_stable_phase(self):
        """Learning rate should be constant during stable phase."""
        base_lr = 1e-4
        optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.randn(10))], lr=base_lr
        )
        scheduler = WSDScheduler(
            optimizer=optimizer,
            base_lr=base_lr,
            warmup_steps=100,
            total_steps=1000,
            warmup_ratio=0.1,
            stable_ratio=0.7,
            decay_ratio=0.2,
        )

        # Skip warmup (first 100 steps)
        for _ in range(100):
            scheduler.step()

        # Check stable phase (next 700 steps)
        lrs = []
        for step in range(600):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])

        # All should be close to base lr
        assert all(abs(lr - base_lr) < 1e-6 for lr in lrs)

    def test_decay_phase(self):
        """Learning rate should decrease during decay."""
        base_lr = 1e-4
        optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.randn(10))], lr=base_lr
        )
        scheduler = WSDScheduler(
            optimizer=optimizer,
            base_lr=base_lr,
            warmup_steps=100,
            total_steps=1000,
            warmup_ratio=0.1,
            stable_ratio=0.7,
            decay_ratio=0.2,
        )

        # Skip to decay phase
        for _ in range(800):
            scheduler.step()

        # Check decay phase
        lrs = []
        for step in range(150):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])

        # Learning rate should decrease
        assert lrs[-1] < lrs[0]


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_config(self):
        """Should create with defaults."""
        config = TrainingConfig()
        assert config.learning_rate > 0
        assert config.total_steps > 0
        assert config.batch_size > 0

    def test_effective_batch_size(self):
        """Should compute effective batch size."""
        config = TrainingConfig(batch_size=4, gradient_accumulation_steps=4)
        assert config.effective_batch_size == 16

    def test_warmup_steps(self):
        """Should compute warmup steps from ratio."""
        config = TrainingConfig(total_steps=1000, warmup_ratio=0.1)
        assert config.warmup_steps == 100


class TestTrainConfig:
    """Test TrainConfig from train.py."""

    def test_default_config(self):
        """Should create with defaults."""
        config = TrainConfig()
        assert config.learning_rate > 0
        assert config.total_steps > 0
        assert config.gradient_accumulation_steps >= 1


class TestTrainingState:
    """Test TrainingState dataclass."""

    def test_default_state(self):
        """Should create with zero values."""
        state = TrainingState()
        assert state.step == 0
        assert state.epoch == 0
        assert state.total_samples == 0
        assert state.best_loss == float("inf")


class TestTrainStepJEPA:
    """Test JEPA training step."""

    @pytest.fixture
    def model(self, tiny_config):
        """Create tiny model."""
        return JEPAWorldModel(tiny_config)

    @pytest.fixture
    def optimizer(self, model):
        """Create optimizer."""
        return torch.optim.AdamW(model.parameters(), lr=1e-4)

    @pytest.fixture
    def scheduler(self, optimizer):
        """Create scheduler."""
        return WSDScheduler(
            optimizer=optimizer,
            base_lr=1e-4,
            warmup_steps=10,
            total_steps=100,
        )

    @pytest.fixture
    def scaler(self):
        """Create gradient scaler."""
        return GradScaler("cpu", enabled=False)

    @pytest.fixture
    def state(self):
        """Create training state."""
        return TrainingState()

    @pytest.fixture
    def training_config(self):
        """Create training config."""
        return TrainConfig(
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            use_mixed_precision=False,
        )

    @pytest.fixture
    def batch(self, tiny_config):
        """Create a batch for JEPA training."""
        batch_size = 2
        return {
            "bytes": torch.randint(0, 256, (batch_size, tiny_config.data.text_max_seq_len)),
            "modality": "text",
        }

    def test_returns_metrics(
        self, model, batch, optimizer, scheduler, scaler, training_config, state
    ):
        """Train step should return metrics dict."""
        metrics = train_step_jepa(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=training_config,
            state=state,
            accumulation_step=0,
            modality="text",
        )

        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "lr" in metrics
        assert "ema_decay" in metrics

    def test_updates_state(
        self, model, batch, optimizer, scheduler, scaler, training_config, state
    ):
        """Train step should update state on accumulation completion."""
        initial_step = state.step

        metrics = train_step_jepa(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=training_config,
            state=state,
            accumulation_step=0,  # Last accumulation step
            modality="text",
        )

        # Should have incremented step
        assert state.step == initial_step + 1

    def test_ema_update(
        self, model, batch, optimizer, scheduler, scaler, training_config, state
    ):
        """Train step should update EMA target encoder."""
        initial_step = model.target_encoder.current_step

        metrics = train_step_jepa(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=training_config,
            state=state,
            accumulation_step=0,
            modality="text",
        )

        # EMA should have been updated
        assert model.target_encoder.current_step > initial_step


class TestEvaluateJEPA:
    """Test JEPA evaluation (using direct model calls)."""

    @pytest.fixture
    def model(self, tiny_config):
        """Create tiny model."""
        return JEPAWorldModel(tiny_config)

    def test_eval_mode_forward(self, model, tiny_config):
        """Model should work in eval mode."""
        model.eval()
        byte_ids = torch.randint(0, 256, (2, tiny_config.data.text_max_seq_len))

        with torch.no_grad():
            loss, outputs = model(byte_ids, modality="text")

        assert not torch.isnan(loss)
        assert "metrics" in outputs

    def test_batch_evaluation(self, model, tiny_config):
        """Test evaluating multiple batches."""
        model.eval()
        losses = []

        with torch.no_grad():
            for _ in range(3):
                byte_ids = torch.randint(0, 256, (2, tiny_config.data.text_max_seq_len))
                loss, _ = model(byte_ids, modality="text")
                losses.append(loss.item())

        # All losses should be valid
        assert all(not torch.isnan(torch.tensor(l)) for l in losses)


class TestJEPATrainingIntegration:
    """Integration tests for JEPA training."""

    def test_jepa_training_loop(self, tiny_config):
        """Test a complete JEPA training loop."""
        from src.data import SyntheticByteDataset
        from torch.utils.data import DataLoader

        model = JEPAWorldModel(tiny_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = WSDScheduler(
            optimizer=optimizer,
            base_lr=1e-4,
            warmup_steps=5,
            total_steps=10,
        )
        scaler = GradScaler("cpu", enabled=False)
        state = TrainingState()
        config = TrainConfig(
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            use_mixed_precision=False,
        )

        dataset = SyntheticByteDataset(
            num_samples=4,
            modalities=["text"],
            text_seq_len=tiny_config.data.text_max_seq_len,
        )
        loader = DataLoader(dataset, batch_size=2)

        # Run a few training steps
        losses = []
        for batch_data in loader:
            batch = {
                "bytes": batch_data["text"],
                "modality": "text",
            }
            metrics = train_step_jepa(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                state=state,
                accumulation_step=0,
                modality="text",
            )
            losses.append(metrics["loss"])

        # Should have completed without error
        assert len(losses) > 0
        assert all(not torch.isnan(torch.tensor(l)) for l in losses)

    def test_vision_modality_training(self, tiny_config):
        """Test training with vision modality."""
        from src.data import SyntheticByteDataset
        from torch.utils.data import DataLoader

        model = JEPAWorldModel(tiny_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = WSDScheduler(
            optimizer=optimizer,
            base_lr=1e-4,
            warmup_steps=5,
            total_steps=10,
        )
        scaler = GradScaler("cpu", enabled=False)
        state = TrainingState()
        config = TrainConfig(
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            use_mixed_precision=False,
        )

        dataset = SyntheticByteDataset(
            num_samples=4,
            modalities=["vision"],
            vision_seq_len=tiny_config.data.vision_seq_len,
        )
        loader = DataLoader(dataset, batch_size=2)

        batch_data = next(iter(loader))
        batch = {
            "bytes": batch_data["vision"],
            "modality": "vision",
            "height": tiny_config.data.image_size[0],
            "width": tiny_config.data.image_size[1],
        }

        metrics = train_step_jepa(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            state=state,
            accumulation_step=0,
            modality="vision",
        )

        assert "loss" in metrics
        assert not torch.isnan(torch.tensor(metrics["loss"]))


class TestGradientAccumulation:
    """Test gradient accumulation in training."""

    def test_accumulation_steps(self, tiny_config):
        """Test that gradient accumulation works correctly."""
        model = JEPAWorldModel(tiny_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = WSDScheduler(
            optimizer=optimizer,
            base_lr=1e-4,
            warmup_steps=5,
            total_steps=10,
        )
        scaler = GradScaler("cpu", enabled=False)
        state = TrainingState()
        config = TrainConfig(
            gradient_accumulation_steps=2,
            max_grad_norm=1.0,
            use_mixed_precision=False,
        )

        batch = {
            "bytes": torch.randint(0, 256, (2, tiny_config.data.text_max_seq_len)),
            "modality": "text",
        }

        # First accumulation step (shouldn't update) - accumulation_step=0
        # With gradient_accumulation_steps=2: (0+1)=1 != 2 -> no update
        initial_step = state.step
        train_step_jepa(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            state=state,
            accumulation_step=0,  # First step (not last)
            modality="text",
        )
        assert state.step == initial_step  # Should not have updated

        # Second accumulation step (should update) - accumulation_step=1
        # With gradient_accumulation_steps=2: (1+1)=2 == 2 -> update
        train_step_jepa(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            state=state,
            accumulation_step=1,  # Last step
            modality="text",
        )
        assert state.step == initial_step + 1  # Should have updated
