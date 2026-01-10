"""Tests for JEPA Training Utilities."""

import pytest
import torch
from torch.amp import GradScaler

from src.train import (
    WSDScheduler,
    TrainingState,
    train_step_jepa,
    evaluate,
    create_optimizer,
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

    def test_state_dict(self):
        """Test state_dict returns scheduler state."""
        optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.randn(10))], lr=1e-4
        )
        scheduler = WSDScheduler(
            optimizer=optimizer,
            base_lr=1e-4,
            warmup_steps=100,
            total_steps=1000,
        )

        # Take some steps
        for _ in range(50):
            scheduler.step()

        state = scheduler.state_dict()
        assert 'current_step' in state
        assert state['current_step'] == 50

    def test_load_state_dict(self):
        """Test load_state_dict restores scheduler state."""
        optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.randn(10))], lr=1e-4
        )
        scheduler = WSDScheduler(
            optimizer=optimizer,
            base_lr=1e-4,
            warmup_steps=100,
            total_steps=1000,
        )

        # Take some steps
        for _ in range(50):
            scheduler.step()

        # Save state
        state = scheduler.state_dict()

        # Take more steps
        for _ in range(50):
            scheduler.step()
        assert scheduler.current_step == 100

        # Restore state
        scheduler.load_state_dict(state)
        assert scheduler.current_step == 50

    def test_with_lr_scale(self):
        """Test scheduler respects lr_scale in param groups."""
        base_lr = 1e-4
        params1 = [torch.nn.Parameter(torch.randn(10))]
        params2 = [torch.nn.Parameter(torch.randn(10))]

        # Create optimizer with lr_scale
        optimizer = torch.optim.AdamW([
            {'params': params1, 'lr': base_lr, 'lr_scale': 10.0},  # 10x higher
            {'params': params2, 'lr': base_lr, 'lr_scale': 1.0},  # normal
        ], lr=base_lr)

        scheduler = WSDScheduler(
            optimizer=optimizer,
            base_lr=base_lr,
            warmup_steps=100,
            total_steps=1000,
        )

        # Skip warmup to get to stable phase
        for _ in range(100):
            scheduler.step()

        # At stable phase, LRs should reflect lr_scale
        # Group 0 should have 10x the base lr
        # Group 1 should have base lr
        lr0 = optimizer.param_groups[0]['lr']
        lr1 = optimizer.param_groups[1]['lr']
        assert abs(lr0 / lr1 - 10.0) < 0.01, f"lr0={lr0}, lr1={lr1}, ratio={lr0/lr1}"


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


class TestMuonOptimizer:
    """Test Muon optimizer configuration and creation."""

    def test_muon_config_defaults(self):
        """Test Muon config has correct defaults."""
        config = TrainingConfig()
        assert config.use_muon == True
        assert config.muon_lr == 1.5e-2
        assert config.adamw_lr == 5e-4
        assert config.muon_momentum == 0.95
        assert config.muon_nesterov == True
        assert config.muon_ns_steps == 5
        assert config.muon_weight_decay == 0.0
        assert config.min_tokens_per_batch == 65536

    def test_create_optimizer_with_config(self, tiny_config):
        """Test optimizer creation with full TrainingConfig."""
        model = JEPAWorldModel(tiny_config)
        training_config = TrainingConfig(
            use_muon=True,
            muon_lr=1e-2,
            adamw_lr=3e-4,
        )
        optimizer = create_optimizer(model, config=training_config)
        assert optimizer is not None

    def test_create_optimizer_legacy_mode(self, tiny_config):
        """Test optimizer creation with legacy params (backward compat)."""
        model = JEPAWorldModel(tiny_config)
        optimizer = create_optimizer(
            model,
            learning_rate=1e-4,
            weight_decay=0.05,
        )
        assert optimizer is not None

    def test_adamw_fallback(self, tiny_config):
        """Test AdamW fallback when Muon disabled."""
        model = JEPAWorldModel(tiny_config)
        training_config = TrainingConfig(use_muon=False)
        optimizer = create_optimizer(model, config=training_config)
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_lr_scale_in_param_groups(self, tiny_config):
        """Test that lr_scale is set in param groups for scheduler."""
        model = JEPAWorldModel(tiny_config)
        training_config = TrainingConfig(
            learning_rate=1e-4,
            muon_lr=1.5e-2,
            adamw_lr=5e-4,
        )
        optimizer = create_optimizer(model, config=training_config)

        # All param groups should have lr_scale
        for group in optimizer.param_groups:
            assert 'lr_scale' in group

    def test_separate_learning_rates(self, tiny_config):
        """Test that Muon and AdamW groups have different LRs."""
        model = JEPAWorldModel(tiny_config)
        training_config = TrainingConfig(
            use_muon=True,
            muon_lr=1.5e-2,
            adamw_lr=5e-4,
        )
        optimizer = create_optimizer(model, config=training_config)

        # Find groups by name
        muon_groups = [g for g in optimizer.param_groups if g.get('name') == 'muon_weights']
        adamw_groups = [g for g in optimizer.param_groups if g.get('name') == 'adamw_params']

        if muon_groups and adamw_groups:
            # Muon group should have higher LR
            assert muon_groups[0]['lr'] > adamw_groups[0]['lr']

    def test_muon_weight_decay_separate(self, tiny_config):
        """Test that Muon and AdamW can have separate weight decays."""
        model = JEPAWorldModel(tiny_config)
        training_config = TrainingConfig(
            use_muon=True,
            muon_weight_decay=0.0,  # No weight decay for Muon
            weight_decay=0.05,  # Weight decay for AdamW
        )
        optimizer = create_optimizer(model, config=training_config)

        # Find groups by name
        muon_groups = [g for g in optimizer.param_groups if g.get('name') == 'muon_weights']
        adamw_groups = [g for g in optimizer.param_groups if g.get('name') == 'adamw_params']

        if muon_groups:
            assert muon_groups[0]['weight_decay'] == 0.0
        if adamw_groups:
            assert adamw_groups[0]['weight_decay'] == 0.05

    def test_param_grouping_excludes_embeddings(self, tiny_config):
        """Test that embeddings are not in Muon group."""
        model = JEPAWorldModel(tiny_config)
        training_config = TrainingConfig(use_muon=True)
        optimizer = create_optimizer(model, config=training_config)

        muon_groups = [g for g in optimizer.param_groups if g.get('use_muon') == True]
        adamw_groups = [g for g in optimizer.param_groups if g.get('use_muon') == False]

        # Get actual parameter ids
        muon_param_ids = set()
        for g in muon_groups:
            for p in g['params']:
                muon_param_ids.add(id(p))

        # Check that byte_embed is not in Muon params
        for name, param in model.named_parameters():
            if 'embed' in name.lower() and param.requires_grad:
                assert id(param) not in muon_param_ids, f"{name} should not be in Muon group"


class TestCombinedOptimizer:
    """Tests for CombinedOptimizer wrapper class."""

    @pytest.fixture
    def tiny_config(self):
        """Create tiny config for testing."""
        return get_tiny_config()

    @pytest.fixture
    def combined_optimizer(self, tiny_config):
        """Create a CombinedOptimizer for testing."""
        model = JEPAWorldModel(tiny_config)
        training_config = TrainingConfig(
            use_muon=True,
            muon_lr=1.5e-2,
            adamw_lr=5e-4,
        )
        optimizer = create_optimizer(model, config=training_config)
        return optimizer, model

    def test_combined_optimizer_type(self, combined_optimizer):
        """Test that optimizer is CombinedOptimizer when using native Muon."""
        optimizer, model = combined_optimizer
        from src.train import CombinedOptimizer
        assert isinstance(optimizer, CombinedOptimizer)

    def test_combined_optimizer_zero_grad(self, combined_optimizer):
        """Test zero_grad clears gradients in both optimizers."""
        optimizer, model = combined_optimizer
        byte_ids = torch.randint(0, 256, (2, 64), dtype=torch.long)

        # Forward pass and backward
        model.train()
        loss, _ = model(byte_ids, modality='text')
        loss.backward()

        # Check some params have gradients
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad

        # Zero gradients
        optimizer.zero_grad()

        # Check all gradients are None or zero
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is None or torch.all(p.grad == 0)

    def test_combined_optimizer_step(self, combined_optimizer):
        """Test step updates parameters in both optimizers."""
        optimizer, model = combined_optimizer
        byte_ids = torch.randint(0, 256, (2, 64), dtype=torch.long)

        # Store initial weights
        initial_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                initial_weights[name] = param.data.clone()

        # Forward pass, backward, and step
        model.train()
        loss, _ = model(byte_ids, modality='text')
        loss.backward()
        optimizer.step()

        # Check that at least some weights changed
        weights_changed = False
        for name, param in model.named_parameters():
            if param.requires_grad and name in initial_weights:
                if not torch.allclose(param.data, initial_weights[name]):
                    weights_changed = True
                    break
        assert weights_changed, "Optimizer step should update weights"

    def test_combined_optimizer_state_dict(self, combined_optimizer):
        """Test state_dict returns both optimizer states."""
        optimizer, model = combined_optimizer
        byte_ids = torch.randint(0, 256, (2, 64), dtype=torch.long)

        # Do a step to populate state
        model.train()
        loss, _ = model(byte_ids, modality='text')
        loss.backward()
        optimizer.step()

        # Get state dict
        state_dict = optimizer.state_dict()
        assert 'muon' in state_dict
        assert 'adamw' in state_dict
        assert isinstance(state_dict['muon'], dict)
        assert isinstance(state_dict['adamw'], dict)

    def test_combined_optimizer_load_state_dict(self, combined_optimizer):
        """Test load_state_dict restores both optimizer states."""
        optimizer, model = combined_optimizer
        byte_ids = torch.randint(0, 256, (2, 64), dtype=torch.long)

        # Do a step to populate state
        model.train()
        loss, _ = model(byte_ids, modality='text')
        loss.backward()
        optimizer.step()

        # Save state
        state_dict = optimizer.state_dict()

        # Do another step
        optimizer.zero_grad()
        loss, _ = model(byte_ids, modality='text')
        loss.backward()
        optimizer.step()

        # Load original state
        optimizer.load_state_dict(state_dict)

        # Verify state was restored
        new_state = optimizer.state_dict()
        assert 'muon' in new_state
        assert 'adamw' in new_state

    def test_combined_optimizer_param_groups(self, combined_optimizer):
        """Test param_groups returns combined groups."""
        optimizer, model = combined_optimizer
        param_groups = optimizer.param_groups

        assert len(param_groups) >= 2
        # Check optimizer type annotations
        optimizer_types = [g.get('optimizer') for g in param_groups]
        assert 'muon' in optimizer_types
        assert 'adamw' in optimizer_types

    def test_combined_optimizer_state_property(self, combined_optimizer):
        """Test state property returns combined state."""
        optimizer, model = combined_optimizer
        byte_ids = torch.randint(0, 256, (2, 64), dtype=torch.long)

        # Do a step to populate state
        model.train()
        loss, _ = model(byte_ids, modality='text')
        loss.backward()
        optimizer.step()

        # Access state property
        state = optimizer.state
        assert isinstance(state, dict)

    def test_combined_optimizer_step_with_closure(self, combined_optimizer):
        """Test step with closure argument."""
        optimizer, model = combined_optimizer
        byte_ids = torch.randint(0, 256, (2, 64), dtype=torch.long)

        def closure():
            model.train()
            loss, _ = model(byte_ids, modality='text')
            loss.backward()
            return loss

        # Step with closure
        loss = optimizer.step(closure=closure)
        assert loss is not None
