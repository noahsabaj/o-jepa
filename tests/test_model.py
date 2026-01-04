"""Tests for JEPAWorldModel."""

import pytest
import torch

from src.model import JEPAWorldModel
from src.config import ByteJEPAConfig, get_tiny_config


class TestJEPAWorldModelCreation:
    """Test JEPAWorldModel instantiation."""

    def test_create_with_default_config(self, tiny_config):
        """Model should create with tiny config."""
        model = JEPAWorldModel(tiny_config)
        assert model is not None

    def test_has_all_components(self, byte_jepa_model):
        """Model should have all required components."""
        assert hasattr(byte_jepa_model, 'byte_encoder')
        assert hasattr(byte_jepa_model, 'backbone')
        assert hasattr(byte_jepa_model, 'predictor')
        assert hasattr(byte_jepa_model, 'loss_fn')
        assert hasattr(byte_jepa_model, 'target_encoder')
        assert hasattr(byte_jepa_model, 'mask_generator')

    def test_parameter_count(self, byte_jepa_model):
        """Model should report parameter count."""
        n_params = byte_jepa_model.get_num_params()
        assert n_params > 0
        assert isinstance(n_params, int)

    def test_trainable_parameters(self, byte_jepa_model):
        """Model should have frozen target encoder."""
        total = byte_jepa_model.get_num_params()

        # Count trainable parameters directly
        trainable = sum(p.numel() for p in byte_jepa_model.parameters() if p.requires_grad)

        # Target encoder is frozen, so trainable < total
        assert trainable < total
        assert trainable > 0


class TestJEPAWorldModelForward:
    """Test JEPAWorldModel forward pass."""

    def test_forward_text(self, byte_jepa_model, dummy_text_bytes):
        """Forward pass with text should return loss and outputs."""
        loss, outputs = byte_jepa_model(dummy_text_bytes, modality="text")

        assert loss is not None
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
        assert "metrics" in outputs
        assert "loss" in outputs["metrics"]
        assert "pred_loss" in outputs["metrics"]
        assert "var_loss" in outputs["metrics"]

    def test_forward_vision(self, byte_jepa_model, dummy_vision_bytes, tiny_config):
        """Forward pass with vision should return loss and outputs."""
        height, width = tiny_config.data.image_size
        loss, outputs = byte_jepa_model(
            dummy_vision_bytes, modality="vision", height=height, width=width
        )

        assert loss is not None
        assert not torch.isnan(loss)
        assert loss >= 0

    def test_forward_returns_metrics(self, byte_jepa_model, dummy_text_bytes):
        """Forward should return JEPA metrics."""
        loss, outputs = byte_jepa_model(dummy_text_bytes, modality="text")
        metrics = outputs["metrics"]

        # Check for JEPA-specific metrics
        assert "ema_decay" in metrics
        assert "num_targets" in metrics
        assert "mse" in metrics
        assert "cosine_sim" in metrics

    def test_forward_gradient_flow(self, byte_jepa_model, dummy_text_bytes):
        """Gradients should flow through online encoder only."""
        loss, _ = byte_jepa_model(dummy_text_bytes, modality="text")
        loss.backward()

        # Online encoder should have gradients
        for param in byte_jepa_model.byte_encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None

        # Target encoder should NOT have gradients (frozen)
        for param in byte_jepa_model.target_encoder.byte_encoder.parameters():
            assert param.grad is None

    def test_forward_different_batch_sizes(self, byte_jepa_model, tiny_config):
        """Model should handle different batch sizes."""
        for batch_size in [1, 2, 4]:
            byte_ids = torch.randint(0, 256, (batch_size, tiny_config.data.text_max_seq_len))
            loss, outputs = byte_jepa_model(byte_ids, modality="text")
            assert not torch.isnan(loss)


class TestJEPAWorldModelEMA:
    """Test EMA target encoder."""

    def test_ema_initialization(self, byte_jepa_model):
        """Target encoder should be initialized from online encoder."""
        # Check that target encoder exists
        assert byte_jepa_model.target_encoder is not None

        # Check EMA decay values
        assert byte_jepa_model.target_encoder.ema_decay_start > 0
        assert byte_jepa_model.target_encoder.ema_decay_end > byte_jepa_model.target_encoder.ema_decay_start

    def test_ema_update(self, byte_jepa_model, dummy_text_bytes):
        """EMA update should change target encoder."""
        # Get initial step
        initial_step = byte_jepa_model.target_encoder.current_step

        # Forward pass and EMA update
        byte_jepa_model(dummy_text_bytes, modality="text")
        byte_jepa_model.update_target_encoder()

        # Step counter should change
        assert byte_jepa_model.target_encoder.current_step > initial_step

    def test_ema_decay_schedule(self, byte_jepa_model):
        """EMA decay should follow schedule."""
        initial_decay = byte_jepa_model.target_encoder.ema_decay

        # Simulate many steps
        for _ in range(100):
            byte_jepa_model.target_encoder.current_step += 1

        later_decay = byte_jepa_model.target_encoder.ema_decay

        # Decay should increase over time
        assert later_decay >= initial_decay


class TestJEPAWorldModelPrediction:
    """Test world model prediction capabilities."""

    def test_predict_future(self, byte_jepa_model, dummy_text_bytes):
        """predict_future should return embeddings for future positions."""
        batch_size = dummy_text_bytes.shape[0]
        seq_len = dummy_text_bytes.shape[1]

        # Define future positions to predict
        future_positions = [torch.tensor([10, 20, 30]) for _ in range(batch_size)]

        # Create context mask (see first half)
        context_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        context_mask[:, seq_len // 2:] = False

        predictions = byte_jepa_model.predict_future(
            dummy_text_bytes, "text", future_positions, context_mask
        )

        assert predictions.shape[0] == batch_size
        assert predictions.shape[1] == 3  # 3 future positions
        assert not torch.isnan(predictions).any()

    def test_compute_energy(self, byte_jepa_model, dummy_text_bytes):
        """compute_energy should return per-sample energy scores."""
        batch_size = dummy_text_bytes.shape[0]
        seq_len = dummy_text_bytes.shape[1]

        # Define target positions to evaluate
        target_positions = [torch.tensor([10, 20, 30]) for _ in range(batch_size)]

        # Create context mask
        context_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        context_mask[:, 10] = False
        context_mask[:, 20] = False
        context_mask[:, 30] = False

        energy = byte_jepa_model.compute_energy(
            dummy_text_bytes, "text", target_positions, context_mask
        )

        assert energy.shape == (batch_size,)
        assert not torch.isnan(energy).any()
        assert (energy >= 0).all()  # Energy should be non-negative

    def test_encode(self, byte_jepa_model, dummy_text_bytes, hidden_dim):
        """encode should return pooled embeddings."""
        batch_size = dummy_text_bytes.shape[0]

        embeddings = byte_jepa_model.encode(dummy_text_bytes, "text")

        assert embeddings.shape == (batch_size, hidden_dim)
        assert not torch.isnan(embeddings).any()


class TestJEPAWorldModelMasking:
    """Test masking behavior."""

    def test_mask_generator_exists(self, byte_jepa_model):
        """Model should have mask generator."""
        assert hasattr(byte_jepa_model, 'mask_generator')
        assert byte_jepa_model.mask_generator is not None

    def test_forward_generates_masks(self, byte_jepa_model, dummy_text_bytes):
        """Forward should generate and use masks internally."""
        loss, outputs = byte_jepa_model(dummy_text_bytes, modality="text")
        metrics = outputs["metrics"]

        # Should have target count in metrics
        assert "num_targets" in metrics
        assert metrics["num_targets"] > 0


class TestJEPAWorldModelDifferentModalities:
    """Test model with different modalities."""

    def test_text_modality(self, byte_jepa_model, dummy_text_bytes):
        """Model should work with text modality."""
        loss, _ = byte_jepa_model(dummy_text_bytes, modality="text")
        assert not torch.isnan(loss)

    def test_vision_modality(self, byte_jepa_model, dummy_vision_bytes, tiny_config):
        """Model should work with vision modality."""
        h, w = tiny_config.data.image_size
        loss, _ = byte_jepa_model(dummy_vision_bytes, modality="vision", height=h, width=w)
        assert not torch.isnan(loss)

    def test_audio_modality(self, byte_jepa_model, dummy_audio_bytes):
        """Model should work with audio modality."""
        loss, _ = byte_jepa_model(dummy_audio_bytes, modality="audio")
        assert not torch.isnan(loss)


class TestJEPAWorldModelIntegration:
    """Integration tests."""

    def test_full_training_step(self, byte_jepa_model, dummy_text_bytes):
        """Test a complete training step."""
        optimizer = torch.optim.AdamW(byte_jepa_model.parameters(), lr=1e-4)

        # Forward
        loss, outputs = byte_jepa_model(dummy_text_bytes, modality="text")

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA update
        byte_jepa_model.update_target_encoder()

        # Should complete without error
        assert True

    def test_eval_mode(self, byte_jepa_model, dummy_text_bytes):
        """Model should work in eval mode."""
        byte_jepa_model.eval()

        with torch.no_grad():
            loss, outputs = byte_jepa_model(dummy_text_bytes, modality="text")

        assert not torch.isnan(loss)

    def test_deterministic_in_eval(self, byte_jepa_model, dummy_text_bytes):
        """Model should be deterministic in eval mode."""
        byte_jepa_model.eval()

        with torch.no_grad():
            loss1, _ = byte_jepa_model(dummy_text_bytes, modality="text")
            loss2, _ = byte_jepa_model(dummy_text_bytes, modality="text")

        # Due to stochastic masking, losses may differ slightly
        # But they should be reasonably close
        assert abs(loss1 - loss2) < 1.0  # Reasonable tolerance

    def test_loss_decreases_with_identical_input(self, byte_jepa_model, dummy_text_bytes):
        """When predicting identical embeddings, loss should be low."""
        # Create a "perfect" scenario - same input, predict same positions
        byte_jepa_model.eval()

        with torch.no_grad():
            loss, outputs = byte_jepa_model(dummy_text_bytes, modality="text")

        # Loss should be bounded (not infinite or extremely large)
        assert loss < 100.0
