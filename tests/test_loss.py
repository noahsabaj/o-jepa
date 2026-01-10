"""Tests for JEPA loss functions."""

import pytest
import torch

from src.loss import (
    JEPALoss,
    JEPALossConfig,
    SmoothL1Loss,
    MSELoss,
    compute_energy,
    compute_prediction_metrics,
)


class TestJEPALossConfig:
    """Tests for JEPALossConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = JEPALossConfig()
        assert config.loss_type == "smooth_l1"
        assert config.beta == 1.0
        assert config.use_variance_loss == True
        assert config.variance_weight == 0.04
        assert config.variance_target == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = JEPALossConfig(
            loss_type="mse",
            variance_weight=0.1,
            variance_target=2.0,
        )
        assert config.loss_type == "mse"
        assert config.variance_weight == 0.1
        assert config.variance_target == 2.0


class TestJEPALoss:
    """Tests for JEPALoss."""

    @pytest.fixture
    def batch_size(self):
        return 8

    @pytest.fixture
    def hidden_dim(self):
        return 256

    @pytest.fixture
    def num_targets(self):
        return 10

    @pytest.fixture
    def loss_fn(self):
        return JEPALoss(JEPALossConfig())

    def test_basic_loss(self, loss_fn, batch_size, num_targets, hidden_dim):
        """Test basic loss computation."""
        predictions = torch.randn(batch_size, num_targets, hidden_dim)
        targets = torch.randn(batch_size, num_targets, hidden_dim)

        loss, metrics = loss_fn(predictions, targets)

        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
        assert loss >= 0  # Non-negative
        assert "loss" in metrics
        assert "pred_loss" in metrics
        assert "var_loss" in metrics

    def test_identical_embeddings(self, loss_fn, batch_size, num_targets, hidden_dim):
        """Test loss with identical embeddings (should be near zero)."""
        embeddings = torch.randn(batch_size, num_targets, hidden_dim)

        loss, metrics = loss_fn(embeddings, embeddings.clone())

        # Loss should be very small for identical embeddings
        assert loss.item() < 0.1
        assert metrics["pred_loss"] < 0.01

    def test_targets_detached(self, loss_fn, batch_size, num_targets, hidden_dim):
        """Test that targets are detached (no gradients flow through)."""
        predictions = torch.randn(batch_size, num_targets, hidden_dim, requires_grad=True)
        targets = torch.randn(batch_size, num_targets, hidden_dim, requires_grad=True)

        loss, _ = loss_fn(predictions, targets)
        loss.backward()

        # Predictions should have gradients
        assert predictions.grad is not None

        # Targets should not have gradients (detached in loss_fn)
        assert targets.grad is None

    def test_variance_loss(self, batch_size, num_targets, hidden_dim):
        """Test variance loss component."""
        config = JEPALossConfig(use_variance_loss=True, variance_weight=1.0)
        loss_fn = JEPALoss(config)

        # Create predictions with low variance (should have high var_loss)
        predictions = torch.ones(batch_size, num_targets, hidden_dim) + torch.randn(batch_size, num_targets, hidden_dim) * 0.01
        targets = torch.randn(batch_size, num_targets, hidden_dim)

        loss, metrics = loss_fn(predictions, targets)

        assert metrics["var_loss"] > 0  # Should penalize low variance

    def test_variance_loss_disabled(self, batch_size, num_targets, hidden_dim):
        """Test with variance loss disabled."""
        config = JEPALossConfig(use_variance_loss=False)
        loss_fn = JEPALoss(config)

        predictions = torch.randn(batch_size, num_targets, hidden_dim)
        targets = torch.randn(batch_size, num_targets, hidden_dim)

        loss, metrics = loss_fn(predictions, targets)

        assert metrics["var_loss"] == 0.0

    def test_mse_loss_type(self, batch_size, num_targets, hidden_dim):
        """Test with MSE loss type."""
        config = JEPALossConfig(loss_type="mse")
        loss_fn = JEPALoss(config)

        predictions = torch.randn(batch_size, num_targets, hidden_dim)
        targets = torch.randn(batch_size, num_targets, hidden_dim)

        loss, metrics = loss_fn(predictions, targets)

        assert not torch.isnan(loss)
        assert loss >= 0

    def test_with_mask(self, loss_fn, batch_size, num_targets, hidden_dim):
        """Test loss with validity mask."""
        predictions = torch.randn(batch_size, num_targets, hidden_dim)
        targets = torch.randn(batch_size, num_targets, hidden_dim)

        # Create mask where only some predictions are valid
        mask = torch.zeros(batch_size, num_targets)
        mask[:, :num_targets // 2] = 1.0  # Only first half valid

        loss, metrics = loss_fn(predictions, targets, mask=mask)

        assert not torch.isnan(loss)

    def test_2d_input(self, loss_fn, batch_size, hidden_dim):
        """Test with 2D input (batch, dim) instead of 3D."""
        predictions = torch.randn(batch_size, hidden_dim)
        targets = torch.randn(batch_size, hidden_dim)

        loss, metrics = loss_fn(predictions, targets)

        assert not torch.isnan(loss)
        assert loss >= 0

    def test_redundancy_loss(self, batch_size, num_targets, hidden_dim):
        """Y2: Test VICReg-style redundancy loss."""
        config = JEPALossConfig(use_redundancy_loss=True, redundancy_weight=1.0)
        loss_fn = JEPALoss(config)

        # Create predictions with correlated features (should have high redundancy_loss)
        base = torch.randn(batch_size, num_targets, 1)
        predictions = base.expand(-1, -1, hidden_dim) + torch.randn(batch_size, num_targets, hidden_dim) * 0.1
        targets = torch.randn(batch_size, num_targets, hidden_dim)

        loss, metrics = loss_fn(predictions, targets)

        assert not torch.isnan(loss)
        assert "redundancy_loss" in metrics
        assert "off_diag_mean" in metrics
        # Correlated features should have non-zero redundancy loss
        assert metrics["redundancy_loss"] > 0

    def test_redundancy_loss_disabled(self, batch_size, num_targets, hidden_dim):
        """Test with redundancy loss disabled (default)."""
        config = JEPALossConfig(use_redundancy_loss=False)
        loss_fn = JEPALoss(config)

        predictions = torch.randn(batch_size, num_targets, hidden_dim)
        targets = torch.randn(batch_size, num_targets, hidden_dim)

        loss, metrics = loss_fn(predictions, targets)

        assert metrics["redundancy_loss"] == 0.0
        assert metrics["off_diag_mean"] == 0.0

    def test_redundancy_loss_independent_features(self, batch_size, num_targets, hidden_dim):
        """Y2: Independent features should have low redundancy loss."""
        config = JEPALossConfig(use_redundancy_loss=True, redundancy_weight=1.0)
        loss_fn = JEPALoss(config)

        # Create predictions with independent features (low correlation)
        predictions = torch.randn(batch_size, num_targets, hidden_dim)
        targets = torch.randn(batch_size, num_targets, hidden_dim)

        _, metrics = loss_fn(predictions, targets)

        # Independent features should have relatively low off-diagonal correlation
        # (not exactly zero due to random sampling, but smaller than correlated case)
        assert metrics["off_diag_mean"] < 0.5  # Reasonable threshold for uncorrelated features


class TestSmoothL1Loss:
    """Tests for SmoothL1Loss wrapper."""

    def test_forward(self):
        """Test forward pass."""
        loss_fn = SmoothL1Loss(beta=1.0)

        predictions = torch.randn(8, 256)
        targets = torch.randn(8, 256)

        loss, metrics = loss_fn(predictions, targets)

        assert loss.ndim == 0
        assert "loss" in metrics
        assert not torch.isnan(loss)


class TestMSELoss:
    """Tests for MSELoss wrapper."""

    def test_forward(self):
        """Test forward pass."""
        loss_fn = MSELoss()

        predictions = torch.randn(8, 256)
        targets = torch.randn(8, 256)

        loss, metrics = loss_fn(predictions, targets)

        assert loss.ndim == 0
        assert "loss" in metrics
        assert not torch.isnan(loss)


class TestComputeEnergy:
    """Tests for compute_energy function."""

    def test_basic_energy(self):
        """Test basic energy computation."""
        predictions = torch.randn(8, 256)
        targets = torch.randn(8, 256)

        energy = compute_energy(predictions, targets)

        assert energy.shape == (8,)
        assert not torch.isnan(energy).any()
        assert (energy >= 0).all()

    def test_identical_low_energy(self):
        """Test that identical inputs have low energy."""
        embeddings = torch.randn(8, 256)

        energy = compute_energy(embeddings, embeddings)

        assert energy.max() < 1e-6  # Should be near zero


class TestComputePredictionMetrics:
    """Tests for compute_prediction_metrics function."""

    def test_basic_metrics(self):
        """Test basic metric computation."""
        predictions = torch.randn(8, 10, 256)
        targets = torch.randn(8, 10, 256)

        metrics = compute_prediction_metrics(predictions, targets)

        assert "mse" in metrics
        assert "cosine_sim" in metrics
        assert "rel_error" in metrics
        assert "pred_std" in metrics
        assert "target_std" in metrics
        assert "std_ratio" in metrics

    def test_identical_embeddings(self):
        """Test metrics with identical embeddings."""
        embeddings = torch.randn(8, 10, 256)

        metrics = compute_prediction_metrics(embeddings, embeddings)

        assert metrics["mse"] < 1e-6
        assert abs(metrics["cosine_sim"] - 1.0) < 1e-5
        assert abs(metrics["std_ratio"] - 1.0) < 1e-5


class TestGradientFlow:
    """Tests for gradient flow through loss."""

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        config = JEPALossConfig()
        loss_fn = JEPALoss(config)

        predictions = torch.randn(4, 8, 128, requires_grad=True)
        targets = torch.randn(4, 8, 128)

        loss, _ = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None
        assert not torch.isnan(predictions.grad).any()

    def test_no_gradient_through_targets(self):
        """Test that no gradients flow through targets."""
        config = JEPALossConfig()
        loss_fn = JEPALoss(config)

        predictions = torch.randn(4, 8, 128, requires_grad=True)
        targets = torch.randn(4, 8, 128, requires_grad=True)

        loss, _ = loss_fn(predictions, targets)
        loss.backward()

        # Targets should have no gradient (detached in forward)
        assert targets.grad is None
