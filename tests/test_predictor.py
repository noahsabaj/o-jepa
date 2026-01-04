"""Tests for JEPAPredictor and JEPAPredictorBlock."""

import pytest
import torch

from src.predictor import JEPAPredictor, JEPAPredictorBlock
from src.config import PredictorConfig


class TestPredictorConfig:
    """Tests for PredictorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PredictorConfig()
        assert config.hidden_dim == 512
        assert config.output_dim == 512
        assert config.num_layers == 4
        assert config.num_heads == 8

    def test_custom_config(self):
        """Test custom configuration."""
        config = PredictorConfig(
            hidden_dim=256,
            output_dim=128,
            num_layers=2,
        )
        assert config.hidden_dim == 256
        assert config.output_dim == 128
        assert config.num_layers == 2

    def test_predictor_dim(self):
        """Test predictor_dim property (narrow by design)."""
        config = PredictorConfig(hidden_dim=512)
        assert config.predictor_dim == 256  # hidden_dim // 2


class TestJEPAPredictorBlock:
    """Tests for JEPAPredictorBlock."""

    @pytest.fixture
    def batch_size(self):
        return 4

    @pytest.fixture
    def hidden_dim(self):
        return 128

    def test_forward_shape(self, batch_size, hidden_dim):
        """Test output shape."""
        block = JEPAPredictorBlock(
            dim=hidden_dim,
            num_heads=4,
            context_dim=hidden_dim * 2,
        )

        queries = torch.randn(batch_size, 10, hidden_dim)
        context = torch.randn(batch_size, 64, hidden_dim * 2)

        out = block(queries, context)
        assert out.shape == queries.shape

    def test_with_context_mask(self, batch_size, hidden_dim):
        """Test with context attention mask."""
        block = JEPAPredictorBlock(
            dim=hidden_dim,
            num_heads=4,
            context_dim=hidden_dim * 2,
        )

        queries = torch.randn(batch_size, 10, hidden_dim)
        context = torch.randn(batch_size, 64, hidden_dim * 2)
        context_mask = torch.ones(batch_size, 64, dtype=torch.bool)
        context_mask[:, 32:] = False

        out = block(queries, context, context_mask)
        assert out.shape == queries.shape

    def test_gradient_flow(self, batch_size, hidden_dim):
        """Test gradient flow through block."""
        block = JEPAPredictorBlock(
            dim=hidden_dim,
            num_heads=4,
            context_dim=hidden_dim * 2,
        )

        queries = torch.randn(batch_size, 10, hidden_dim, requires_grad=True)
        context = torch.randn(batch_size, 64, hidden_dim * 2, requires_grad=True)

        out = block(queries, context)
        loss = out.sum()
        loss.backward()

        assert queries.grad is not None
        assert context.grad is not None

    def test_with_dropout(self, batch_size, hidden_dim):
        """Test with dropout enabled."""
        block = JEPAPredictorBlock(
            dim=hidden_dim,
            num_heads=4,
            context_dim=hidden_dim * 2,
            dropout=0.1,
        )
        block.train()

        queries = torch.randn(batch_size, 10, hidden_dim)
        context = torch.randn(batch_size, 64, hidden_dim * 2)

        out = block(queries, context)
        assert out.shape == queries.shape


class TestJEPAPredictor:
    """Tests for JEPAPredictor."""

    @pytest.fixture
    def batch_size(self):
        return 4

    @pytest.fixture
    def hidden_dim(self):
        return 256

    @pytest.fixture
    def predictor(self, hidden_dim):
        config = PredictorConfig(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            max_seq_len=1024,
        )
        return JEPAPredictor(config)

    def test_forward_shape(self, predictor, batch_size, hidden_dim):
        """Test output shape."""
        context = torch.randn(batch_size, 64, hidden_dim)
        target_positions = [
            torch.tensor([5, 10, 15, 20, 25]),
            torch.tensor([3, 8, 13, 18]),
            torch.tensor([0, 5, 10, 15, 20, 25, 30]),
            torch.tensor([12]),
        ]

        predictions, pred_mask = predictor(context, target_positions)

        max_targets = max(len(pos) for pos in target_positions)
        assert predictions.shape == (batch_size, max_targets, hidden_dim)
        assert pred_mask.shape == (batch_size, max_targets)

    def test_validity_mask(self, predictor, batch_size, hidden_dim):
        """Test that validity mask correctly identifies valid predictions."""
        context = torch.randn(batch_size, 64, hidden_dim)
        target_positions = [
            torch.tensor([5, 10]),  # 2 targets
            torch.tensor([3, 8, 13, 18]),  # 4 targets
            torch.tensor([0]),  # 1 target
            torch.tensor([12, 15, 20]),  # 3 targets
        ]

        predictions, pred_mask = predictor(context, target_positions)

        # Check validity mask
        assert pred_mask[0, :2].all()  # First 2 valid
        assert not pred_mask[0, 2:].any()  # Rest invalid
        assert pred_mask[1, :4].all()  # First 4 valid
        assert pred_mask[2, :1].all()  # First 1 valid
        assert not pred_mask[2, 1:].any()  # Rest invalid
        assert pred_mask[3, :3].all()  # First 3 valid

    def test_with_context_mask(self, predictor, batch_size, hidden_dim):
        """Test with context attention mask."""
        context = torch.randn(batch_size, 64, hidden_dim)
        target_positions = [torch.tensor([5, 10, 15]) for _ in range(batch_size)]
        context_mask = torch.ones(batch_size, 64, dtype=torch.bool)
        context_mask[:, 32:] = False

        predictions, pred_mask = predictor(context, target_positions, context_mask)

        assert predictions.shape == (batch_size, 3, hidden_dim)
        assert not torch.isnan(predictions).any()

    def test_gradient_flow(self, predictor, batch_size, hidden_dim):
        """Test gradient flow through predictor."""
        context = torch.randn(batch_size, 64, hidden_dim, requires_grad=True)
        target_positions = [torch.tensor([5, 10, 15]) for _ in range(batch_size)]

        predictions, _ = predictor(context, target_positions)
        loss = predictions.sum()
        loss.backward()

        assert context.grad is not None

    def test_deterministic_eval(self, predictor, batch_size, hidden_dim):
        """Test deterministic output in eval mode."""
        predictor.eval()
        context = torch.randn(batch_size, 64, hidden_dim)
        target_positions = [torch.tensor([5, 10, 15]) for _ in range(batch_size)]

        predictions1, _ = predictor(context, target_positions)
        predictions2, _ = predictor(context, target_positions)

        assert torch.allclose(predictions1, predictions2)

    def test_predict_single(self, predictor, batch_size, hidden_dim):
        """Test predict_single convenience method."""
        context = torch.randn(batch_size, 64, hidden_dim)
        position = 25

        prediction = predictor.predict_single(context, position)

        assert prediction.shape == (batch_size, hidden_dim)
        assert not torch.isnan(prediction).any()

    def test_predict_single_with_mask(self, predictor, batch_size, hidden_dim):
        """Test predict_single with context mask."""
        context = torch.randn(batch_size, 64, hidden_dim)
        context_mask = torch.ones(batch_size, 64, dtype=torch.bool)
        context_mask[:, 32:] = False
        position = 25

        prediction = predictor.predict_single(context, position, context_mask)

        assert prediction.shape == (batch_size, hidden_dim)

    def test_get_num_params(self, predictor):
        """Test get_num_params method."""
        num_params = predictor.get_num_params()
        assert num_params > 0
        assert isinstance(num_params, int)

    def test_extra_repr(self, predictor, hidden_dim):
        """Test extra_repr method."""
        repr_str = predictor.extra_repr()
        # JEPAPredictor uses predictor_dim (narrow) instead of hidden_dim
        assert f"predictor_dim={hidden_dim // 2}" in repr_str
        assert f"output_dim={hidden_dim}" in repr_str

    def test_gradient_checkpointing(self, batch_size, hidden_dim):
        """Test with gradient checkpointing enabled."""
        config = PredictorConfig(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            use_gradient_checkpointing=True,
            max_seq_len=1024,
        )
        predictor = JEPAPredictor(config)
        predictor.train()

        context = torch.randn(batch_size, 64, hidden_dim, requires_grad=True)
        target_positions = [torch.tensor([5, 10, 15]) for _ in range(batch_size)]

        predictions, _ = predictor(context, target_positions)
        loss = predictions.sum()
        loss.backward()

        assert context.grad is not None

    def test_narrow_architecture(self, hidden_dim):
        """Test that predictor uses narrow internal dimension."""
        config = PredictorConfig(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            max_seq_len=1024,
        )
        predictor = JEPAPredictor(config)

        # Predictor dim should be half of hidden_dim (narrow by design)
        assert predictor.predictor_dim == hidden_dim // 2

    def test_empty_targets(self, predictor, batch_size, hidden_dim):
        """Test handling of empty target positions."""
        context = torch.randn(batch_size, 64, hidden_dim)
        target_positions = [
            torch.tensor([5, 10, 15]),
            torch.tensor([]),  # Empty
            torch.tensor([3, 8]),
            torch.tensor([12]),
        ]

        predictions, pred_mask = predictor(context, target_positions)

        # Should handle empty gracefully
        assert predictions.shape[0] == batch_size
        assert not pred_mask[1].any()  # All invalid for empty

    def test_mask_token_learning(self, predictor):
        """Test that mask token is learnable."""
        assert predictor.mask_token.requires_grad

    def test_position_embedding(self, predictor):
        """Test that position embeddings are applied."""
        # Position embedding should be part of the predictor
        assert hasattr(predictor, 'pos_embed')
        assert predictor.pos_embed is not None


class TestPredictorIntegration:
    """Integration tests for predictor in JEPA context."""

    def test_predictor_with_encoder_output(self):
        """Test predictor works with typical encoder output dimensions."""
        batch_size = 4
        hidden_dim = 256
        seq_len = 128

        config = PredictorConfig(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            max_seq_len=1024,
        )
        predictor = JEPAPredictor(config)

        # Simulate encoder output (context)
        context = torch.randn(batch_size, seq_len // 2, hidden_dim)

        # Simulate target positions (masked regions)
        target_positions = [
            torch.randint(0, seq_len, (20,)) for _ in range(batch_size)
        ]

        predictions, pred_mask = predictor(context, target_positions)

        assert predictions.shape[0] == batch_size
        assert predictions.shape[2] == hidden_dim
        assert pred_mask.shape[0] == batch_size

    def test_predictor_loss_compatible(self):
        """Test that predictor output is compatible with JEPA loss."""
        from src.loss import JEPALoss, JEPALossConfig

        batch_size = 4
        hidden_dim = 256
        num_targets = 20

        config = PredictorConfig(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            max_seq_len=1024,
        )
        predictor = JEPAPredictor(config)
        loss_fn = JEPALoss(JEPALossConfig())

        context = torch.randn(batch_size, 64, hidden_dim)
        target_positions = [torch.arange(num_targets) for _ in range(batch_size)]

        predictions, pred_mask = predictor(context, target_positions)

        # Simulate target embeddings
        targets = torch.randn(batch_size, num_targets, hidden_dim)

        # Loss should work
        loss, metrics = loss_fn(predictions, targets, mask=pred_mask.float())
        assert not torch.isnan(loss)
        assert loss >= 0
