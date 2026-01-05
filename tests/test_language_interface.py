"""
Tests for O-JEPA Language Interface.

Tests the WorldToLanguageProjection and LanguageInterface components
that allow the world model to speak through Qwen3-4B.
"""

import pytest
import torch
import torch.nn as nn

from src.config import LanguageInterfaceConfig, ByteJEPAConfig, get_tiny_config
from src.language_interface import WorldToLanguageProjection, LanguageInterface


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestLanguageInterfaceConfig:
    """Tests for LanguageInterfaceConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LanguageInterfaceConfig()

        assert config.qwen_model_name == "Qwen/Qwen3-4B"
        assert config.num_soft_tokens == 8
        assert config.projection_hidden_dim == 2560  # Qwen3-4B hidden size
        assert config.ojepa_hidden_dim == 512
        assert config.freeze_qwen is True
        assert config.freeze_ojepa is True
        assert config.use_4bit is False
        assert config.max_new_tokens == 256
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.device_map == "auto"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LanguageInterfaceConfig(
            num_soft_tokens=16,
            ojepa_hidden_dim=256,
            use_4bit=True,
            max_new_tokens=512,
        )

        assert config.num_soft_tokens == 16
        assert config.ojepa_hidden_dim == 256
        assert config.use_4bit is True
        assert config.max_new_tokens == 512


# =============================================================================
# PROJECTION TESTS
# =============================================================================

class TestWorldToLanguageProjection:
    """Tests for WorldToLanguageProjection layer."""

    def test_output_shape(self, world_to_language_projection, language_interface_config):
        """Test projection produces correct output shape."""
        batch_size = 4
        ojepa_dim = language_interface_config.ojepa_hidden_dim
        qwen_dim = language_interface_config.projection_hidden_dim
        num_tokens = language_interface_config.num_soft_tokens

        # Input: [batch, ojepa_dim]
        world_embedding = torch.randn(batch_size, ojepa_dim)

        # Output: [batch, num_soft_tokens, qwen_dim]
        soft_tokens = world_to_language_projection(world_embedding)

        assert soft_tokens.shape == (batch_size, num_tokens, qwen_dim)

    def test_batch_size_one(self, world_to_language_projection, language_interface_config):
        """Test projection works with batch size 1."""
        ojepa_dim = language_interface_config.ojepa_hidden_dim
        qwen_dim = language_interface_config.projection_hidden_dim
        num_tokens = language_interface_config.num_soft_tokens

        world_embedding = torch.randn(1, ojepa_dim)
        soft_tokens = world_to_language_projection(world_embedding)

        assert soft_tokens.shape == (1, num_tokens, qwen_dim)

    def test_gradient_flow(self, world_to_language_projection, language_interface_config):
        """Test gradients flow through projection."""
        ojepa_dim = language_interface_config.ojepa_hidden_dim

        world_embedding = torch.randn(2, ojepa_dim, requires_grad=True)
        soft_tokens = world_to_language_projection(world_embedding)

        # Compute loss and backprop
        loss = soft_tokens.sum()
        loss.backward()

        # Check gradients exist
        assert world_embedding.grad is not None
        assert world_embedding.grad.shape == world_embedding.shape

        # Check projection parameters have gradients
        for param in world_to_language_projection.parameters():
            assert param.grad is not None

    def test_deterministic(self, world_to_language_projection, language_interface_config):
        """Test projection is deterministic (no dropout during eval)."""
        ojepa_dim = language_interface_config.ojepa_hidden_dim

        world_to_language_projection.eval()
        world_embedding = torch.randn(2, ojepa_dim)

        output1 = world_to_language_projection(world_embedding)
        output2 = world_to_language_projection(world_embedding)

        assert torch.allclose(output1, output2)

    def test_parameter_count(self, language_interface_config):
        """Test projection has expected parameter count."""
        ojepa_dim = language_interface_config.ojepa_hidden_dim
        qwen_dim = language_interface_config.projection_hidden_dim
        num_tokens = language_interface_config.num_soft_tokens

        projection = WorldToLanguageProjection(
            ojepa_dim=ojepa_dim,
            qwen_dim=qwen_dim,
            num_soft_tokens=num_tokens,
        )

        # Expected: Linear(ojepa->qwen) + Linear(qwen->qwen*num_tokens)
        # No biases
        expected_params = ojepa_dim * qwen_dim + qwen_dim * (qwen_dim * num_tokens)
        actual_params = sum(p.numel() for p in projection.parameters())

        assert actual_params == expected_params

    def test_extra_repr(self, world_to_language_projection, language_interface_config):
        """Test extra_repr provides useful info."""
        repr_str = world_to_language_projection.extra_repr()

        assert "ojepa_dim=" in repr_str
        assert "qwen_dim=" in repr_str
        assert "num_soft_tokens=" in repr_str


# =============================================================================
# PROJECTION SHAPE TESTS (VARIOUS CONFIGURATIONS)
# =============================================================================

class TestProjectionShapes:
    """Test projection with various dimension configurations."""

    @pytest.mark.parametrize("ojepa_dim,qwen_dim,num_tokens", [
        (256, 256, 4),   # Tiny test config
        (512, 2560, 8),  # Default O-JEPA -> Qwen3-4B
        (384, 2560, 8),  # Small O-JEPA -> Qwen3-4B
        (768, 4096, 16), # Large O-JEPA -> Qwen3-8B
    ])
    def test_various_dimensions(self, ojepa_dim, qwen_dim, num_tokens):
        """Test projection with various dimension configurations."""
        batch_size = 2
        projection = WorldToLanguageProjection(
            ojepa_dim=ojepa_dim,
            qwen_dim=qwen_dim,
            num_soft_tokens=num_tokens,
        )

        world_embedding = torch.randn(batch_size, ojepa_dim)
        soft_tokens = projection(world_embedding)

        assert soft_tokens.shape == (batch_size, num_tokens, qwen_dim)


# =============================================================================
# LANGUAGE INTERFACE TESTS (WITHOUT QWEN)
# =============================================================================

class TestLanguageInterfaceWithoutQwen:
    """Tests for LanguageInterface that don't require loading Qwen."""

    def test_init_with_ojepa(self, byte_jepa_model, language_interface_config):
        """Test initialization with O-JEPA model."""
        interface = LanguageInterface(
            ojepa_model=byte_jepa_model,
            config=language_interface_config,
        )

        assert interface.ojepa is byte_jepa_model
        assert interface.config == language_interface_config
        assert interface.projection is not None

    def test_projection_created(self, byte_jepa_model, language_interface_config):
        """Test projection layer is created correctly."""
        interface = LanguageInterface(
            ojepa_model=byte_jepa_model,
            config=language_interface_config,
        )

        assert isinstance(interface.projection, WorldToLanguageProjection)
        assert interface.projection.ojepa_dim == language_interface_config.ojepa_hidden_dim
        assert interface.projection.qwen_dim == language_interface_config.projection_hidden_dim
        assert interface.projection.num_soft_tokens == language_interface_config.num_soft_tokens

    def test_freeze_ojepa(self, byte_jepa_model, language_interface_config):
        """Test O-JEPA freezing when configured."""
        language_interface_config.freeze_ojepa = True

        interface = LanguageInterface(
            ojepa_model=byte_jepa_model,
            config=language_interface_config,
        )

        # All O-JEPA parameters should be frozen
        for param in interface.ojepa.parameters():
            assert param.requires_grad is False

    def test_ojepa_not_frozen(self, byte_jepa_model, language_interface_config):
        """Test O-JEPA not frozen when configured."""
        language_interface_config.freeze_ojepa = False

        interface = LanguageInterface(
            ojepa_model=byte_jepa_model,
            config=language_interface_config,
        )

        # O-JEPA parameters should be trainable
        trainable_count = sum(1 for p in interface.ojepa.parameters() if p.requires_grad)
        assert trainable_count > 0

    def test_trainable_params_projection_only(self, byte_jepa_model, language_interface_config):
        """Test only projection is trainable when both models frozen."""
        language_interface_config.freeze_ojepa = True
        language_interface_config.freeze_qwen = True

        interface = LanguageInterface(
            ojepa_model=byte_jepa_model,
            config=language_interface_config,
        )

        # Trainable params should equal projection params
        trainable = interface.get_trainable_params()
        projection_params = interface.get_projection_params()

        assert trainable == projection_params

    def test_extra_repr(self, byte_jepa_model, language_interface_config):
        """Test extra_repr provides useful info."""
        interface = LanguageInterface(
            ojepa_model=byte_jepa_model,
            config=language_interface_config,
        )

        repr_str = interface.extra_repr()

        assert "ojepa_dim=" in repr_str
        assert "num_soft_tokens=" in repr_str
        assert "qwen=" in repr_str


# =============================================================================
# ENCODE WORLD TESTS
# =============================================================================

class TestEncodeWorld:
    """Tests for encode_world method."""

    def test_encode_world_text(self, byte_jepa_model, language_interface_config, tiny_config):
        """Test encoding text through world model."""
        interface = LanguageInterface(
            ojepa_model=byte_jepa_model,
            config=language_interface_config,
        )

        batch_size = 2
        seq_len = tiny_config.data.text_max_seq_len
        text_bytes = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)

        embedding = interface.encode_world(text_bytes, modality="text")

        # Should produce [batch, hidden_dim]
        assert embedding.shape == (batch_size, tiny_config.backbone.hidden_dim)

    def test_encode_world_vision(self, byte_jepa_model, language_interface_config, tiny_config):
        """Test encoding vision through world model."""
        interface = LanguageInterface(
            ojepa_model=byte_jepa_model,
            config=language_interface_config,
        )

        batch_size = 2
        seq_len = tiny_config.data.vision_seq_len
        vision_bytes = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)

        embedding = interface.encode_world(vision_bytes, modality="vision")

        assert embedding.shape == (batch_size, tiny_config.backbone.hidden_dim)

    def test_encode_world_audio(self, byte_jepa_model, language_interface_config, tiny_config):
        """Test encoding audio through world model."""
        interface = LanguageInterface(
            ojepa_model=byte_jepa_model,
            config=language_interface_config,
        )

        batch_size = 2
        seq_len = tiny_config.data.audio_max_seq_len
        audio_bytes = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)

        embedding = interface.encode_world(audio_bytes, modality="audio")

        assert embedding.shape == (batch_size, tiny_config.backbone.hidden_dim)


# =============================================================================
# END-TO-END PROJECTION TESTS
# =============================================================================

class TestEndToEndProjection:
    """Test full pipeline from bytes to soft tokens (without Qwen)."""

    def test_bytes_to_soft_tokens(self, byte_jepa_model, language_interface_config, tiny_config):
        """Test full flow from bytes to soft tokens."""
        interface = LanguageInterface(
            ojepa_model=byte_jepa_model,
            config=language_interface_config,
        )

        batch_size = 2
        seq_len = tiny_config.data.text_max_seq_len
        text_bytes = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)

        # Encode through world model
        world_embedding = interface.encode_world(text_bytes, modality="text")

        # Project to soft tokens
        soft_tokens = interface.projection(world_embedding)

        expected_shape = (
            batch_size,
            language_interface_config.num_soft_tokens,
            language_interface_config.projection_hidden_dim,
        )
        assert soft_tokens.shape == expected_shape

    def test_gradient_flow_full_pipeline(self, tiny_config):
        """Test gradients flow through full pipeline when O-JEPA not frozen."""
        from src.model import JEPAWorldModel

        # Create fresh model with unfrozen O-JEPA
        ojepa = JEPAWorldModel(tiny_config)

        # Create config with O-JEPA NOT frozen
        config = LanguageInterfaceConfig(
            ojepa_hidden_dim=tiny_config.backbone.hidden_dim,
            num_soft_tokens=4,
            projection_hidden_dim=256,
            freeze_ojepa=False,  # Key: don't freeze O-JEPA
        )

        interface = LanguageInterface(
            ojepa_model=ojepa,
            config=config,
        )

        batch_size = 2
        seq_len = tiny_config.data.text_max_seq_len
        text_bytes = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)

        # Forward pass
        world_embedding = interface.encode_world(text_bytes, modality="text")
        soft_tokens = interface.projection(world_embedding)

        # Backward pass
        loss = soft_tokens.sum()
        loss.backward()

        # Check O-JEPA has gradients
        ojepa_grad_count = sum(
            1 for p in interface.ojepa.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert ojepa_grad_count > 0

        # Check projection has gradients
        proj_grad_count = sum(
            1 for p in interface.projection.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert proj_grad_count > 0


# =============================================================================
# CONFIG SYNC TESTS
# =============================================================================

class TestConfigSync:
    """Test that configs stay synchronized."""

    def test_language_interface_in_main_config(self):
        """Test LanguageInterfaceConfig is included in ByteJEPAConfig."""
        config = ByteJEPAConfig()

        assert hasattr(config, "language_interface")
        assert isinstance(config.language_interface, LanguageInterfaceConfig)

    def test_hidden_dim_sync(self):
        """Test hidden dims stay synchronized."""
        config = ByteJEPAConfig()

        # Language interface should auto-sync with backbone
        assert config.language_interface.ojepa_hidden_dim == config.backbone.hidden_dim

    def test_tiny_config_sync(self):
        """Test tiny config has synchronized dims."""
        config = get_tiny_config()

        assert config.language_interface.ojepa_hidden_dim == config.backbone.hidden_dim
        assert config.language_interface.ojepa_hidden_dim == 256  # Tiny dim


# =============================================================================
# QWEN LAZY LOADING TESTS
# =============================================================================

class TestQwenLazyLoading:
    """Test Qwen lazy loading behavior."""

    def test_qwen_not_loaded_on_init(self, byte_jepa_model, language_interface_config):
        """Test Qwen is not loaded during init."""
        interface = LanguageInterface(
            ojepa_model=byte_jepa_model,
            config=language_interface_config,
        )

        # _qwen should be None until accessed
        assert interface._qwen is None
        assert interface._tokenizer is None

    def test_projection_works_without_qwen(self, byte_jepa_model, language_interface_config, tiny_config):
        """Test projection can be trained without loading Qwen."""
        interface = LanguageInterface(
            ojepa_model=byte_jepa_model,
            config=language_interface_config,
        )

        batch_size = 2
        seq_len = tiny_config.data.text_max_seq_len
        text_bytes = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)

        # This should work without loading Qwen
        world_embedding = interface.encode_world(text_bytes, modality="text")
        soft_tokens = interface.projection(world_embedding)

        # Qwen still not loaded
        assert interface._qwen is None

        # But output is valid
        assert soft_tokens.shape[0] == batch_size
