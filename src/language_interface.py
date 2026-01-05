"""
O-JEPA Language Interface: World Model Speaks Through Qwen3-4B

Projects world model embeddings into LLM embedding space as soft tokens.
Following LLaVA pattern: prefix soft tokens + frozen LLM generation.

The world model understands. Qwen speaks.
"""

from dataclasses import dataclass
from typing import Optional, List, Union

import torch
import torch.nn as nn

from .config import LanguageInterfaceConfig


class WorldToLanguageProjection(nn.Module):
    """
    Projects O-JEPA world embeddings to Qwen's embedding space.

    O-JEPA produces a single pooled embedding per input. This projection
    expands it into multiple soft tokens that Qwen can attend to.

    Architecture:
        [batch, ojepa_dim] -> Linear -> SiLU -> Linear -> [batch, num_tokens, qwen_dim]
    """

    def __init__(
        self,
        ojepa_dim: int = 512,
        qwen_dim: int = 2560,
        num_soft_tokens: int = 8,
    ):
        super().__init__()
        self.num_soft_tokens = num_soft_tokens
        self.qwen_dim = qwen_dim
        self.ojepa_dim = ojepa_dim

        self.projection = nn.Sequential(
            nn.Linear(ojepa_dim, qwen_dim, bias=False),
            nn.SiLU(),
            nn.Linear(qwen_dim, qwen_dim * num_soft_tokens, bias=False),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.projection:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, world_embedding: torch.Tensor) -> torch.Tensor:
        """
        Project world embedding to soft tokens.

        Args:
            world_embedding: [batch, ojepa_dim]

        Returns:
            soft_tokens: [batch, num_soft_tokens, qwen_dim]
        """
        batch_size = world_embedding.shape[0]
        projected = self.projection(world_embedding)
        return projected.view(batch_size, self.num_soft_tokens, self.qwen_dim)

    def extra_repr(self) -> str:
        return (
            f"ojepa_dim={self.ojepa_dim}, "
            f"qwen_dim={self.qwen_dim}, "
            f"num_soft_tokens={self.num_soft_tokens}"
        )


class LanguageInterface(nn.Module):
    """
    O-JEPA Language Interface using Qwen3-4B.

    Data flow:
        Input bytes -> O-JEPA.encode() -> [batch, 512]
                            |
                    WorldToLanguageProjection
                            |
                    [batch, num_tokens, 2560] (soft tokens)
                            |
                    Prepend to Qwen input embeddings
                            |
                    Qwen3-4B.generate()
                            |
                    Generated text

    The world model does the thinking. Qwen is just the mouth.
    """

    def __init__(
        self,
        ojepa_model: nn.Module,
        config: Optional[LanguageInterfaceConfig] = None,
    ):
        super().__init__()
        self.config = config or LanguageInterfaceConfig()
        self.ojepa = ojepa_model

        # Create projection layer
        self.projection = WorldToLanguageProjection(
            ojepa_dim=self.config.ojepa_hidden_dim,
            qwen_dim=self.config.projection_hidden_dim,
            num_soft_tokens=self.config.num_soft_tokens,
        )

        # Move projection to same device as O-JEPA
        self._sync_device()

        # Lazy load Qwen (avoid importing transformers until needed)
        self._qwen = None
        self._tokenizer = None

        # Apply freezing
        if self.config.freeze_ojepa:
            self._freeze_ojepa()

    @property
    def qwen(self):
        """Lazy load Qwen model."""
        if self._qwen is None:
            self._load_qwen()
        return self._qwen

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            self._load_qwen()
        return self._tokenizer

    def _load_qwen(self):
        """Load Qwen model and tokenizer from HuggingFace."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.qwen_model_name,
            trust_remote_code=True,
        )

        load_kwargs = {
            "device_map": self.config.device_map,
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }

        if self.config.use_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self._qwen = AutoModelForCausalLM.from_pretrained(
            self.config.qwen_model_name,
            **load_kwargs,
        )

        if self.config.freeze_qwen:
            self._freeze_qwen()

    def _freeze_qwen(self):
        """Freeze Qwen parameters."""
        if self._qwen is not None:
            for param in self._qwen.parameters():
                param.requires_grad = False

    def _freeze_ojepa(self):
        """Freeze O-JEPA parameters."""
        for param in self.ojepa.parameters():
            param.requires_grad = False

    def _sync_device(self):
        """Move projection to same device as O-JEPA (GPU if O-JEPA is on GPU)."""
        ojepa_device = next(self.ojepa.parameters()).device
        self.projection = self.projection.to(ojepa_device)

    def get_trainable_params(self) -> int:
        """Count trainable parameters (should be just projection)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_projection_params(self) -> int:
        """Count projection layer parameters."""
        return sum(p.numel() for p in self.projection.parameters())

    def encode_world(
        self,
        byte_ids: torch.Tensor,
        modality: str = "text",
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode input through O-JEPA world model.

        Unlike model.encode() which uses @torch.no_grad for inference,
        this method preserves gradients for training the projection.

        Args:
            byte_ids: Raw bytes [batch, seq_len]
            modality: Modality name
            attention_mask: Optional attention mask

        Returns:
            World embedding [batch, ojepa_dim]
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(byte_ids, dtype=torch.bool)

        # Directly use O-JEPA components to preserve gradients
        # (model.encode() has @torch.no_grad decorator for inference)
        byte_emb = self.ojepa.byte_encoder(byte_ids, attention_mask)
        _, pooled = self.ojepa.backbone(byte_emb, modality, attention_mask)

        return pooled

    def forward(
        self,
        byte_ids: torch.Tensor,
        modality: str = "text",
        prompt: Optional[str] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Union[str, List[str]]:
        """
        Generate language from world model understanding.

        Args:
            byte_ids: Input as raw bytes [batch, seq_len]
            modality: Input modality
            prompt: Optional text prompt to append after soft tokens
            attention_mask: Optional attention mask for O-JEPA
            max_new_tokens: Override config max_new_tokens
            temperature: Override config temperature
            top_p: Override config top_p

        Returns:
            Generated text string (single) or list of strings (batch)
        """
        # Get world embedding from O-JEPA
        world_embedding = self.encode_world(byte_ids, modality, attention_mask)

        # Project to soft tokens
        soft_tokens = self.projection(world_embedding)

        # Generate with soft tokens as prefix
        return self.generate_from_embedding(
            soft_tokens,
            prompt=prompt,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            temperature=temperature or self.config.temperature,
            top_p=top_p or self.config.top_p,
        )

    def generate_from_embedding(
        self,
        soft_tokens: torch.Tensor,
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Union[str, List[str]]:
        """
        Generate text with soft tokens as prefix.

        Args:
            soft_tokens: [batch, num_tokens, qwen_dim]
            prompt: Optional text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            Generated text
        """
        device = soft_tokens.device
        batch_size = soft_tokens.shape[0]

        # Ensure soft tokens match Qwen dtype
        soft_tokens = soft_tokens.to(dtype=self.qwen.dtype)

        # Build input embeddings
        if prompt:
            # Tokenize prompt
            prompt_tokens = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.to(device)

            # Get prompt embeddings from Qwen
            prompt_embeds = self.qwen.model.embed_tokens(prompt_tokens)
            prompt_embeds = prompt_embeds.to(dtype=self.qwen.dtype)

            # Expand for batch if needed
            if prompt_embeds.shape[0] != batch_size:
                prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)

            # Concatenate: [soft_tokens, prompt_embeds]
            inputs_embeds = torch.cat([soft_tokens, prompt_embeds], dim=1)

            # Create attention mask
            attention_mask = torch.ones(
                batch_size, inputs_embeds.shape[1],
                dtype=torch.long, device=device
            )
        else:
            inputs_embeds = soft_tokens
            attention_mask = torch.ones(
                batch_size, soft_tokens.shape[1],
                dtype=torch.long, device=device
            )

        # Generate
        with torch.no_grad():
            outputs = self.qwen.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Decode output
        generated_text = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )

        return generated_text[0] if batch_size == 1 else generated_text

    def chat(
        self,
        message: str,
        modality: str = "text",
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Simple chat interface.

        Encodes the message through the world model and generates a response.

        Args:
            message: User message (will be encoded as bytes)
            modality: Input modality
            system_prompt: Optional system prompt

        Returns:
            Generated response
        """
        # Convert message to bytes
        byte_ids = torch.tensor(
            [list(message.encode("utf-8"))],
            dtype=torch.long,
            device=next(self.ojepa.parameters()).device,
        )

        # Build prompt
        prompt = system_prompt or "You are a helpful assistant. Respond to the user's message."

        return self.forward(byte_ids, modality=modality, prompt=prompt)

    @classmethod
    def from_pretrained(
        cls,
        ojepa_checkpoint: str,
        config: Optional[LanguageInterfaceConfig] = None,
    ) -> "LanguageInterface":
        """
        Load from pretrained O-JEPA checkpoint.

        Args:
            ojepa_checkpoint: Path to O-JEPA checkpoint
            config: Optional config override

        Returns:
            Initialized LanguageInterface
        """
        from .model import JEPAWorldModel
        from .config import ByteJEPAConfig

        # Load checkpoint
        checkpoint = torch.load(ojepa_checkpoint, map_location="cpu")

        # Create O-JEPA model
        if "config" in checkpoint:
            ojepa_config = ByteJEPAConfig.from_dict(checkpoint["config"])
        else:
            ojepa_config = ByteJEPAConfig()

        ojepa = JEPAWorldModel(ojepa_config)

        if "model_state_dict" in checkpoint:
            ojepa.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            ojepa.load_state_dict(checkpoint["state_dict"])

        # Create config with matching hidden dim
        if config is None:
            config = LanguageInterfaceConfig(
                ojepa_hidden_dim=ojepa_config.hidden_dim,
            )

        return cls(ojepa, config)

    def extra_repr(self) -> str:
        return (
            f"ojepa_dim={self.config.ojepa_hidden_dim}, "
            f"num_soft_tokens={self.config.num_soft_tokens}, "
            f"qwen={self.config.qwen_model_name}"
        )
