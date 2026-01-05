#!/usr/bin/env python3
"""
Test Qwen3-4B inference - both standalone and through O-JEPA.

This script tests:
1. Qwen3-4B loads correctly
2. Standalone generation works
3. O-JEPA -> projection -> Qwen pipeline works
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_gpu_info():
    """Print GPU memory info."""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name}")
        print(f"Total memory: {gpu.total_memory / 1e9:.1f} GB")
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        print("No GPU available, using CPU")


def test_qwen_standalone(use_4bit: bool = True):
    """Test 1: Load and generate with Qwen3-4B directly."""
    print("\n" + "=" * 60)
    print("TEST 1: Qwen3-4B Standalone Generation")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-4B"
    print(f"\nLoading {model_name}...")
    print_gpu_info()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded")

    # Load model
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )
        print("Model loaded with 4-bit quantization")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.float16,
        )
        print("Model loaded (fp16)")

    print_gpu_info()

    # Test generation
    prompt = "What is a world model in AI?"
    print(f"\nPrompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResponse:\n{response}")

    print("\nQwen standalone test PASSED")
    return model, tokenizer


def test_ojepa_encode():
    """Test 2: O-JEPA encoding works."""
    print("\n" + "=" * 60)
    print("TEST 2: O-JEPA World Model Encoding")
    print("=" * 60)

    from src.model import JEPAWorldModel
    from src.config import get_default_config

    config = get_default_config()
    print(f"\nCreating O-JEPA model (hidden_dim={config.backbone.hidden_dim})...")

    model = JEPAWorldModel(config)

    if torch.cuda.is_available():
        model = model.cuda()
        print("O-JEPA moved to GPU")

    print_gpu_info()

    # Test encoding text
    text = "Hello, world!"
    text_bytes = torch.tensor([[ord(c) for c in text]], dtype=torch.long)
    # Pad to expected length
    text_bytes = torch.nn.functional.pad(
        text_bytes,
        (0, config.data.text_max_seq_len - text_bytes.shape[1]),
        value=0
    )

    if torch.cuda.is_available():
        text_bytes = text_bytes.cuda()

    print(f"\nEncoding: '{text}'")
    print(f"Input shape: {text_bytes.shape}")

    # Use encode_context for inference (with proper mask)
    mask = torch.ones_like(text_bytes, dtype=torch.bool)
    mask[:, len(text):] = False  # Mask padding

    with torch.no_grad():
        embedding = model.encode(text_bytes, modality="text", attention_mask=mask)

    print(f"Output embedding shape: {embedding.shape}")
    print(f"Embedding norm: {embedding.norm().item():.4f}")

    print("\nO-JEPA encoding test PASSED")
    return model, config


def test_full_pipeline(use_4bit: bool = True):
    """Test 3: Full O-JEPA -> Projection -> Qwen pipeline."""
    print("\n" + "=" * 60)
    print("TEST 3: Full Pipeline (O-JEPA -> Projection -> Qwen)")
    print("=" * 60)

    from src.model import JEPAWorldModel
    from src.config import get_default_config, LanguageInterfaceConfig
    from src.language_interface import LanguageInterface

    # Create O-JEPA
    ojepa_config = get_default_config()
    ojepa = JEPAWorldModel(ojepa_config)

    if torch.cuda.is_available():
        ojepa = ojepa.cuda()

    # Create language interface
    lang_config = LanguageInterfaceConfig(
        ojepa_hidden_dim=ojepa_config.backbone.hidden_dim,
        use_4bit=use_4bit,
    )

    print(f"\nCreating LanguageInterface...")
    print(f"  O-JEPA dim: {lang_config.ojepa_hidden_dim}")
    print(f"  Soft tokens: {lang_config.num_soft_tokens}")
    print(f"  Qwen dim: {lang_config.projection_hidden_dim}")
    print(f"  4-bit: {lang_config.use_4bit}")

    interface = LanguageInterface(
        ojepa_model=ojepa,
        config=lang_config,
    )

    print_gpu_info()

    # Test encoding through interface
    text = "The cat sat on the mat."
    text_bytes = torch.tensor([[ord(c) for c in text]], dtype=torch.long)
    text_bytes = torch.nn.functional.pad(
        text_bytes,
        (0, ojepa_config.data.text_max_seq_len - text_bytes.shape[1]),
        value=0
    )

    if torch.cuda.is_available():
        text_bytes = text_bytes.cuda()

    print(f"\nInput text: '{text}'")

    # Encode through world model
    world_embedding = interface.encode_world(text_bytes, modality="text")
    print(f"World embedding shape: {world_embedding.shape}")

    # Project to soft tokens
    soft_tokens = interface.projection(world_embedding)
    print(f"Soft tokens shape: {soft_tokens.shape}")

    # Generate with Qwen (this will load Qwen on first call)
    print("\nLoading Qwen and generating...")
    print_gpu_info()

    response = interface(
        text_bytes,
        modality="text",
        prompt="Describe what you understood:",
        max_new_tokens=100,
    )

    print(f"\nGenerated response:\n{response}")
    print_gpu_info()

    print("\nFull pipeline test PASSED")


def main():
    """Run all tests."""
    print("O-JEPA + Qwen3-4B Integration Tests")
    print("=" * 60)

    # Check GPU
    print_gpu_info()

    # Determine if we should use 4-bit based on GPU memory
    use_4bit = True
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_mem > 16:
            use_4bit = False
            print(f"\nGPU has {gpu_mem:.1f}GB - using fp16")
        else:
            print(f"\nGPU has {gpu_mem:.1f}GB - using 4-bit quantization")

    try:
        # Test 1: Qwen standalone
        qwen_model, tokenizer = test_qwen_standalone(use_4bit)

        # Clear Qwen from memory before next test
        del qwen_model, tokenizer
        torch.cuda.empty_cache()

        # Test 2: O-JEPA encoding
        ojepa_model, config = test_ojepa_encode()

        # Clear O-JEPA before full pipeline
        del ojepa_model
        torch.cuda.empty_cache()

        # Test 3: Full pipeline
        test_full_pipeline(use_4bit)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
