#!/usr/bin/env python3
"""
Trial training script for O-JEPA on LUMA dataset.

This is a minimal training script to verify the full pipeline works:
- LUMA multimodal dataset loading
- O-JEPA model (ByteEncoder + SharedBackbone + Predictor)
- Muon + AdamW hybrid optimizer with separate learning rates
- Optional hierarchical byte encoding

Usage:
    python examples/trial_luma.py --data_dir /path/to/LUMA/data

For tiny test (default):
    python examples/trial_luma.py --data_dir /home/nsabaj/ai-workshop/LUMA/data
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TrainingConfig
from src.model import JEPAWorldModel
from src.train import create_optimizer, WSDScheduler
from src.data.luma_dataset import LUMALocalDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Trial O-JEPA training on LUMA")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/nsabaj/ai-workshop/LUMA/data",
        help="Path to LUMA data directory",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="vision",
        choices=["vision", "text", "audio"],
        help="Modality to train on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (small for RTX 4060 Ti 8GB)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of training steps",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Model hidden dimension",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)",
    )
    return parser.parse_args()


def collate_fn(batch):
    """Collate function for LUMA batches."""
    # batch is list of dicts with 'vision'/'text'/'audio' keys
    result = {}

    # Get first sample to determine keys
    keys = [k for k in batch[0].keys() if k != 'masks']

    for key in keys:
        tensors = [sample[key] for sample in batch]
        # Pad to max length in batch
        max_len = max(t.shape[0] for t in tensors)
        padded = []
        masks = []
        for t in tensors:
            if t.shape[0] < max_len:
                pad = torch.zeros(max_len - t.shape[0], dtype=t.dtype)
                padded.append(torch.cat([t, pad]))
                mask = torch.cat([
                    torch.ones(t.shape[0], dtype=torch.bool),
                    torch.zeros(max_len - t.shape[0], dtype=torch.bool)
                ])
            else:
                padded.append(t)
                mask = torch.ones(t.shape[0], dtype=torch.bool)
            masks.append(mask)

        result[key] = torch.stack(padded)
        result[f'{key}_mask'] = torch.stack(masks)

    return result


def main():
    args = parse_args()

    print("=" * 60)
    print("O-JEPA Trial Training on LUMA Dataset")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Modality: {args.modality}")
    print(f"Batch size: {args.batch_size} x {args.grad_accum} accum = {args.batch_size * args.grad_accum} effective")
    print(f"Steps: {args.num_steps}")
    print(f"Hierarchical encoding: always enabled (scales 4, 16, 64)")
    print(f"Hidden dim: {args.hidden_dim}")
    print()

    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Create model config
    print("Creating model...")
    from src.config import (
        ByteJEPAConfig, ByteEncoderConfig, BackboneConfig,
        PredictorConfig, MaskingConfig, EMAConfig
    )

    # Build config with desired hidden_dim
    hidden_dim = args.hidden_dim
    num_layers = 4
    num_heads = 4

    byte_encoder_config = ByteEncoderConfig(
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=num_heads,
        max_seq_len=4096,
        hierarchical_scales=(4, 16, 64),
        hierarchical_gating="softmax",
    )

    model_config = ByteJEPAConfig(
        byte_encoder=byte_encoder_config,
        backbone=BackboneConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            use_gradient_checkpointing=False,
        ),
        predictor=PredictorConfig(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=num_heads,
            output_dim=hidden_dim,
            max_seq_len=4096,
        ),
        masking=MaskingConfig(
            num_target_blocks=2,
            target_scale_min=0.1,
            target_scale_max=0.3,
        ),
        ema=EMAConfig(
            ema_warmup_steps=50,
        ),
    )

    # Create model
    model = JEPAWorldModel(model_config)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Create optimizer with Muon + AdamW
    print("Creating optimizer...")
    training_config = TrainingConfig(
        use_muon=True,
        muon_lr=1.5e-2,
        adamw_lr=5e-4,
        learning_rate=1e-4,  # Base for scheduler
        total_steps=args.num_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
    )
    optimizer = create_optimizer(model, config=training_config)
    print()

    # Create scheduler
    scheduler = WSDScheduler(
        optimizer=optimizer,
        base_lr=training_config.learning_rate,
        warmup_steps=min(10, args.num_steps // 10),
        total_steps=args.num_steps,
    )

    # Create dataset
    print("Loading LUMA dataset...")

    # Set sequence length based on modality
    if args.modality == "vision":
        # 32x32x3 = 3072 bytes
        vision_size = (32, 32)
        text_max_len = 1024
        audio_max_len = 8000
    elif args.modality == "text":
        vision_size = (32, 32)
        text_max_len = 2048
        audio_max_len = 8000
    else:  # audio
        vision_size = (32, 32)
        text_max_len = 1024
        audio_max_len = 16000

    dataset = LUMALocalDataset(
        data_dir=args.data_dir,
        split="train",
        modalities=[args.modality],
        vision_size=vision_size,
        text_max_len=text_max_len,
        audio_max_len=audio_max_len,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0,  # IterableDataset works best with 0 workers
    )

    print(f"Dataset size: ~{len(dataset)} samples")
    print()

    # Training loop
    print("Starting training...")
    print("-" * 60)

    model.train()
    step = 0
    total_loss = 0.0
    start_time = time.time()
    accum_loss = 0.0

    data_iter = iter(dataloader)

    while step < args.num_steps:
        optimizer.zero_grad()

        # Gradient accumulation loop
        for accum_step in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Move to device
            byte_ids = batch[args.modality].to(device)

            # Forward pass (scale loss for accumulation)
            loss, outputs = model(byte_ids, modality=args.modality)
            scaled_loss = loss / args.grad_accum

            # Backward pass (gradients accumulate)
            scaled_loss.backward()

            accum_loss += loss.item()

        # Gradient clipping (after accumulation)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step (once per accumulated batch)
        optimizer.step()
        scheduler.step()

        step_loss = accum_loss / args.grad_accum
        total_loss += step_loss
        step += 1
        accum_loss = 0.0

        # Logging
        if step % args.log_every == 0 or step == 1:
            avg_loss = total_loss / step
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            current_lr = optimizer.param_groups[0]['lr']

            # Memory usage
            if device.type == "cuda":
                mem_used = torch.cuda.max_memory_allocated() / 1e9
                mem_str = f" | Mem: {mem_used:.2f}GB"
            else:
                mem_str = ""

            print(
                f"Step {step:4d}/{args.num_steps} | "
                f"Loss: {step_loss:.4f} | "
                f"Avg: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Speed: {steps_per_sec:.1f} steps/s"
                f"{mem_str}"
            )

    # Final stats
    print("-" * 60)
    elapsed = time.time() - start_time
    print(f"Training complete!")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Final avg loss: {total_loss / args.num_steps:.4f}")
    print(f"Average speed: {args.num_steps / elapsed:.1f} steps/s")

    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print()
    print("Trial training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
