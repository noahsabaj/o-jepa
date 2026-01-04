#!/usr/bin/env python3
"""
ByteJEPA Training

Usage:
    python train.py                     # Train with defaults
    python train.py --data ./my_data    # Train on your data
    python train.py --tiny              # Quick test run
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ByteJEPAConfig, get_tiny_config, get_default_config
from src.model import ByteJEPA
from src.train import train, TrainingConfig
from src.data import SyntheticByteDataset, LUMALocalDataset, get_collator
from src.progress import create_reporter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ByteJEPA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train.py                        Train with defaults
    python train.py --data ./data          Train on your data
    python train.py --tiny                 Quick test (small model, 1000 steps)
    python train.py --steps 50000          Train for 50k steps
    python train.py --resume               Resume from last checkpoint

For advanced options: python train.py --help-all
""",
    )

    # Essential options
    parser.add_argument("--data", type=str, help="Data directory")
    parser.add_argument("--tiny", action="store_true", help="Quick test mode")
    parser.add_argument("--steps", type=int, help="Training steps")
    parser.add_argument("--resume", nargs="?", const=True, default=False,
                        help="Resume training (optionally specify checkpoint path)")
    parser.add_argument("--output", type=str, default="./outputs", help="Output directory")

    # Show all options
    parser.add_argument("--help-all", action="store_true", help="Show all options")

    # Advanced options (hidden from main help)
    advanced = parser.add_argument_group("advanced options")
    advanced.add_argument("--config", type=str, help="YAML config file")
    advanced.add_argument("--batch-size", type=int, default=32)
    advanced.add_argument("--lr", type=float, default=1e-4)
    advanced.add_argument("--hidden-dim", type=int, default=512)
    advanced.add_argument("--modalities", nargs="+", default=["vision", "text"],
                          choices=["vision", "text", "audio"])
    advanced.add_argument("--bidirectional", action="store_true", default=True)
    advanced.add_argument("--device", type=str,
                          default="cuda" if torch.cuda.is_available() else "cpu")
    advanced.add_argument("--wandb", action="store_true", help="Log to W&B")
    advanced.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Show full help if requested
    if args.help_all:
        parser.print_help()
        sys.exit(0)

    return args


def main():
    args = parse_args()

    # Build config
    if args.tiny:
        config = get_tiny_config()
        steps = args.steps or 1000
        batch_size = 4
    else:
        config = get_default_config()
        steps = args.steps or 100000
        batch_size = args.batch_size

    # Override from YAML if provided
    if args.config:
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)
        # Apply yaml overrides here if needed

    # Override hidden dim if specified
    if args.hidden_dim != 512 and not args.tiny:
        config.byte_encoder.hidden_dim = args.hidden_dim
        config.backbone.hidden_dim = args.hidden_dim
        config.predictor.hidden_dim = args.hidden_dim
        config.predictor.output_dim = args.hidden_dim

    # Create model
    model = ByteJEPA(config)
    model = model.to(args.device)

    # Create data loader
    if args.data:
        if Path(args.data).exists():
            dataset = LUMALocalDataset(
                data_dir=args.data,
                modalities=tuple(args.modalities),
                vision_seq_len=config.data.vision_seq_len,
                text_max_seq_len=config.data.text_max_seq_len,
            )
        else:
            print(f"Warning: {args.data} not found, using synthetic data")
            dataset = SyntheticByteDataset(
                num_samples=10000,
                modalities=tuple(args.modalities),
            )
    else:
        dataset = SyntheticByteDataset(
            num_samples=10000,
            modalities=tuple(args.modalities),
        )

    collator = get_collator(
        mode="paired",
        source_modality=args.modalities[0],
        target_modality=args.modalities[1] if len(args.modalities) > 1 else args.modalities[0],
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    # Training config
    training_config = TrainingConfig(
        learning_rate=args.lr,
        total_steps=steps,
        batch_size=batch_size,
    )

    # Handle resume
    resume_path = None
    if args.resume:
        if isinstance(args.resume, str):
            resume_path = args.resume
        else:
            # Auto-find latest checkpoint
            ckpt_dir = Path(args.output)
            if ckpt_dir.exists():
                ckpts = list(ckpt_dir.glob("*.pt"))
                if ckpts:
                    resume_path = str(max(ckpts, key=os.path.getmtime))
                    print(f"Resuming from {resume_path}")

    # Create reporter
    reporter = create_reporter()

    # Train
    print(f"\n{'='*50}")
    print(f"ByteJEPA Training")
    print(f"{'='*50}")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Data: {len(dataset):,} samples")
    print(f"Steps: {steps:,}")
    print(f"Device: {args.device}")
    print(f"{'='*50}\n")

    train(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        config=config,
        training_config=training_config,
        save_dir=args.output,
        resume_path=resume_path,
        device=args.device,
        use_wandb=args.wandb,
        bidirectional=args.bidirectional,
        source_modality=args.modalities[0],
        target_modality=args.modalities[1] if len(args.modalities) > 1 else args.modalities[0],
        reporter=reporter,
    )


if __name__ == "__main__":
    main()
