# O-JEPA

Omni-modality Joint Embedding Predictive Architecture.

A byte-level world model following Yann LeCun's vision: predict abstract representations, not pixels.

## Install

```bash
# Editable install (recommended for development)
pip install -e .

# Or with all optional dependencies
pip install -e ".[full]"

# Or just runtime dependencies
pip install -r requirements.txt
```

## Usage

```python
import torch
from src.model import JEPAWorldModel
from src.config import get_default_config

config = get_default_config()
model = JEPAWorldModel(config)

# Encode any modality as bytes (0-255)
byte_ids = torch.randint(0, 256, (1, 512))
loss, outputs = model(byte_ids, modality="text")  # or "vision", "audio"

# World model capabilities
embedding = model.encode(byte_ids, modality="text")
future = model.predict_future(byte_ids, modality="text", positions=[100, 101, 102])
energy = model.compute_energy(byte_ids, modality="text", target_embeddings=future)
```

## Training

```bash
# Trial training on LUMA dataset
python examples/trial_luma.py --data_dir /path/to/LUMA/data --num_steps 100

# With custom settings
python examples/trial_luma.py \
    --data_dir /path/to/LUMA/data \
    --modality vision \
    --batch_size 8 \
    --hidden_dim 256 \
    --num_steps 1000
```

## Test

```bash
pytest tests/ -v
```

## Architecture

- **ByteEncoder**: bytes (0-255) to embeddings with hierarchical multi-scale processing
- **HierarchicalChunking**: Loom-inspired multi-scale convolutions (4, 16, 64 byte windows) with learned gating
- **SharedBackbone**: transformer for cross-modal representation
- **JEPAPredictor**: cross-attention predictor for masked regions
- **EMA target encoder**: momentum-updated frozen encoder
- **Muon + AdamW optimizer**: native PyTorch 2.9 Muon for 2D weights, AdamW for embeddings/norms

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
