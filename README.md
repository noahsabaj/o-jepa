# O-JEPA

Omni-Joint Embedding Predictive Architecture.

A byte-level world model following Yann LeCun's vision: predict abstract representations, not pixels.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```python
from src import JEPAWorldModel, get_default_config

config = get_default_config()
model = JEPAWorldModel(config)

# Encode any modality as bytes
byte_ids = torch.randint(0, 256, (1, 512))
loss, outputs = model(byte_ids, modality="bytes")

# World model capabilities
embedding = model.encode(byte_ids)
future = model.predict_future(byte_ids, positions=[100, 101, 102])
energy = model.compute_energy(state_bytes, action_bytes)
```

## Test

```bash
pytest tests/ -v
```

## Architecture

- ByteEncoder: bytes (0-255) to embeddings
- SharedBackbone: 12-layer transformer
- JEPAPredictor: cross-attention predictor
- EMA target encoder (frozen)
- MSE loss in latent space (non-contrastive)

## License

MIT
