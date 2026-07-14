# Murmur-RSM Training Framework

Reproducible training pipeline for Murmur-RSM: a recurrent-depth causal language model with shared recurrent core, GQA attention, and optional Mamba-3 mixer.

## Architecture

- **Prelude**: Initial transformer blocks (non-recurrent)
- **Recurrent Core**: Shared transformer blocks executed T times with stable input injection
- **Coda**: Final transformer blocks (non-recurrent)
- **LM Head**: Tied with token embeddings

## Quick Start

### Local CPU Test

```bash
cd science
pip install -e .
pytest tests/unit -v
```

### Automatic dataset collection

```bash
python scripts/build_hf_corpus.py --config configs/data/base_mix_v0_1.toml
```

The builder uses `datasets.load_dataset(..., streaming=True)` and writes `corpus.txt`, `train.jsonl`, `val.jsonl` and `provenance.json`. For strict reproducibility, replace each `revision = "main"` in the TOML with a dataset commit SHA.

### Kaggle T4 Training

```bash
# 1. Hardware probe
python scripts/hardware_probe.py --output artifacts/hardware.json

# 2. Train tokenizer
python scripts/train_tokenizer.py --corpus data/corpus.txt --output artifacts/tokenizer.model

# 3. Prepare data
python scripts/prepare_data.py --config configs/smoke_gqa.toml --tokenizer artifacts/tokenizer.model --train-input data/train.jsonl --val-input data/val.jsonl --output artifacts/data

# 4. Train
python scripts/train.py --config configs/smoke_gqa.toml --run-dir artifacts/runs/smoke_01

# 5. Evaluate
python scripts/evaluate.py --config configs/smoke_gqa.toml --checkpoint artifacts/runs/smoke_01/checkpoints/last --output artifacts/eval.json

# 6. Generate
python scripts/generate.py --config configs/smoke_gqa.toml --tokenizer artifacts/tokenizer.model --checkpoint artifacts/runs/smoke_01/checkpoints/last --prompt "The future of AI is"
```

## Project Structure

```
science/
├── configs/              # TOML configuration files
├── scripts/              # CLI entry points
├── src/murmur/          # Core library
│   ├── config.py        # Run configuration
│   ├── tokenizer.py     # SentencePiece BPE tokenizer
│   ├── data/            # Data pipeline (manifest, packing, dataset, sampler)
│   ├── model/           # Model architecture (attention, MLP, recurrent core)
│   ├── training/        # Trainer, optimizer, scheduler, checkpoint
│   └── evaluation/      # Evaluation metrics and generation
├── tests/               # Unit, integration, acceptance tests
├── third_party/         # External dependencies (Mamba lock)
└── artifacts/           # Training outputs (checkpoints, logs, eval)
```

## Configuration

See `configs/smoke_gqa.toml` for Smoke S (30-38M params) and `configs/prototype_gqa.toml` for Prototype P (140-158M params).

## Requirements

- Python 3.11+
- PyTorch 2.0+ with CUDA 12.x
- NVIDIA T4 (16 GB VRAM) for Smoke S training
- Optional: `mamba-ssm` for Mamba-3 mixer (requires H100 for full performance)

## License

Apache-2.0 (model code), upstream dependencies retain their original licenses.
