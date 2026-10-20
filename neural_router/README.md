# `neural_router/` — WhiFlash-Neural

The learned routing policy, **WhiFlash-Neural**, is a small MLP, trained offline once per target model, that takes the target's final hidden state `h_t` and predicts the **acceptance-length gain** of switching drafters (ΔAL = AL_pos − AL_neg). The discrete routing decision is then made hardware-aware via a threshold. This directory contains its training pipeline.

## Files

| File | Purpose |
|---|---|
| `modeling.py` | `RouterModel` — a 2-layer MLP that maps the target model's final hidden state to a scalar score. |
| `generate_training_data.py` | Runs drafters (the default set is EAGLE-3 and DFlash) over a dataset and records each step's hidden state plus per-drafter acceptance lengths as training examples. |
| `train.py` | Trains `RouterModel` with the HF `Trainer` using an MSE loss on the acceptance-length difference (`label_pos_col − label_neg_col`; defaults to `eagle3 − dflash`). |

## Workflow

The pipeline has three stages: (1) generate training data, (2) train the router, and (3) export the checkpoint, each driven by a script in [`../scripts/release/train/`](../scripts/release/train/). All scripts share one checkpoint root, `WHIFLASH_CKPT_ROOT` (default `~/checkpoints/whiflash`), and read/write under its `training_data/`, `router_models/` (output path of `train_router.sh` and input path of `export_router_models.sh`), and `neural_routers/` (output path of `export_router_models.sh` and directory that [`benchmark.sh`](../scripts/release/benchmark/benchmark.sh) uses for evaluation) subdirectories. Because the router conditions only on the target's hidden state, training data is collected on a task-agnostic instruction corpus (the paper uses the Dolci Instruct SFT dataset) and the resulting router generalises across downstream tasks, so this cost is amortised.

Example scripts for Eagle3 & DFlash drafters

```bash
export WHIFLASH_CKPT_ROOT="$HOME/checkpoints/whiflash"   # base dir for all router artifacts

# 1) Generate training data for a (target model, dataset) pair.
bash scripts/release/train/generate_training_data.sh Qwen/Qwen3-8B dolci
bash scripts/release/train/generate_training_data.sh Qwen/Qwen3-8B dolci-test

# 2) Train the router for one (target, dataset)...
bash scripts/release/train/train_router.sh Qwen/Qwen3-8B dolci
#    ...or train all supported targets x datasets in one go:
bash scripts/release/train/batch_train_router.sh

# 3) Export trained checkpoints to neural_routers/<run_name>_<version>.safetensors
bash scripts/release/train/export_router_models.sh v1
```

Example scripts for Eagle3 & DFlash & DDTree drafters

```bash
export WHIFLASH_CKPT_ROOT="$HOME/checkpoints/whiflash"   # base dir for all router artifacts

# 1) Generate training data for a (target model, dataset) pair.
bash scripts/release/train/generate_training_data.sh Qwen/Qwen3-8B dolci eagle3 dflash ddtree
bash scripts/release/train/generate_training_data.sh Qwen/Qwen3-8B dolci-test eagle3 dflash ddtree

# 2) Train the router for one (target, dataset)...
bash scripts/release/train/train_router.sh Qwen/Qwen3-8B dolci eagle3-dflash-ddtree eagle_acceptance_lengths ddtree_b128_acceptance_lengths
#    ...or train all supported targets x datasets in one go:
bash scripts/release/train/batch_train_router.sh

# 3) Export trained checkpoints to neural_routers/<run_name>_<version>.safetensors
bash scripts/release/train/export_router_models.sh v1
```

The exported `.safetensors` is what you pass to WhiFlash via `--whiflash-neural-router-path` (see the top-level README's Quick Start). That flag also accepts a Hugging Face Hub repo id or a local directory, so our published routers can be used without this pipeline at all — see [Released Checkpoints](../README.md#released-checkpoints).

**Notes.**

- Supported targets in these scripts are `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B`, and `meta-llama/Llama-3.1-8B-Instruct`.
- For Qwen targets, `generate_training_data.sh` uses our released EAGLE-3 drafters (`WhiFlash/Qwen3-8B-EAGLE3`, `WhiFlash/Qwen3-4B-EAGLE3`), the stronger drafters from the paper.
