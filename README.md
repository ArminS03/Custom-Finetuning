# GRPO Image-to-Code Generation

A minimal implementation of GRPO (Group Relative Policy Optimization) for fine-tuning Qwen3-VL-8B-Instruct to generate matplotlib code from images.

## Quick Start

1. **Setup environment:**
   ```bash
   ./train_grpo.sh setup
   ```

2. **Train the model:**
   ```bash
   ./train_grpo.sh train configs/grpo_config.yaml
   ```

3. **Evaluate the model:**
   ```bash
   python evaluate_grpo.py --model_path outputs/grpo --dataset_path data/image_code_test.jsonl --generate_examples
   ```

## Files

- `train_grpo.py` - Main training script
- `train_grpo.sh` - Training automation script
- `evaluate_grpo.py` - Model evaluation script
- `create_dataset.py` - Dataset creation utilities
- `src/grpo_trainer.py` - GRPO implementation
- `configs/grpo_config.yaml` - Training configuration
- `requirements.txt` - Dependencies

## Usage

```bash
# Setup
./train_grpo.sh setup

# Train with default config
./train_grpo.sh train configs/grpo_config.yaml

# Train with custom epochs
./train_grpo.sh train configs/grpo_config.yaml 10
```

For detailed documentation, see `GRPO_README.md`.
