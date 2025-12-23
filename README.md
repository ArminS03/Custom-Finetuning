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
