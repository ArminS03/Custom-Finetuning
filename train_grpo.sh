#!/bin/bash
set -e

EPOCHS="3"
DATA_PATH="/fs/nexus-projects/figure-agent/Custom_Qwen3_Finetuning/data/chartcoder-16k"
OUTPUT_DIR="/fs/nexus-projects/figure-agent/Custom_Qwen3_Finetuning/output/Qwen3_VL-v1"

# Config Files
CONFIG_FILE="./configs/grpo_config.yaml"

echo "Config file: ${CONFIG_FILE}"
echo "Data path: ${DATA_PATH}"
echo "Output dir: ${OUTPUT_DIR}"

source ~/.bashrc
conda activate grpo

# Set DeepSpeed environment variables
export TOKENIZERS_PARALLELISM=false

accelerate launch --num_processes 8 --num_machines 1 \
    --config_file ./configs/deepspeed_zero3.yaml train.py \
    --config ${CONFIG_FILE} \
    --epochs ${EPOCHS} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \

echo "GRPO training completed successfully"
