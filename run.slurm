#!/bin/bash
#SBATCH --job-name=qwen_grpo_trainer        # Name of the job
#SBATCH --partition=scavenger               # Partition name
#SBATCH --qos=scavenger                     # QOS level
#SBATCH --gres=gpu:l40s:6                   # GPU request
#SBATCH --time=4:00:00                      # Max wall time (hh:mm:ss)
#SBATCH --account=scavenger                 # Account name
#SBATCH --mem=40G
#SBATCH --output=slurm-%j.out               # Standard output and error log

module avail cuda

module load cuda/12.1.1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Set working directory
cd /fs/nexus-projects/figure-agent/Custom_Qwen3_Finetuning


bash train_grpo.sh

