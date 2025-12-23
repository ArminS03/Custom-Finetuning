"""
Main training script for GRPO image-to-code generation.
"""

import os
import yaml
import logging
import argparse
from torch import cuda

# Import from src module
from src.grpo_trainer import CustomGRPOTrainer, ImageCodeConfig, load_and_process_dataset


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> ImageCodeConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract GRPO-specific config
    grpo_dict = config_dict.get('grpo', {})
    
    return ImageCodeConfig(
        model_name=config_dict['model']['name'],
        max_seq_length=config_dict['model']['max_seq_length'],
        max_completion_length=config_dict['model']['max_completion_length'],
        dtype=config_dict['model']['dtype'],
        load_in_4bit=config_dict['model']['load_in_4bit'],
        group_size=grpo_dict.get('group_size', 4),
        learning_rate=config_dict['training']['learning_rate'],
        batch_size=config_dict['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config_dict['training']['gradient_accumulation_steps'],
        num_epochs=config_dict['training']['num_train_epochs'],
        resume_from_checkpoint=config_dict['training']['resume_from_checkpoint'],
        code_execution_weight=config_dict['reward']['code_execution_weight'],
        image_similarity_weight=config_dict['reward']['image_similarity_weight'],
        syntax_error_penalty=config_dict['reward']['syntax_error_penalty'],
        runtime_error_penalty=config_dict['reward']['runtime_error_penalty'],
    )

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="GRPO Training for Image-to-Code Generation")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to configuration file")
    parser.add_argument("--data_path", type=str, 
                       help="Path to training data (overrides config)")
    parser.add_argument("--output_dir", type=str,
                       help="Output directory (overrides config)")
    parser.add_argument("--epochs", type=int,
                       help="Number of training epochs (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_path:
        config.data_path = args.data_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.num_epochs = args.epochs
    
    if cuda.is_available():
        num_gpus = cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")

    # Initialize trainer
    trainer = CustomGRPOTrainer(config)
    
    # Setup model
    logger.info("Setting up model...")
    trainer.setup_model()
    
    # Load dataset
    data_path = getattr(config, 'data_path', "data/image_code_train.jsonl")
    logger.info(f"Loading dataset from {data_path}")
    
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}")
        return
    
    dataset = load_and_process_dataset(data_path, trainer.processor)
    
    # Run training
    logger.info("Starting GRPO training...")
    trainer.train(dataset, config.num_epochs)
    
    # Save model
    output_dir = getattr(config, 'output_dir', './outputs/grpo')
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving model to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    
    # Save training config
    with open(os.path.join(output_dir, "training_config.yaml"), 'w') as f:
        yaml.dump(config.__dict__, f, default_flow_style=False)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
