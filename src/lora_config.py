"""
LoRA configuration for GRPO training.
"""

from peft import LoraConfig, TaskType


def get_simple_lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,                    # Rank
        lora_alpha=32,           # Scaling factor (2x rank)
        lora_dropout=0.05,       # Dropout rate
        bias="none",             # Don't adapt bias
        target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj", 
            "gate_proj", 
            "up_proj",
            "down_proj"
         ],  # Targeted modules
        task_type=TaskType.CAUSAL_LM,
    )


def get_memory_efficient_lora_config() -> LoraConfig:
    """Get memory-efficient LoRA configuration."""
    return LoraConfig(
        r=8,                     # Lower rank
        lora_alpha=16,           # Scaling factor
        lora_dropout=0.0,        # No dropout
        bias="none",
        target_modules=[         # Fewer target modules
            "q_proj",
            "v_proj"
        ],
        task_type=TaskType.CAUSAL_LM,
    )


def get_high_performance_lora_config() -> LoraConfig:
    """Get high-performance LoRA configuration."""
    return LoraConfig(
        r=32,                    # Higher rank
        lora_alpha=64,           # Scaling factor
        lora_dropout=0.1,        # More dropout
        bias="lora_only",        # Adapt bias
        target_modules=[         # More target modules
            "q_proj",
            "k_proj",
            "v_proj", 
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        task_type=TaskType.CAUSAL_LM,
    )


# Default configuration
DEFAULT_LORA_CONFIG = get_simple_lora_config()