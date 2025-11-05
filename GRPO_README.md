# GRPO Image-to-Code Generation for Qwen3-VL-8B-Instruct

This project implements **GRPO (Group Relative Policy Optimization)** specifically for fine-tuning Qwen3-VL-8B-Instruct to generate matplotlib code from images.

## 🎯 Project Overview

The system takes an image as input and generates Python matplotlib code that recreates the image. It uses GRPO to optimize the model based on:
1. **Code Execution Success**: Whether the generated code runs without errors
2. **Image Similarity**: How closely the generated image matches the input image
3. **Custom Scoring**: Your custom evaluation metrics

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd Custom_Qwen3_Finetuning
pip install -r requirements.txt
pip install opencv-python scikit-image scikit-learn
```

### 2. Create Sample Dataset
```bash
python create_dataset.py --create_sample
```

### 3. Train with GRPO
```bash
python train_grpo.py --config configs/grpo_config.yaml --create_sample_data
```

### 4. Evaluate Model
```bash
python evaluate_grpo.py --model_path outputs/grpo --dataset_path data/image_code_test.jsonl --generate_examples
```

## 📁 Project Structure

```
Custom_Qwen3_Finetuning/
├── src/
│   └── grpo_trainer.py          # GRPO implementation
├── configs/
│   └── grpo_config.yaml         # GRPO configuration
├── train_grpo.py                # Training script
├── evaluate_grpo.py             # Evaluation script
├── create_dataset.py            # Dataset utilities
└── data/
    ├── image_code_train.jsonl   # Training data
    ├── image_code_eval.jsonl    # Evaluation data
    └── image_code_test.jsonl    # Test data
```

## 🔧 GRPO Algorithm Details

### Core Components

1. **Group Generation**: Generate multiple responses per prompt (default: 4)
2. **Reward Calculation**: Evaluate each response using:
   - Code execution success
   - Image similarity metrics (SSIM, MSE, cosine similarity, etc.)
   - Custom scoring function
3. **Policy Optimization**: Use relative rewards within groups to update policy

### Reward Function

```python
total_reward = (
    code_execution_success * 0.3 +
    image_similarity_score * 0.7 +
    syntax_error_penalty +
    runtime_error_penalty
)
```

### Image Similarity Metrics

- **SSIM**: Structural similarity index
- **MSE**: Mean squared error (normalized)
- **Cosine Similarity**: Vector similarity
- **Histogram Similarity**: Color distribution comparison
- **Edge Similarity**: Edge detection comparison

## ⚙️ Configuration

### GRPO Settings
```yaml
grpo:
  group_size: 4              # Number of responses per prompt
  beta: 0.1                  # KL penalty coefficient
  gamma: 0.99                # Discount factor
  clip_ratio: 0.2            # PPO clipping ratio
```

### Reward Weights
```yaml
reward:
  code_execution_weight: 0.3
  image_similarity_weight: 0.7
  syntax_error_penalty: -1.0
  runtime_error_penalty: -0.5
```

## 📊 Dataset Format

### JSONL Format
```json
{
  "id": "sample_1",
  "description": "A simple red circle",
  "image_path": "sample_images/red_circle.png",
  "reference_code": "import matplotlib.pyplot as plt\n..."
}
```

### Base64 Format (for remote training)
```json
{
  "id": "sample_1", 
  "description": "A simple red circle",
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "reference_code": "import matplotlib.pyplot as plt\n..."
}
```

## 🛠️ Usage Examples

### Create Dataset
```bash
# Create sample dataset with matplotlib images
python create_dataset.py --create_sample

# Convert images to base64
python create_dataset.py --convert_base64 data/image_code_dataset.jsonl

# Split into train/eval/test
python create_dataset.py --split_dataset data/image_code_dataset.jsonl
```

### Training
```bash
# Basic training
python train_grpo.py --config configs/grpo_config.yaml

# Custom parameters
python train_grpo.py --config configs/grpo_config.yaml --epochs 10 --data_path data/custom_dataset.jsonl
```

### Evaluation
```bash
# Basic evaluation
python evaluate_grpo.py --model_path outputs/grpo --dataset_path data/image_code_test.jsonl

# Generate examples
python evaluate_grpo.py --model_path outputs/grpo --dataset_path data/image_code_test.jsonl --generate_examples --max_examples 20
```

## 🎨 Custom Scoring Integration

To integrate your custom scoring function, modify the `CustomRewardFunction` class:

```python
class CustomRewardFunction:
    def calculate_reward(self, generated_code: str, reference_image: Image.Image,
                        prompt: str = "") -> Dict[str, float]:
        # ... existing code ...
        
        # Add your custom scoring
        custom_score = self.your_custom_scoring_function(
            generated_code, reference_image, execution_result
        )
        
        reward_components['custom_score'] = custom_score
        reward_components['total_reward'] += custom_score * 0.2  # Adjust weight
        
        return reward_components
```

## 📈 Evaluation Metrics

The system tracks:
- **Execution Success Rate**: Percentage of code that runs without errors
- **Image Similarity Score**: Average similarity between generated and reference images
- **Overall Score**: Combined metric incorporating all factors
- **Code Quality**: Length, complexity, and best practices

## 🚨 Troubleshooting

### Common Issues

1. **Code Execution Errors**:
   - Check matplotlib installation
   - Verify code syntax
   - Review execution timeout settings

2. **Image Similarity Issues**:
   - Ensure images are same size
   - Check color space consistency
   - Verify similarity threshold settings

3. **Memory Issues**:
   - Reduce batch size
   - Use gradient accumulation
   - Enable 4-bit quantization

### Performance Tips

- Use GPU acceleration for image processing
- Enable mixed precision training (bf16)
- Use gradient checkpointing for memory efficiency
- Optimize image preprocessing pipeline

## 🔬 Research Background

GRPO is based on:
- **Group Relative Policy Optimization**: Optimizing relative performance within groups
- **Image-to-Code Generation**: Converting visual information to executable code
- **Reinforcement Learning**: Using rewards to guide policy updates
- **Multimodal Learning**: Combining vision and language understanding

## 📚 Dependencies

Key packages:
- `torch` >= 2.1.0
- `transformers` >= 4.36.0
- `matplotlib` >= 3.7.0
- `opencv-python` >= 4.8.0
- `scikit-image` >= 0.21.0
- `scikit-learn` >= 1.3.0
- `PIL` >= 9.0.0

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional image similarity metrics
- Better code execution safety
- More sophisticated reward functions
- Performance optimizations

## 📄 License

MIT License - see LICENSE file for details.
