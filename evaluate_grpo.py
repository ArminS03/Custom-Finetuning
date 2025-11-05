"""
Evaluation script for GRPO-trained image-to-code generation models.
"""

import os
import sys
import json
import yaml
import logging
import argparse
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import from src module
from src.grpo_trainer import ImageCodeGRPOTrainer, ImageCodeConfig, ImageCodeDataset, CodeExecutor, ImageSimilarityEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GRPOEvaluator:
    """Evaluator for GRPO-trained models."""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.tokenizer = None
        self.code_executor = CodeExecutor()
        self.image_evaluator = ImageSimilarityEvaluator()
        
    def load_model(self):
        """Load the trained model."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def generate_code(self, image: Image.Image, description: str = "") -> str:
        """Generate matplotlib code for given image."""
        
        # Create prompt
        prompt = f"""<|im_start|>user
I have an image that I want to recreate using matplotlib. Please generate Python code that creates a matplotlib figure that looks as similar as possible to the provided image.

Description: {description}

Please provide only the Python code using matplotlib, numpy, and other standard libraries. The code should be executable and produce a figure that matches the image.

<|im_start|>assistant
```python
"""
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate code
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract code from response
        if '```python' in response:
            code = response.split('```python')[1]
        elif '```' in response:
            code = response.split('```')[0]
        else:
            code = response
        
        if '```' in code:
            code = code.split('```')[0]
        
        return code.strip()
    
    def evaluate_single_sample(self, image: Image.Image, description: str = "", 
                              reference_code: str = "") -> Dict[str, Any]:
        """Evaluate a single image-to-code sample."""
        
        # Generate code
        generated_code = self.generate_code(image, description)
        
        # Execute code
        execution_result = self.code_executor.execute_code(generated_code)
        
        # Calculate metrics
        metrics = {
            'generated_code': generated_code,
            'code_length': len(generated_code),
            'execution_success': execution_result['success'],
            'execution_time': execution_result.get('execution_time', 0),
            'execution_error': execution_result.get('error'),
            'image_similarity': 0.0,
            'similarity_metrics': {}
        }
        
        if execution_result['success'] and execution_result['image'] is not None:
            # Calculate image similarity
            similarity_metrics = self.image_evaluator.calculate_similarity(
                execution_result['image'], image
            )
            metrics['image_similarity'] = similarity_metrics['combined_similarity']
            metrics['similarity_metrics'] = similarity_metrics
        
        # Calculate overall score
        if metrics['execution_success']:
            metrics['overall_score'] = metrics['image_similarity']
        else:
            metrics['overall_score'] = 0.0
        
        return metrics
    
    def evaluate_dataset(self, dataset_path: str, max_samples: int = None) -> Dict[str, Any]:
        """Evaluate model on entire dataset."""
        
        # Load dataset
        dataset = ImageCodeDataset(dataset_path, self.tokenizer)
        
        if max_samples:
            dataset.data = dataset.data[:max_samples]
        
        logger.info(f"Evaluating on {len(dataset)} samples")
        
        results = []
        total_scores = []
        execution_successes = []
        image_similarities = []
        
        for i, sample in enumerate(dataset):
            logger.info(f"Evaluating sample {i+1}/{len(dataset)}")
            
            metrics = self.evaluate_single_sample(
                sample['image'],
                sample['description'],
                sample.get('reference_code', '')
            )
            
            results.append(metrics)
            total_scores.append(metrics['overall_score'])
            execution_successes.append(metrics['execution_success'])
            image_similarities.append(metrics['image_similarity'])
        
        # Calculate aggregate metrics
        aggregate_metrics = {
            'total_samples': len(dataset),
            'execution_success_rate': np.mean(execution_successes),
            'mean_image_similarity': np.mean(image_similarities),
            'mean_overall_score': np.mean(total_scores),
            'std_overall_score': np.std(total_scores),
            'max_score': np.max(total_scores),
            'min_score': np.min(total_scores),
            'results': results
        }
        
        return aggregate_metrics
    
    def generate_examples(self, dataset_path: str, output_dir: str, 
                         max_examples: int = 10) -> List[Dict[str, Any]]:
        """Generate and save example outputs."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        dataset = ImageCodeDataset(dataset_path, self.tokenizer)
        
        if max_examples:
            dataset.data = dataset.data[:max_examples]
        
        examples = []
        
        for i, sample in enumerate(dataset):
            logger.info(f"Generating example {i+1}/{len(dataset)}")
            
            # Generate code
            generated_code = self.generate_code(sample['image'], sample['description'])
            
            # Execute code
            execution_result = self.code_executor.execute_code(generated_code)
            
            example = {
                'sample_id': sample.get('item_id', i),
                'description': sample['description'],
                'generated_code': generated_code,
                'execution_success': execution_result['success'],
                'execution_error': execution_result.get('error')
            }
            
            # Save images
            if execution_result['success'] and execution_result['image'] is not None:
                # Save original image
                original_path = os.path.join(output_dir, f"original_{i}.png")
                sample['image'].save(original_path)
                example['original_image_path'] = original_path
                
                # Save generated image
                generated_path = os.path.join(output_dir, f"generated_{i}.png")
                execution_result['image'].save(generated_path)
                example['generated_image_path'] = generated_path
                
                # Calculate similarity
                similarity_metrics = self.image_evaluator.calculate_similarity(
                    execution_result['image'], sample['image']
                )
                example['similarity_metrics'] = similarity_metrics
                example['overall_similarity'] = similarity_metrics['combined_similarity']
            
            examples.append(example)
        
        # Save examples to JSON
        examples_path = os.path.join(output_dir, "examples.json")
        with open(examples_path, 'w') as f:
            json.dump(examples, f, indent=2)
        
        logger.info(f"Examples saved to {output_dir}")
        return examples


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate GRPO Image-to-Code Model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to evaluation dataset")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--max_samples", type=int,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--max_examples", type=int, default=10,
                       help="Maximum number of examples to generate")
    parser.add_argument("--generate_examples", action="store_true",
                       help="Generate example outputs")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = GRPOEvaluator(args.model_path)
    evaluator.load_model()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluator.evaluate_dataset(args.dataset_path, args.max_samples)
    
    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples: {results['total_samples']}")
    print(f"Execution success rate: {results['execution_success_rate']:.3f}")
    print(f"Mean image similarity: {results['mean_image_similarity']:.3f}")
    print(f"Mean overall score: {results['mean_overall_score']:.3f}")
    print(f"Score std: {results['std_overall_score']:.3f}")
    print(f"Max score: {results['max_score']:.3f}")
    print(f"Min score: {results['min_score']:.3f}")
    print("="*50)
    
    # Generate examples if requested
    if args.generate_examples:
        logger.info("Generating examples...")
        examples = evaluator.generate_examples(
            args.dataset_path, 
            os.path.join(args.output_dir, "examples"),
            args.max_examples
        )
        
        print(f"\nGenerated {len(examples)} examples")
        print(f"Examples saved to {os.path.join(args.output_dir, 'examples')}")
    
    logger.info(f"Evaluation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
