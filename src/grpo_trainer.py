"""
GRPO implementation using TRL's built-in GRPOTrainer for image-to-code generation.
"""
import uuid
import os
import json
import logging
from typing import Dict, Any
from dataclasses import dataclass
from PIL import Image
import base64
from datasets import Dataset
import multiprocessing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .lora_config import DEFAULT_LORA_CONFIG
from .image_compare import DINOEvaluator
# TRL imports
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import get_peft_model
from trl import GRPOTrainer, GRPOConfig

logger = logging.getLogger(__name__)


@dataclass
class ImageCodeConfig:
    """Configuration for image-to-code generation training."""
    # Model settings
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    max_seq_length: int = 16384
    max_completion_length: int = 4096
    dtype: str = "bf16"
    load_in_4bit: bool = False
    
    # Output directory
    output_dir: str = "./output/Qwen3_VL-v1"
    
    # GRPO specific settings
    group_size: int = 4         # Number of responses per prompt
    
    # Training settings
    learning_rate: float = 1e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    
    # Reward settings
    code_execution_weight: float = 0.3
    image_similarity_weight: float = 1
    syntax_error_penalty: float = -1.0
    runtime_error_penalty: float = -1.0
    
    # DeepSpeed settings
    deepspeed_config_file: str = None


def load_and_process_dataset(data_path: str, processor):
    prompts_file = os.path.join(data_path, "prompts.json")
    with open(prompts_file, "r") as f:
        raw_list = json.load(f)
    logger.info(f"Loaded {len(raw_list)} samples from {prompts_file}")
    
    # Convert to HF dataset
    hf_ds = Dataset.from_list(raw_list)
    
    def process_example(example):
        messages = example["messages"]
        image_rel = example["images"][0] if example.get("images") else None
        user_msg = next((m for m in messages if m["role"] == "user"), None)
        prompt = user_msg["content"].replace("<image>\n", "").strip() if user_msg else ""
        image_rel = "images/" + image_rel.split("/")[-1] if image_rel else None
        image_path = os.path.join(data_path, image_rel) if image_rel else None

        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        inputs = [
            {
                "role": "user",
                "content": [ 
                    {
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            inputs,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        return {
            "prompt": inputs,
            "reference_image": image_path,
            "item_id": example.get("item_id", None)
        }
    
    hf_ds = hf_ds.map(process_example, remove_columns=list(hf_ds.column_names))
    
    return hf_ds

class CodeExecutor:
    def __init__(self, timeout: int = 30, temp_dir: str = "./temp_outputs"):
        self.timeout = timeout
        os.makedirs(temp_dir, exist_ok=True)
        self.temp_dir = temp_dir

    def _make_temp_path(self) -> str:
        return os.path.join(self.temp_dir, f"plot_{uuid.uuid4().hex}.png")
    
    def _worker(self, code, success_flag, image_path):
        """Worker function that runs in a separate process to execute plotting code."""
        try:
            # Execute user's code
            exec(code)
            
            fig = plt.gcf()
            if fig.get_axes():
                # Save the figure
                fig.savefig(image_path)
                # If file exists, mark success
                if os.path.exists(image_path):
                    success_flag.value = True
                else:
                    success_flag.value = False
            else:
                success_flag.value = False
        except (SyntaxError, IndentationError) as e:
            # Syntax problems, cannot run
            success_flag.value = False
            # Raise here or just let it end? We set flag; parent will inspect
        except Exception as e:
            # Other runtime errors
            success_flag.value = False
        finally:
            try:
                plt.close()
                matplotlib.rcdefaults()
                plt.cla()
                plt.clf()
                plt.close("all")
            except Exception:
                pass
    
    def execute_code(self, code: str) -> Dict[str, Any]:
        code = self._clean_code(code)
        # Shared flag to communicate success
        success_flag = multiprocessing.Value("b", False)
        image_path = self._make_temp_path()
        
        proc = multiprocessing.Process(target=self._worker, args=(code, success_flag, image_path))
        proc.start()
        proc.join(self.timeout)
        
        if proc.is_alive():
            proc.terminate()
            proc.join()
            error_type = "timeout"
            error = f"Execution exceeded timeout {self.timeout} seconds"
            # Clean up any partial file
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except Exception:
                    pass
            return {
                "success": False,
                "image": None,
                "error_type": error_type,
                "error": error
            }
        
        # At this point the process has ended
        if success_flag.value:
            # The worker believed it succeeded; check file
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path).convert("RGB")
                    try:
                        os.remove(image_path)
                    except Exception:
                        pass
                    return {
                        "success": True,
                        "image": img,
                        "error_type": None,
                        "error": None
                    }
                except Exception as e:
                    error_type = "runtime"
                    error = f"Failed to load image: {e}"
                    return {
                        "success": False,
                        "image": None,
                        "error_type": error_type,
                        "error": error
                    }
            else:
                error_type = "no_image"
                error = "No image file generated by code"
                return {
                    "success": False,
                    "image": None,
                    "error_type": error_type,
                    "error": error
                }
        else:
            error_type = "runtime"
            error = "Code execution did not succeed (worker flagged false)"
            return {
                "success": False,
                "image": None,
                "error_type": error_type,
                "error": error
            }

    
    def _clean_code(self, code: str) -> str:
        """Clean and prepare code for execution."""
        # Remove markdown code blocks
        if '```python' in code:
            code = code.split('```python')[1]
            if '```' in code:
                code = code.split('```')[0]
        
        return code.strip()


class ImageSimilarityEvaluator:
    """Evaluate similarity between generated and reference images using DINO."""
    
    def __init__(self):
        self.dino_evaluator = DINOEvaluator()
    
    def calculate_similarity(self, generated_image: Image.Image, 
                           reference_image: Image.Image) -> Dict[str, float]:
        """Calculate similarity between images using DINO model."""
        similarity = self.dino_evaluator.score(generated_image, reference_image)
        return {
            'dino_similarity': similarity,
            'combined_similarity': similarity
        }


class CustomRewardFunction:
    """Custom reward function for image-to-code generation."""
    
    def __init__(self, config: ImageCodeConfig):
        self.config = config
        self.code_executor = CodeExecutor()
        self.image_evaluator = ImageSimilarityEvaluator()
    
    def calculate_reward(self, generated_code: str, reference_image: Image.Image) -> Dict[str, float]:
        """Calculate reward for generated code."""
        
        # Execute the generated code
        execution_result = self.code_executor.execute_code(generated_code)
        
        rc = {
            "code_execution": 0.0,
            "image_similarity": 0.0,
            "syntax_penalty": 0.0,
            "runtime_penalty": 0.0,
            "total_reward": 0.0,
        }

        if execution_result["success"]:
            rc["code_execution"] = 1.0
            gen_img = execution_result.get("image")
            if gen_img is not None:
                sim = self.image_evaluator.calculate_similarity(gen_img, reference_image)
                rc["image_similarity"] = sim["combined_similarity"]
            else:
                rc["code_execution"] = -0.3
        else:
            et = execution_result.get("error_type")
            if et == "syntax":
                rc["syntax_penalty"] = self.config.syntax_error_penalty
            elif et == "runtime":
                rc["runtime_penalty"] = self.config.runtime_error_penalty
            else:
                rc["runtime_penalty"] = self.config.runtime_error_penalty

        total = (
            rc["code_execution"] * self.config.code_execution_weight
            + rc["image_similarity"] * self.config.image_similarity_weight
            + rc["syntax_penalty"]
            + rc["runtime_penalty"]
        )
        rc["total_reward"] = total
        return rc


class CustomGRPOTrainer:
    """GRPO trainer using TRL's built-in GRPOTrainer."""
    
    def __init__(self, config: ImageCodeConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.reward_function = CustomRewardFunction(config)
        self.grpo_trainer = None
    
    def setup_model(self):
        """Setup model and processor."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            use_fast=True,
            trust_remote_code=True,
        )

        # model, processor = FastVisionModel.from_pretrained(
        #     "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
        #     max_seq_length = int(self.config.max_seq_length),
        #     fast_inference = False,
        #     device_map="auto",
        #     gpu_memory_utilization = 0.8,
        #     use_gradient_checkpointing = "unsloth",
        # )
        # self.model = model
        # self.processor = processor
        
        # Load model
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            dtype=self.config.dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        
        # Apply LoRA
        lora_config = DEFAULT_LORA_CONFIG
        
        # self.model = FastVisionModel.get_peft_model(
        #     self.model,
        #     finetune_vision_layers     = False,
        #     finetune_language_layers   = True,
        #     finetune_attention_modules = True,
        #     finetune_mlp_modules       = True,

        #     r=16,                       # Rank
        #     lora_alpha=32,              # Scaling factor (2x rank)
        #     lora_dropout=0.0,           # Dropout rate
        #     bias="none",                # Don't adapt bias
        #     target_modules=[
        #         "q_proj", 
        #         "k_proj", 
        #         "v_proj", 
        #         "o_proj", 
        #         "gate_proj", 
        #         "up_proj",
        #         "down_proj"
        #     ],
        # )
        
        
        self.model = get_peft_model(self.model, lora_config)

        self.model.print_trainable_parameters()
        
        self.model.train()

        # FastVisionModel.for_training(
        #     self.model,
        # )

        logger.info("Model setup completed")
        

        # if not hasattr(self, "_timing_tracker"):
        #     self._timing_tracker = TimingTracker()
        # original_generate = self.model.generate
        # tracker = self._timing_tracker
        # def timed_generate(*args, **kwargs):
        #     start = _now()
        #     try:
        #         return original_generate(*args, **kwargs)
        #     finally:
        #         tracker.record_generation_time(_now() - start)
        # self.model.generate = timed_generate
    
    def create_reward_function(self):
        """Create reward function compatible with TRL's GRPOTrainer."""

        def reward_func(completions, **kwargs):
            rewards = []
            # tracker = getattr(self, "_timing_tracker", None)
            
            # It could be a single image or a list of images
            reference_images = kwargs.get('reference_image', [])
            if not isinstance(reference_images, list):
                reference_images = [reference_images]
            
            # Ensure we have the same number of completions and reference images
            min_len = min(len(completions), len(reference_images))
            
            for i in range(min_len):
                completion = completions[i]
                reference_image = reference_images[i]
                
                # Load reference image if it's a path
                if isinstance(reference_image, str):
                    if os.path.exists(reference_image):
                        ref_img = Image.open(reference_image)
                    else:
                        logger.warning(f"Reference image not found: {reference_image}")
                        rewards.append(0.0)
                        continue
                elif isinstance(reference_image, Image.Image):
                    ref_img = reference_image
                else:
                    logger.warning(f"Invalid reference image type: {type(reference_image)}")
                    rewards.append(0.0)
                    continue
                
                # Calculate reward
                # _t0 = _now() if tracker is not None else None
                reward_dict = self.reward_function.calculate_reward(completion, ref_img)
                # if tracker is not None and _t0 is not None:
                #     tracker.record_reward_time(_now() - _t0)

                rewards.append(reward_dict['total_reward'])
            
            # Pad with zeros if necessary
            while len(rewards) < len(completions):
                rewards.append(0.0)
            print(f"Mean rewards: {sum(rewards) / len(rewards)}")
            return rewards
        
        return reward_func
    
    def train(self, dataset: Dataset, num_epochs: int = None):
        """Train using TRL's GRPOTrainer."""
        if self.model is None:
            self.setup_model()
        
        epochs = num_epochs or int(self.config.num_epochs)
        
        # splits = dataset.train_test_split(test_size=0.05, seed=42)
        # train_ds = splits["train"]
        # eval_ds  = splits["test"]
        # Create GRPO config
        
        # Prepare DeepSpeed config if provided
        print(f"\nnum_train_epochs:{epochs}")
        print(f"per_device_train_batch_size:{int(self.config.batch_size)}")
        print(f"gradient_accumulation_steps:{int(self.config.gradient_accumulation_steps)}")

        print(f"num_generations:{int(self.config.group_size)}")
        print(f"max_prompt_length:{int(self.config.max_seq_length)}")
        print(f"max_completion_length:{int(self.config.max_completion_length)}")
        print(f"learning_rate:{float(self.config.learning_rate)}\n")

        grpo_config_kwargs = {
            "output_dir": self.config.output_dir,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": int(self.config.batch_size),
            "gradient_accumulation_steps": int(self.config.gradient_accumulation_steps),
            "learning_rate": float(self.config.learning_rate),
            "weight_decay": 0.1,
            "warmup_ratio": 0.1,

            "logging_steps": 5,
            "num_generations": int(self.config.group_size),
            "max_prompt_length": int(self.config.max_seq_length),
            "max_completion_length": int(self.config.max_completion_length),
            "temperature": 1.0,
            "save_steps": 100,
            "eval_steps": 100,
            "bf16": True,
            "gradient_checkpointing": True,
            # "use_vllm": True,
        }
        
        grpo_config = GRPOConfig(**grpo_config_kwargs)
        
        reward_func = self.create_reward_function()
        
        self.grpo_trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.processor,
            args=grpo_config,   
            train_dataset=dataset,
            # eval_dataset=eval_ds, 
            reward_funcs=reward_func,
        )
        
        logger.info(f"Starting GRPO training for {epochs} epochs")
        
        self.grpo_trainer.train()
        
        logger.info("GRPO training completed")


# # ---------- Timing utilities ----------
# import time

# def _now() -> float:
#     try:
#         return time.perf_counter()
#     except Exception:
#         return time.time()


# class TimingTracker:
#     """Tracks durations for generation, reward evaluation, and total train step."""
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.generation_time_total = 0.0
#         self.reward_time_total = 0.0
#         self.step_time_total = 0.0
#         self.steps = 0
#         self._last_step_start = None
#         self._last_gen = 0.0
#         self._last_reward = 0.0

#     def step_begin(self):
#         self._last_step_start = _now()
#         self._last_gen = 0.0
#         self._last_reward = 0.0

#     def record_generation_time(self, seconds: float):
#         self.generation_time_total += seconds
#         self._last_gen += seconds

#     def record_reward_time(self, seconds: float):
#         self.reward_time_total += seconds
#         self._last_reward += seconds

#     def step_end(self):
#         if self._last_step_start is not None:
#             self.step_time_total += (_now() - self._last_step_start)
#             self.steps += 1
#             self._last_step_start = None

#     def current_averages(self):
#         denom = max(self.steps, 1)
#         return {
#             "timing/avg_generation_s": self.generation_time_total / denom,
#             "timing/avg_reward_eval_s": self.reward_time_total / denom,
#             "timing/avg_step_s": self.step_time_total / denom,
#             # Derived non-generation compute (incl. backward/optimizer, misc)
#             "timing/avg_non_gen_s": max((self.step_time_total - self.generation_time_total), 0.0) / denom,
#         }


# class TrainingTimingCallback(TrainerCallback):
#     """Adds fine-grained timing logs for GRPO steps."""
#     def __init__(self, tracker: TimingTracker, log_every: int = 1):
#         self.tracker = tracker
#         self.log_every = max(int(log_every), 1)

#     def on_step_begin(self, args, state, control, **kwargs):
#         self.tracker.step_begin()

#     def on_step_end(self, args, state, control, **kwargs):
#         self.tracker.step_end()
#         if state.global_step % self.log_every == 0:
#             avgs = self.tracker.current_averages()
#             # Log to stdout
#             logger.info(
#                 f"Timing (avg over {self.tracker.steps} steps) — "
#                 f"gen: {avgs['timing/avg_generation_s']:.3f}s, "
#                 f"reward: {avgs['timing/avg_reward_eval_s']:.3f}s, "
#                 f"non_gen: {avgs['timing/avg_non_gen_s']:.3f}s, "
#                 f"step: {avgs['timing/avg_step_s']:.3f}s"
#             )
#             # Also integrate with Trainer logging backends
#             if hasattr(self, "trainer") and self.trainer is not None:
#                 self.trainer.log(avgs)
