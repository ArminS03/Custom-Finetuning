#!/usr/bin/env python3
import argparse
from PIL import Image
import torch

try:
    from transformers import AutoImageProcessor, AutoModel
except Exception as e:
    raise SystemExit(f"transformers is required: {e}")

DINO_MODEL_ID = "facebook/dinov2-base"

class DINOEvaluator:
    """Efficient DINO model evaluator with caching."""
    
    def __init__(self, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self._initialized = False
    
    def _ensure_initialized(self):
        if not self._initialized:
            self.processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
            self.model = AutoModel.from_pretrained(DINO_MODEL_ID).to(self.device).eval()
            self._initialized = True
    
    def score(self, img_a: Image.Image, img_b: Image.Image) -> float:
        """Compute DINO similarity between two PIL Image objects (0..1)."""
        self._ensure_initialized()
        
        with torch.no_grad():
            # Encode A
            inputs_a = self.processor(images=img_a.convert("RGB"), return_tensors="pt").to(self.device)
            out_a = self.model(**inputs_a)
            feat_a = out_a.last_hidden_state.mean(dim=1)  # mean-pool tokens
            
            # Encode B
            inputs_b = self.processor(images=img_b.convert("RGB"), return_tensors="pt").to(self.device)
            out_b = self.model(**inputs_b)
            feat_b = out_b.last_hidden_state.mean(dim=1)
            
            # Cosine similarity in [-1, 1]
            cos_sim = torch.nn.functional.cosine_similarity(feat_a, feat_b).item()
        
        # Map to [0, 1] for a clean "score"
        sim_01 = (cos_sim + 1.0) / 2.0
        return float(sim_01)
    
    def score_paths(self, img_a_path: str, img_b_path: str) -> float:
        """Compute DINO similarity between two image paths (0..1)."""
        img_a = Image.open(img_a_path).convert("RGB")
        img_b = Image.open(img_b_path).convert("RGB")
        return self.score(img_a, img_b)


# Global evaluator instance for backward compatibility
_evaluator = None

def dino_score(img_a_path: str, img_b_path: str, device: str) -> float:
    """Compute DINO similarity between two image paths (0..1)."""
    global _evaluator
    if _evaluator is None:
        _evaluator = DINOEvaluator(device)
    return _evaluator.score_paths(img_a_path, img_b_path)

def main():
    parser = argparse.ArgumentParser(description="Compute DINO similarity score between two images (0..1).")
    parser.add_argument("image_a", type=str)
    parser.add_argument("image_b", type=str)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    score = dino_score(args.image_a, args.image_b, args.device)
    # Print just the number (no extra text)
    print(f"{score:.6f}")

if __name__ == "__main__":
    main()
