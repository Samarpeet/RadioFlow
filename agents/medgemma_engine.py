"""
MedGemma Engine - Unified interface for MedGemma inference
Supports both MLX (local Mac) and Transformers (GPU/CPU)
"""

import os
import time
from typing import Optional, Dict, Any

# Detect available backends
MLX_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

try:
    from mlx_lm import load, generate
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


class MedGemmaEngine:
    """
    Unified MedGemma inference engine.
    Automatically selects the best available backend:
    - MLX for Apple Silicon (M1/M2/M3/M4) - preferred locally
    - Transformers + CUDA for NVIDIA GPUs (HuggingFace Spaces)
    - Transformers + CPU as fallback
    """
    
    # Model configurations
    MLX_MODEL = "mlx-community/medgemma-4b-it-4bit"
    HF_MODEL = "google/medgemma-4b-it"
    
    def __init__(self, prefer_mlx: bool = None, force_demo: bool = False):
        # Auto-detect best backend preference
        # On HuggingFace Spaces, prefer transformers (MLX won't work)
        import os
        is_spaces = os.environ.get("SPACE_ID") is not None
        
        if prefer_mlx is None:
            prefer_mlx = not is_spaces  # Prefer MLX locally, transformers on Spaces
        self.model = None
        self.tokenizer = None
        self.backend = None
        self.is_loaded = False
        self.force_demo = force_demo
        self.prefer_mlx = prefer_mlx
        
        if force_demo:
            self.backend = "demo"
            self.is_loaded = True
            print("⚠️ MedGemma running in DEMO mode (no real inference)")
        
    def load(self) -> bool:
        """Load the model using the best available backend."""
        if self.force_demo:
            return True
            
        if self.is_loaded:
            return True
        
        # Try MLX first (best for Mac)
        if self.prefer_mlx and MLX_AVAILABLE:
            try:
                print(f"🔄 Loading MedGemma with MLX ({self.MLX_MODEL})...")
                start = time.time()
                self.model, self.tokenizer = load(self.MLX_MODEL)
                self.backend = "mlx"
                self.is_loaded = True
                print(f"✅ MedGemma loaded with MLX in {time.time()-start:.1f}s")
                return True
            except Exception as e:
                print(f"⚠️ MLX loading failed: {e}")
        
        # Try Transformers with GPU
        if TRANSFORMERS_AVAILABLE:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"🔄 Loading MedGemma with Transformers on {device}...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.HF_MODEL, 
                    trust_remote_code=True
                )
                
                if device == "cuda":
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.HF_MODEL,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.HF_MODEL,
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                    )
                
                self.backend = f"transformers-{device}"
                self.is_loaded = True
                print(f"✅ MedGemma loaded with Transformers ({device})")
                return True
                
            except Exception as e:
                print(f"⚠️ Transformers loading failed: {e}")
        
        # Fallback to demo mode
        print("⚠️ No model backend available - using demo mode")
        self.backend = "demo"
        self.is_loaded = True
        return True
    
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate a response from MedGemma."""
        if not self.is_loaded:
            self.load()
        
        if self.backend == "demo":
            return self._demo_response(prompt)
        
        try:
            if self.backend == "mlx":
                return self._generate_mlx(prompt, max_tokens)
            else:
                return self._generate_transformers(prompt, max_tokens)
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            return self._demo_response(prompt)
    
    def _generate_mlx(self, prompt: str, max_tokens: int) -> str:
        """Generate using MLX backend."""
        response = generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=max_tokens,
            verbose=False
        )
        # Clean up the response (remove the prompt if echoed)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        # Clean up repetitive disclaimers and truncated text
        response = self._clean_output(response)
        return response
    
    def _clean_output(self, text: str) -> str:
        """Remove repetitive disclaimers and clean up model output."""
        import re
        
        # Split into lines
        lines = text.split('\n')
        seen_lines = set()
        cleaned_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            # Skip empty lines after disclaimer section starts
            if not line_stripped:
                if cleaned_lines and 'Disclaimer' not in cleaned_lines[-1]:
                    cleaned_lines.append(line)
                continue
            
            # Skip repetitive disclaimer lines
            if line_stripped.startswith('**Disclaimer:**') or line_stripped.startswith('Disclaimer:'):
                if line_stripped in seen_lines:
                    continue  # Skip duplicate
                seen_lines.add(line_stripped)
                # Only keep the first disclaimer
                if len([l for l in cleaned_lines if 'Disclaimer' in l]) >= 1:
                    continue
            
            # Skip lines that are just "[Your" or similar incomplete placeholders
            if line_stripped in ['[Your', '[Date', '[Your Institution]', '[Date of Report]']:
                continue
                
            cleaned_lines.append(line)
        
        # Join and truncate at any obvious repetition
        result = '\n'.join(cleaned_lines)
        
        # Remove everything after second "Disclaimer" if present
        disclaimer_count = 0
        final_result = []
        for line in result.split('\n'):
            if 'Disclaimer' in line:
                disclaimer_count += 1
                if disclaimer_count > 1:
                    break
            final_result.append(line)
        
        return '\n'.join(final_result).strip()
    
    def _generate_transformers(self, prompt: str, max_tokens: int) -> str:
        """Generate using Transformers backend."""
        import torch
        
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        
        attention_mask = torch.ones_like(inputs)
        
        if hasattr(self.model, 'device'):
            inputs = inputs.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:], 
            skip_special_tokens=True
        )
        return response.strip()
    
    def _demo_response(self, prompt: str) -> str:
        """Fallback demo responses when no model is available."""
        prompt_lower = prompt.lower()
        
        if "interpret" in prompt_lower or "finding" in prompt_lower:
            return "Based on the imaging findings, clinical correlation is recommended. The described abnormality may represent an infectious, inflammatory, or neoplastic process. Further workup including laboratory studies and clinical examination would be beneficial for definitive diagnosis."
        
        elif "report" in prompt_lower or "generate" in prompt_lower:
            return """FINDINGS:
The visualized structures are assessed. Any noted abnormalities are described with their location, size, and characteristics.

IMPRESSION:
1. Findings as described above.
2. Clinical correlation recommended.

RECOMMENDATIONS:
Follow-up imaging as clinically indicated."""
        
        elif "priority" in prompt_lower or "urgent" in prompt_lower:
            return "PRIORITY LEVEL: ROUTINE. Based on the findings, this case does not require immediate attention but should be reviewed in standard workflow timeframe. Clinical correlation with patient symptoms is recommended."
        
        else:
            return "Clinical correlation recommended. Please consult with a radiologist for definitive interpretation."
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "is_loaded": self.is_loaded,
            "backend": self.backend,
            "mlx_available": MLX_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "model_name": self.MLX_MODEL if self.backend == "mlx" else self.HF_MODEL
        }


# Global engine instance
_engine: Optional[MedGemmaEngine] = None


def get_engine(force_demo: bool = False) -> MedGemmaEngine:
    """Get or create the global MedGemma engine."""
    global _engine
    if _engine is None:
        _engine = MedGemmaEngine(force_demo=force_demo)
        _engine.load()
    return _engine


def generate_response(prompt: str, max_tokens: int = 256) -> str:
    """Convenience function to generate a response."""
    engine = get_engine()
    return engine.generate(prompt, max_tokens)
