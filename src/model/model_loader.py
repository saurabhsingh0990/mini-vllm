"""Model loading and initialization module.

Handles model and tokenizer loading from HuggingFace, device placement,
and inference interface. Supports optional INT8 quantization for memory efficiency.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from src.inference.decoding import generate_text
from typing import Optional


class ModelLoader:
    """Loads and manages LLM models with device placement and quantization.
    
    Handles:
    - Model and tokenizer loading from HuggingFace Hub
    - Automatic device selection (CUDA/CPU)
    - Model evaluation mode setup
    - Optional INT8 dynamic quantization with fallback
    """
    
    def __init__(self, model_name: str = "gpt2", quantized: bool = False):
        """Initialize and load the model.
        
        Args:
            model_name: HuggingFace model identifier (default: 'gpt2').
            quantized: Whether to attempt INT8 quantization (default: False).
                      Falls back gracefully if not supported on platform.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        # Load tokenizer and model from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Quantization flag - may be disabled if not supported
        self.quantized = False

        # Attempt INT8 quantization (only on CPU backends with fbgemm support)
        if quantized and self.device == "cpu":
            try:
                # Check if quantization engine is available
                if torch.backends.quantized.engine == "none":
                    print("[Model] Quantization engine not available, using standard model")
                else:
                    print("[Model] Applying INT8 dynamic quantization...")
                    model = torch.quantization.quantize_dynamic(
                        model,
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )
                    self.quantized = True
            except Exception as e:
                # Graceful fallback: continue with non-quantized model
                print(f"[Model] Quantization failed: {e}")
                print("[Model] Falling back to standard model")

        # Place model on device and set to evaluation mode
        self.model = model.to(self.device)
        self.model.eval()
        
        print(f"[Model] Loaded {model_name} on {self.device} (quantized={self.quantized})")

    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        strategy: str = "top_p",
        use_cache: bool = True
    ) -> str:
        """Generate text given a prompt.
        
        Delegates to the decoding module which handles token-by-token generation
        with support for multiple decoding strategies and KV cache optimization.
        
        Args:
            prompt: Input text to complete.
            max_length: Maximum number of tokens to generate.
            strategy: Token selection strategy ('greedy', 'top_k', 'top_p').
            use_cache: Enable KV cache to reduce redundant computation.
            
        Returns:
            Generated text string.
        """
        return generate_text(
            self.model,
            self.tokenizer,
            prompt,
            self.device,
            max_length=max_length,
            strategy=strategy,
            use_cache=use_cache
        )