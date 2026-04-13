from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from src.inference.decoding import generate_text


class ModelLoader:
    def __init__(self, model_name="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt,
        max_length=50,
        strategy="top_p",
        use_cache=True
    ):
        return generate_text(
            self.model,
            self.tokenizer,
            prompt,
            self.device,
            max_length=max_length,
            strategy=strategy,
            use_cache=use_cache
        )