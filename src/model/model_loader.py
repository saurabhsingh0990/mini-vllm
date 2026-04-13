from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from src.inference.decoding import greedy_decode, top_k_decode, top_p_decode

class ModelLoader:
    def __init__(self, model_name="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

  

    def generate(self, prompt, max_length=50):
        return top_p_decode(
            self.model,
            self.tokenizer,
            prompt,
            self.device,
            max_length,
            p=0.9
        )