from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ModelLoader:
    def __init__(self, model_name="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)