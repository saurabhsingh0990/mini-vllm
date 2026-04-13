from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class ModelLoader:
    def __init__(self, model_name="gpt2", quantized=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForCausalLM.from_pretrained(model_name)

        self.quantized = False  # default

        # 🔥 Safe quantization block
        if quantized and self.device == "cpu":
            try:
                if torch.backends.quantized.engine == "none":
                    print("⚠️ Quantization not supported on this system. Falling back.")
                else:
                    print("Applying dynamic INT8 quantization...")
                    model = torch.quantization.quantize_dynamic(
                        model,
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )
                    self.quantized = True
            except Exception as e:
                print(f"⚠️ Quantization failed: {e}")
                print("Falling back to normal model.")

        self.model = model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt,
        max_length=50,
        strategy="top_p",
        use_cache=True
    ):
        from src.inference.decoding import generate_text

        return generate_text(
            self.model,
            self.tokenizer,
            prompt,
            self.device,
            max_length=max_length,
            strategy=strategy,
            use_cache=use_cache
        )