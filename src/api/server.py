# 📄 src/api/server.py

from fastapi import FastAPI
from pydantic import BaseModel
import time

from src.model.model_loader import ModelLoader

# Initialize FastAPI app
app = FastAPI()

# Load model once at startup
model = ModelLoader()


# ----------------------------
# Request Schema
# ----------------------------
class Request(BaseModel):
    prompt: str
    max_length: int = 50
    strategy: str = "top_p"   # "greedy", "top_k", "top_p"
    use_cache: bool = True


# ----------------------------
# Health Check (Optional but professional)
# ----------------------------
@app.get("/")
def root():
    return {"message": "Mini vLLM Inference Engine is running 🚀"}


# ----------------------------
# Text Generation Endpoint
# ----------------------------
@app.post("/generate")
def generate_text(req: Request):
    start_time = time.time()   # ⏱ start timer

    output = model.generate(
        prompt=req.prompt,
        max_length=req.max_length,
        strategy=req.strategy,
        use_cache=req.use_cache
    )

    end_time = time.time()     # ⏱ end timer

    return {
        "response": output,
        "strategy": req.strategy,
        "kv_cache_enabled": req.use_cache,
        "time_taken_seconds": round(end_time - start_time, 4)
    }
