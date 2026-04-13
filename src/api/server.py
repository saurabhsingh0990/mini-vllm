from fastapi import FastAPI
from pydantic import BaseModel
import threading
import time

from src.model.model_loader import ModelLoader
from src.inference.batcher import RequestBatcher


# ----------------------------
# Initialize App
# ----------------------------
app = FastAPI()


# ----------------------------
# Load Models
# ----------------------------
model = ModelLoader(quantized=False)
quantized_model = ModelLoader(quantized=True)


# ----------------------------
# Create Batchers
# ----------------------------
normal_batcher = RequestBatcher(model)
quantized_batcher = RequestBatcher(quantized_model)


# Start batcher threads
threading.Thread(target=normal_batcher.process_batch, daemon=True).start()
threading.Thread(target=quantized_batcher.process_batch, daemon=True).start()


# ----------------------------
# Request Schema
# ----------------------------
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 50
    strategy: str = "top_p"
    use_cache: bool = True
    batching: bool = True
    quantized: bool = False   # 🔥 NEW


# ----------------------------
# Health Check
# ----------------------------
@app.get("/")
def root():
    return {"message": "Mini vLLM Inference Engine is running 🚀"}


# ----------------------------
# Generate Endpoint
# ----------------------------
@app.post("/generate")
def generate_text(req: GenerateRequest):
    start_time = time.time()

    request_data = {
        "prompt": req.prompt,
        "max_length": req.max_length,
        "strategy": req.strategy,
        "use_cache": req.use_cache
    }

    # 🔥 Select model/batcher
    batcher = quantized_batcher if req.quantized else normal_batcher
    model_instance = quantized_model if req.quantized else model

    # 🔹 Batching enabled
    if req.batching:
        event, result = batcher.add_request(request_data)

        if not event.wait(timeout=30):
            return {"error": "Request timed out"}

        response = result.get("response", "Error: No response generated")
        batch_size = result.get("batch_size", 1)

    # 🔹 Batching disabled
    else:
        response = model_instance.generate(**request_data)
        batch_size = 1

    end_time = time.time()

    return {
        "response": response,
        "strategy": req.strategy,
        "kv_cache_enabled": req.use_cache,
        "batching_enabled": req.batching,
        "quantized": model_instance.quantized,
        "batch_size": batch_size,
        "time_taken_seconds": round(end_time - start_time, 4)
    }