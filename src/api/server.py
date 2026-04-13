
from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.batcher import RequestBatcher
import threading
import time
from src.model.model_loader import ModelLoader


# Initialize FastAPI app
app = FastAPI()

# Load model once at startup
model = ModelLoader()

# Initialize batcher
batcher = RequestBatcher(model)

# Start background batching thread
threading.Thread(target=batcher.process_batch, daemon=True).start()


# ----------------------------
# Request Schema
# ----------------------------
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 50
    strategy: str = "top_p"
    use_cache: bool = True
    batching: bool = True   # ✅ NEW


# ----------------------------
# Health Check
# ----------------------------
@app.get("/")
def root():
    return {"message": "Mini vLLM Inference Engine is running 🚀"}


# ----------------------------
# Text Generation Endpoint
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

    # 🔥 Batching ON
    if req.batching:
        event, result = batcher.add_request(request_data)

        if not event.wait(timeout=30):
            return {"error": "Request timed out"}

        response = result.get("response", "Error: No response generated")
        batch_size = result.get("batch_size", 1)

    # 🔥 Batching OFF (direct inference)
    else:
        response = model.generate(**request_data)
        batch_size = 1

    end_time = time.time()

    return {
        "response": response,
        "strategy": req.strategy,
        "kv_cache_enabled": req.use_cache,
        "batching_enabled": req.batching,
        "batch_size": batch_size,
        "time_taken_seconds": round(end_time - start_time, 4)
    }