"""FastAPI inference server with dynamic batching and model quantization.

Provides REST API endpoints for text generation with optional batching and quantization.
The server loads both standard and quantized models on startup and maintains separate
batchers for each model variant to handle concurrent inference efficiently.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import threading
import time
from typing import Dict, Any

from src.model.model_loader import ModelLoader
from src.inference.batcher import RequestBatcher


# Initialize FastAPI app with metadata
app = FastAPI(
    title="Mini vLLM Inference API",
    description="High-performance LLM inference with batching, quantization, and KV cache",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ----------------------------
# Model Loading (Singleton Pattern)
# ----------------------------
# Load models once at startup - reused across all requests
model = ModelLoader(quantized=False)
quantized_model = ModelLoader(quantized=True)


# ----------------------------
# Batcher Initialization
# ----------------------------
# Separate batchers for each model variant to manage load independently
normal_batcher = RequestBatcher(model, batch_size=8, wait_time=0.2)
quantized_batcher = RequestBatcher(quantized_model, batch_size=8, wait_time=0.2)

# Start background worker threads (daemon=True ensures they don't block server shutdown)
threading.Thread(target=normal_batcher.process_batch, daemon=True).start()
threading.Thread(target=quantized_batcher.process_batch, daemon=True).start()

print("[Server] Batchers initialized and workers started")



# ----------------------------
# Request Schema
# ----------------------------
class GenerateRequest(BaseModel):
    """Schema for text generation requests.
    
    Attributes:
        prompt: Input text to complete or generate from.
        max_length: Maximum number of tokens to generate (default: 50).
        strategy: Token selection strategy - 'greedy', 'top_k', or 'top_p' (default: 'top_p').
        use_cache: Enable KV cache optimization to reduce recomputation (default: True).
        batching: Enable dynamic request batching for better throughput (default: True).
        quantized: Use INT8 quantized model for reduced memory (default: False).
    """
    prompt: str
    max_length: int = 50
    strategy: str = "top_p"
    use_cache: bool = True
    batching: bool = True
    quantized: bool = False

# ----------------------------
# Health Check Endpoint
# ----------------------------
@app.get("/")
def root() -> Dict[str, str]:
    """Simple health check endpoint.
    
    Returns:
        Dict with status message indicating server is operational.
    """
    return {"message": "Mini vLLM Inference Engine is running 🚀"}


# ----------------------------
# Text Generation Endpoint
# ----------------------------
@app.post("/generate")
def generate_text(req: GenerateRequest) -> Dict[str, Any]:
    """Generate text based on a prompt with configurable inference options.
    
    This endpoint supports two execution modes:
    - Batching ON: Request is queued and processed with others (improved throughput, higher latency)
    - Batching OFF: Request bypasses queue for immediate inference (lower latency, lower throughput)
    
    Args:
        req: GenerateRequest with prompt, strategy, and optimization flags.
        
    Returns:
        Dict containing:
        - response: Generated text
        - strategy: Decoding strategy used
        - kv_cache_enabled: Whether KV cache optimization was applied
        - batching_enabled: Whether batching was used
        - quantized: Whether quantized model was used
        - batch_size: Number of requests processed together
        - time_taken_seconds: Total end-to-end latency
    """
    start_time = time.time()

    # Prepare request data for model
    request_data = {
        "prompt": req.prompt,
        "max_length": req.max_length,
        "strategy": req.strategy,
        "use_cache": req.use_cache
    }

    # Select appropriate model and batcher based on quantization flag
    batcher = quantized_batcher if req.quantized else normal_batcher
    model_instance = quantized_model if req.quantized else model

    # Path 1: Batching enabled - join request queue
    # Requests arriving within the time window are batched together
    if req.batching:
        event, result = batcher.add_request(request_data)

        # Wait for result with timeout protection
        if not event.wait(timeout=30):
            return {"error": "Request timed out in batching queue"}

        response = result.get("response", "Error: No response generated")
        batch_size = result.get("batch_size", 1)

    # Path 2: Direct inference - skip queue for lower latency
    # Useful for latency-critical single requests
    else:
        response = model_instance.generate(**request_data)
        batch_size = 1

    end_time = time.time()

    # Return comprehensive response with metrics
    return {
        "response": response,
        "strategy": req.strategy,
        "kv_cache_enabled": req.use_cache,
        "batching_enabled": req.batching,
        "quantized": model_instance.quantized,
        "batch_size": batch_size,
        "time_taken_seconds": round(end_time - start_time, 4)
    }