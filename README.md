# 🚀 Mini vLLM: High-Performance LLM Inference Engine

---

## 📌 Overview

Mini vLLM is a **production-inspired LLM inference engine** focused on understanding and implementing system-level optimizations for efficient text generation.

Unlike typical LLM projects that rely on high-level APIs, this project builds **core inference logic from scratch**, giving full control over how tokens are generated and optimized.

---

## 🎯 Objectives

* Build a **mini version of modern inference systems (like vLLM)**
* Implement **custom decoding strategies**
* Optimize:

  * Latency
  * Throughput
  * Memory usage
* Benchmark different inference techniques

---

## 🧠 Key Features

### ✅ Custom Decoding Engine (Implemented)

* Greedy Decoding
* Top-K Sampling
* Top-P (Nucleus) Sampling

👉 Fully replaces HuggingFace `.generate()` with manual token-by-token generation.

---

### ⚡ KV Cache Optimization (Implemented)

* Uses `past_key_values` to avoid recomputation
* Processes only the latest token after first step
* Significantly reduces inference latency
* Works across **all decoding strategies (Greedy / Top-K / Top-P)**

### 🚀 Dynamic Batching (Implemented)

- Groups multiple incoming requests within a short time window
- Improves system throughput under concurrent load
- Configurable:
  - batch size
  - batching window (`wait_time`)
- Demonstrates trade-off between:
  - latency (increases)
  - throughput (improves)

---

### ⏱️ Latency Measurement (Implemented)

* End-to-end response time measured at API layer
* Enables comparison of:

  * KV Cache ON vs OFF
  * Different decoding strategies

---

### ⚡ Upcoming Features

* Dynamic Batching
* Quantized Inference (INT8 / 4-bit)
* Streaming API (token-by-token response)
* SSD-based offloading (optional)

---

## 🏗️ System Architecture

```
User Requests
     ↓
API Layer (FastAPI)
     ↓
Batching Layer (Request Queue + Worker)
     ↓
Inference Engine
     ├── KV Cache Layer
     └── Decoding Strategy (Greedy / Top-K / Top-P)
     ↓
LLM Model (GPT-2)
     ↓
Generated Output
```

---

## 📂 Project Structure

```
mini-vllm/
│
├── src/
│   ├── model/
│   │   └── model_loader.py
│
│   ├── inference/
│   │   └── decoding.py       # Unified decoding + KV cache engine
│
│   ├── api/
│   │   └── server.py         # FastAPI server
│
├── run.py
├── requirements.txt
├── README.md
├── report.md                # Project report (work in progress)
└── .gitignore
```

---

## ⚙️ Prerequisites

### 🖥️ Hardware

* Minimum: CPU
* Recommended: GPU (for faster inference)

---

### 🧰 Software

* Python 3.9+
* PyTorch
* Transformers
* FastAPI
* Uvicorn

---

## 🔧 Installation

```bash
git clone https://github.com/your-username/mini-vllm.git
cd mini-vllm

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## ▶️ Usage

### Run server:

```bash
python run.py
```

---

### Open API docs:

```
http://127.0.0.1:8000/docs
```

---

## 🧪 Example Request

```bash
curl -X POST http://127.0.0.1:8000/generate \
-H "Content-Type: application/json" \
-d '{
  "prompt": "the elephant was sitting on the",
  "max_length": 50,
  "strategy": "top_p",
  "use_cache": true
}'
```

---

## 📥 Example Response

```json
{
  "response": "...",
  "strategy": "top_p",
  "kv_cache_enabled": true,
  "batching_enabled": true,
  "batch_size": 4,
  "time_taken_seconds": 0.7421
}
```

---

## 📊 Decoding Comparison (Current Results)

| Method | Behavior                       |
| ------ | ------------------------------ |
| Greedy | Repetitive, degenerate loops ❌ |
| Top-K  | Improved diversity ⚠️          |
| Top-P  | Natural, coherent, dynamic ✅   |

---

## 🧠 Key Learnings

* Greedy decoding leads to **mode collapse**
* Sampling-based methods improve **diversity and coherence**
* KV caching reduces **redundant computation** and improves latency
* Token-by-token control is essential for:

  * KV cache
  * batching
  * real-world inference systems

---

## 📅 Roadmap

- [x] Basic inference engine  
- [x] Custom decoding module  
- [x] KV cache optimization  
- [x] Dynamic batching  
- [ ] Quantization  
- [ ] Benchmarking & evaluation  

---

## 📚 References

* Attention is All You Need — Vaswani et al. (2017)
* vLLM: Efficient Memory Management for LLM Serving — Kwon et al. (2023)

---

## 💼 Project Motivation

This project is part of **LLMs: A Hands-on Approach (IISc)** and aims to bridge the gap between:

* theoretical understanding of transformers
* real-world system-level LLM deployment

---

## 📜 License

For academic and research purposes.
