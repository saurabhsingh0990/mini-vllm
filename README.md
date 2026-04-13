# 🚀 Mini vLLM: High-Performance LLM Inference Engine

---

## 📌 Overview

Mini vLLM is a **production-inspired LLM inference engine** focused on understanding and implementing system-level optimizations for efficient text generation.

Unlike typical LLM projects that rely on high-level APIs, this project builds **core inference logic from scratch**, giving full control over how tokens are generated, optimized, and served.

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

### ✅ Custom Decoding Engine

* Greedy Decoding
* Top-K Sampling
* Top-P (Nucleus) Sampling

👉 Fully replaces HuggingFace `.generate()` with manual token-by-token generation.

---

### ⚡ KV Cache Optimization

* Uses `past_key_values` to avoid recomputation
* Processes only the latest token after first step
* Reduces inference latency significantly
* Works across all decoding strategies

---

### 🚀 Dynamic Batching

* Groups incoming requests within a configurable time window
* Improves throughput under concurrent load
* Built using:

  * request queue
  * background worker thread

👉 Demonstrates real-world trade-off:

* Higher latency
* Improved throughput

---

### ⚡ Quantized Inference

* Supports dynamic INT8 quantization using PyTorch
* Reduces memory footprint and improves CPU performance
* Includes **graceful fallback** for unsupported platforms

⚠️ Notes:

* Works on CPU backends like `fbgemm` (e.g., Colab CPU)
* Not supported on GPU or macOS ARM

---

### ⏱️ Latency Measurement

* End-to-end response time tracked at API level
* Enables comparison across:

  * decoding strategies
  * KV cache
  * batching
  * quantization

---

### 🎨 Interactive UI (Streamlit)

* Modern frontend to experiment with:

  * decoding strategies
  * KV cache
  * batching
  * quantization
* Displays:

  * generated output
  * latency metrics
  * batch size

👉 Makes the system **demo-ready and interactive**

---

## 🏗️ System Architecture

```
User Requests
     ↓
Streamlit UI (Optional)
     ↓
API Layer (FastAPI)
     ↓
Batching Layer (Queue + Worker)
     ↓
Inference Engine
     ├── KV Cache Layer
     ├── Decoding Strategies
     └── Quantization Layer
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
│   │   └── decoding.py
│   │   └── batcher.py
│
│   ├── api/
│   │   └── server.py
│
├── app.py                 # Streamlit UI
├── run.py
├── test_batch.py
├── requirements.txt
├── README.md
├── report.md
└── .gitignore
```

---

## ▶️ Usage

### 1. Start Backend

```bash
python run.py
```

---

### 2. Start UI

```bash
streamlit run app.py
```

---

### 3. Open

* API Docs → http://127.0.0.1:8000/docs
* UI → http://localhost:8501

---

## 🧪 Example Request

```bash
curl -X POST http://127.0.0.1:8000/generate \
-H "Content-Type: application/json" \
-d '{
  "prompt": "the elephant was sitting on the",
  "strategy": "top_p",
  "batching": true
}'
```

---

## 📊 Example Response

```json
{
  "response": "...",
  "strategy": "top_p",
  "kv_cache_enabled": true,
  "batching_enabled": true,
  "quantized": false,
  "batch_size": 4,
  "time_taken_seconds": 0.7421
}
```

---

## 🧠 Key Learnings

* Greedy decoding leads to **mode collapse**
* Sampling improves **diversity and realism**
* KV caching reduces **redundant computation**
* Dynamic batching improves **throughput with latency trade-offs**
* Quantization improves efficiency but depends on **hardware support**
* Production systems must handle **fallback scenarios**

---

## 📅 Roadmap

* [x] Decoding module
* [x] KV cache
* [x] Dynamic batching
* [x] Quantization
* [ ] Benchmarking & evaluation

---

## 💼 Project Motivation

This project is part of **LLMs: A Hands-on Approach (IISc)** and bridges:

* theoretical understanding
* real-world system-level implementation

---

## 📜 License

For academic and research purposes.
