# 🚀 Mini vLLM: High-Performance LLM Inference Engine

## 📌 Overview

Mini vLLM is a **production-inspired LLM inference engine** designed to explore and implement system-level optimizations for serving Large Language Models efficiently.

This project focuses on **low-latency, high-throughput inference** by integrating techniques such as:

* ⚡ KV Cache Optimization
* 🔁 Dynamic Batching
* 📉 Quantized Inference
* 💾 Memory-efficient KV management
* 🚀 (Optional) SSD-based offloading

The goal is to **bridge the gap between theoretical LLM concepts and real-world deployment systems**.

---

## 🎯 Objectives

* Build a **mini version of modern inference engines (like vLLM)**
* Understand **transformer inference bottlenecks**
* Optimize:

  * Latency
  * Throughput
  * Memory usage
* Benchmark different optimization strategies

---

## 🧠 Key Features

### 1. KV Cache Optimization

Avoid recomputation of attention states to significantly reduce inference time.

### 2. Dynamic Batching

Efficiently batch multiple user requests to maximize GPU/CPU utilization.

### 3. Multiple Decoding Strategies

* Greedy
* Beam Search
* Top-K Sampling
* Top-P (Nucleus Sampling)

### 4. Quantization Support

* FP16 / INT8 / 4-bit inference
* Compare performance vs accuracy trade-offs

### 5. Streaming Inference API

* Token-by-token output (similar to ChatGPT)
* Built using FastAPI / WebSockets

### 6. (Optional) SSD Offloading

Simulate large-scale systems by offloading KV cache to disk.

---

## 🏗️ System Architecture

```
User Requests
     ↓
Request Queue
     ↓
Dynamic Batching Engine
     ↓
KV Cache Manager
     ↓
LLM Inference Engine
     ↓
Quantized Model
     ↓
Streaming API (FastAPI)
```

---

## 📂 Project Structure

```
mini-vllm/
│
├── src/
│   ├── model/              # Model loading & wrappers
│   ├── inference/          # Core inference logic
│   │   ├── kv_cache.py
│   │   ├── decoding.py
│   │   ├── batching.py
│   │
│   ├── scheduler/          # Request scheduling logic
│   ├── quantization/       # Quantization utilities
│   ├── api/                # FastAPI server
│   │   └── server.py
│   │
│   └── utils/              # Helper functions
│
├── experiments/            # Benchmark scripts
│   ├── latency_test.py
│   ├── throughput_test.py
│   └── memory_analysis.py
│
├── notebooks/              # Colab/Jupyter experiments
│
├── configs/                # Config files
│
├── tests/                  # Unit tests
│
├── requirements.txt
├── README.md
└── run.py                  # Entry point
```

---

## ⚙️ Prerequisites

### 🖥️ Hardware

* Minimum: CPU (development)
* Recommended: GPU (for benchmarking)

### 🧰 Software

* Python 3.9+
* PyTorch
* HuggingFace Transformers
* FastAPI
* Uvicorn

---

## 🔧 Installation

```bash
git clone https://github.com/your-username/mini-vllm.git
cd mini-vllm

pip install -r requirements.txt
```

---

## ▶️ Usage

### Run inference server:

```bash
python run.py
```

### Example request:

```bash
curl -X POST http://localhost:8000/generate \
-H "Content-Type: application/json" \
-d '{"prompt": "Explain transformers in simple terms"}'
```

---

## 📊 Evaluation Plan

We evaluate performance across:

* ⏱ Latency (per token & full response)
* 🚀 Throughput (requests/sec)
* 💾 Memory usage
* 📉 Impact of quantization
* 🔬 Ablation:

  * KV Cache ON vs OFF
  * Different batch sizes
  * Decoding strategies

---

## 📅 Roadmap

* [ ] Basic GPT-2 inference
* [ ] Implement decoding methods
* [ ] Add KV cache
* [ ] Build API server
* [ ] Dynamic batching
* [ ] Quantization
* [ ] Benchmarking & analysis
* [ ] SSD offloading (optional)

---

## 🧪 Experiments

* Latency vs Batch Size
* Memory vs Quantization
* Throughput vs Concurrency
* Decoding Quality Comparison

---

## 📚 References

* Attention is All You Need — Vaswani et al. (2017)
* vLLM: Efficient Memory Management for LLM Serving — Kwon et al. (2023)

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome!

---

## 📜 License

This project is for academic and research purposes.
