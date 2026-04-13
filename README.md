# 🚀 Mini vLLM: High-Performance LLM Inference Engine

## 📌 Overview

Mini vLLM is a **production-inspired LLM inference engine** focused on understanding and implementing system-level optimizations for efficient text generation.

Unlike typical LLM projects that rely on high-level APIs, this project builds **core inference logic from scratch**, giving full control over how tokens are generated.

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

### ⚡ Upcoming Features

* KV Cache Optimization
* Dynamic Batching
* Quantized Inference (INT8 / 4-bit)
* Streaming API (token-by-token response)
* SSD-based offloading (optional)

---

## 🏗️ System Architecture

```
User Requests
     ↓
Request Queue
     ↓
Inference Engine
     ↓
Custom Decoding Module
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
│   │   └── decoding.py   # Custom decoding logic
│
│   ├── api/
│   │   └── server.py
│
├── run.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Prerequisites

### 🖥️ Hardware

* Minimum: CPU
* Recommended: GPU (for faster inference)

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

### Open API docs:

```
http://127.0.0.1:8000/docs
```

---

## 🧪 Example Request

```bash
curl -X POST http://127.0.0.1:8000/generate \
-H "Content-Type: application/json" \
-d '{"prompt": "the elephant was sitting on the", "max_length": 50}'
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
* Sampling-based methods improve **diversity**
* Token-by-token control is essential for:

  * KV cache
  * batching
  * real-world inference systems

---

## 📅 Roadmap

* [x] Basic inference engine
* [x] Custom decoding module
* [ ] KV cache optimization
* [ ] Dynamic batching
* [ ] Quantization
* [ ] Benchmarking & evaluation

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
