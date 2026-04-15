"""Streamlit UI for Mini vLLM inference engine.

Interactive web interface for experimenting with:
- Multiple decoding strategies (greedy, top-k, top-p)
- KV cache optimization
- Dynamic request batching
- Model quantization

Run with: streamlit run app.py
"""

import streamlit as st
import requests
import time

# API endpoint for inference
API_URL = "http://127.0.0.1:8000/generate"

st.set_page_config(
    page_title="Mini vLLM UI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Page Title and Description
# ----------------------------
st.title("🚀 Mini vLLM Inference Engine")
st.markdown("### Experiment with decoding strategies, KV cache, batching, and quantization")

# ----------------------------
# Sidebar Controls and Configuration
# ----------------------------
st.sidebar.header("⚙️ Configuration")

# Input prompt text area
prompt = st.sidebar.text_area(
    "📝 Enter Prompt",
    "The elephant was sitting on the",
    height=100,
    help="Text to complete or generate from"
)

# Token generation limit
max_length = st.sidebar.slider(
    "📏 Max Length", 
    min_value=10, 
    max_value=200, 
    value=50,
    help="Maximum number of tokens to generate"
)

# Decoding strategy selection
strategy = st.sidebar.selectbox(
    "🎲 Decoding Strategy",
    ["greedy", "top_k", "top_p"],
    help="greedy: deterministic | top_k: sample from top-k | top_p: nucleus sampling"
)

# Optimization flags
st.sidebar.markdown("### Optimizations")
use_cache = st.sidebar.toggle("⚡ KV Cache", value=True, help="Reduces redundant computation during generation")
batching = st.sidebar.toggle("📦 Dynamic Batching", value=True, help="Groups concurrent requests for improved throughput")
quantized = st.sidebar.toggle("🗜️ Quantization", value=False, help="Uses INT8 quantized model for reduced memory")

# ----------------------------
# Main Content
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Input Prompt")
    st.code(prompt, language="text")

with col2:
    st.subheader("⚙️ Active Configuration")
    st.json({
        "strategy": strategy,
        "kv_cache": use_cache,
        "batching": batching,
        "quantized": quantized,
        "max_length": max_length
    })

# ----------------------------
# Generate Button and Inference
# ----------------------------
generate_btn = st.sidebar.button("🚀 Generate", use_container_width=True)
if generate_btn:
    with st.spinner("🔄 Generating..."):
        start = time.time()

        # Prepare API request payload
        payload = {
            "prompt": prompt,
            "max_length": max_length,
            "strategy": strategy,
            "use_cache": use_cache,
            "batching": batching,
            "quantized": quantized
        }

        try:
            # Send request to backend API
            response = requests.post(API_URL, json=payload, timeout=60)
            result = response.json()
        except Exception as e:
            st.error(f"❌ API Error: {e}")
            st.stop()

        end = time.time()

    st.success("✅ Generation Complete")

    # ----------------------------
    # Generated Output
    # ----------------------------
    st.subheader("📤 Generated Output")
    st.write(result.get("response", "No response"))

    # ----------------------------
    # Performance Metrics
    # ----------------------------
    st.subheader("📊 Performance Metrics")

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("⏱️ Server Time", f"{result.get('time_taken_seconds', 0):.4f}s", delta="seconds")
    with col2:
        st.metric("⏱️ Client Time", f"{round(end - start, 4):.4f}s", delta="seconds")
    with col3:
        st.metric("📦 Batch Size", result.get("batch_size", 1))
    with col4:
        st.metric("🗜️ Quantized", "Yes" if result.get("quantized", False) else "No")

    # ----------------------------
    # Additional Metrics and Raw Response
    # ----------------------------
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Generation Config")
        st.write({
            "strategy": result.get("strategy"),
            "kv_cache_enabled": result.get("kv_cache_enabled"),
            "batching_enabled": result.get("batching_enabled")
        })
    
    with col2:
        st.subheader("🔍 Raw API Response")
        st.json(result)

