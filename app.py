import streamlit as st
import requests
import time

# ----------------------------
# Config
# ----------------------------
API_URL = "http://127.0.0.1:8000/generate"

st.set_page_config(
    page_title="Mini vLLM UI",
    page_icon="🚀",
    layout="wide"
)

# ----------------------------
# Title
# ----------------------------
st.title("🚀 Mini vLLM Inference Engine")
st.markdown("### Experiment with decoding, KV cache, batching & quantization")

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("⚙️ Controls")

prompt = st.sidebar.text_area(
    "Enter Prompt",
    "The elephant was sitting on the",
    height=120
)

max_length = st.sidebar.slider("Max Length", 10, 200, 50)

strategy = st.sidebar.selectbox(
    "Decoding Strategy",
    ["greedy", "top_k", "top_p"]
)

use_cache = st.sidebar.toggle("KV Cache", value=True)
batching = st.sidebar.toggle("Dynamic Batching", value=True)
quantized = st.sidebar.toggle("Quantization", value=False)

# ----------------------------
# Generate Button
# ----------------------------
generate_btn = st.sidebar.button("🚀 Generate")

# ----------------------------
# Main Layout
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Prompt")
    st.code(prompt, language="text")

with col2:
    st.subheader("⚙️ Selected Configuration")
    st.write({
        "strategy": strategy,
        "kv_cache": use_cache,
        "batching": batching,
        "quantized": quantized
    })

# ----------------------------
# Generate Output
# ----------------------------
if generate_btn:
    with st.spinner("Generating... 🚀"):
        start = time.time()

        payload = {
            "prompt": prompt,
            "max_length": max_length,
            "strategy": strategy,
            "use_cache": use_cache,
            "batching": batching,
            "quantized": quantized
        }

        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

        end = time.time()

    st.success("✅ Generation Complete")

    # ----------------------------
    # Output
    # ----------------------------
    st.subheader("📤 Output")
    st.write(result.get("response", ""))

    # ----------------------------
    # Metrics
    # ----------------------------
    st.subheader("📊 Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Server Time (s)", result.get("time_taken_seconds", 0))
    col2.metric("Client Time (s)", round(end - start, 4))
    col3.metric("Batch Size", result.get("batch_size", 1))
    col4.metric("Quantized", result.get("quantized", False))

    # ----------------------------
    # Raw JSON (for debugging)
    # ----------------------------
    with st.expander("🔍 Raw Response"):
        st.json(result)

