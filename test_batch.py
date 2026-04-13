# 📄 test_batch.py (Updated & Improved)


import requests
import time
import concurrent.futures

URL = "http://localhost:8000/generate"


def send_request(prompt, idx):
    payload = {
        "prompt": prompt,
        "max_length": 20,
        "strategy": "top_p",
        "use_cache": True,
        "batching": True
    }

    start = time.time()
    response = requests.post(URL, json=payload)
    end = time.time()

    try:
        result = response.json()
    except Exception:
        return {
            "request_id": idx,
            "error": "Invalid JSON response",
            "client_time": round(end - start, 4)
        }

    return {
        "request_id": idx,
        "prompt": prompt,
        "response": result.get("response", "")[:60],
        "batching_enabled": result.get("batching_enabled"),
        "batch_size": result.get("batch_size"),
        "server_time": result.get("time_taken_seconds"),
        "client_time": round(end - start, 4)
    }


def main():
    prompts = [
        "Hello, how are you?",
        "What is the weather like?",
        "Tell me a joke.",
        "Explain quantum physics.",
        "What is AI?",
        "Describe a cat."
    ]

    print("🚀 Sending concurrent requests to test batching...\n")

    NUM_REQUESTS = len(prompts)

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_REQUESTS) as executor:
        futures = []

        # Slight stagger to help batching window
        for i, prompt in enumerate(prompts):
            futures.append(executor.submit(send_request, prompt, i))
            time.sleep(0.02)  # small delay (important)

        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    print("\n📊 Results:\n")

    for res in sorted(results, key=lambda x: x["request_id"]):
        print(f"Request {res['request_id'] + 1}:")
        print(f"  Prompt           : {res['prompt']}")
        print(f"  Response         : {res['response']}...")
        print(f"  Batching Enabled : {res['batching_enabled']}")
        print(f"  Batch Size       : {res['batch_size']}")
        print(f"  Server Time      : {res['server_time']}s")
        print(f"  Client Time      : {res['client_time']}s")
        print()


if __name__ == "__main__":
    main()

