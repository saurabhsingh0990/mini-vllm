# src/inference/batcher.py

import time
import threading


class RequestBatcher:
    def __init__(self, model, batch_size=4, wait_time=0.05):
        self.model = model
        self.batch_size = batch_size
        self.wait_time = wait_time

        self.queue = []
        self.lock = threading.Lock()

    def add_request(self, request_data):
        event = threading.Event()
        result = {}

        with self.lock:
            self.queue.append((request_data, event, result))

        return event, result

    def process_batch(self):
        while True:
            # Wait until at least one request is present
            if not self.queue:
                time.sleep(0.01)
                continue

            # First request arrived → start batching window
            time.sleep(self.wait_time)

            with self.lock:
                batch = self.queue[:self.batch_size]
                self.queue = self.queue[self.batch_size:]

            print(f"[Batcher] Processing batch of size: {len(batch)}")

            for request_data, event, result in batch:
                try:
                    output = self.model.generate(**request_data)
                    result["response"] = output
                    result["batch_size"] = len(batch)
                except Exception as e:
                    result["response"] = f"Error: {str(e)}"
                    result["batch_size"] = len(batch)

                event.set()