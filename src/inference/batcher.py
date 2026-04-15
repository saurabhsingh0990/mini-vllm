"""Dynamic request batching module.

Provides a RequestBatcher class that accumulates inference requests and processes them
in batches to improve throughput. Implements a simple but effective batching strategy:
accumulate requests within a fixed time window, then process up to batch_size requests
together. This trades off latency for improved hardware utilization.
"""

import time
import threading
from typing import List, Tuple, Dict, Any


class RequestBatcher:
    """Groups inference requests and processes them in batches.
    
    Implements a worker thread pattern that continuously:
    1. Waits for wait_time seconds
    2. Collects queued requests
    3. Processes them in groups up to batch_size
    4. Sets result events for waiting clients
    """
    
    def __init__(self, model, batch_size=8, wait_time=0.2):
        """Initialize the batcher.
        
        Args:
            model: The inference model with a generate() method.
            batch_size: Maximum requests to process together (default: 8).
            wait_time: Time window in seconds to accumulate requests (default: 0.2s).
        """
        self.model = model
        self.batch_size = batch_size
        self.wait_time = wait_time

        self.queue: List[Tuple] = []
        self.lock = threading.Lock()

    def add_request(self, request_data: Dict[str, Any]) -> Tuple[threading.Event, Dict[str, Any]]:
        """Queue a request and return synchronization handles.
        
        This method is thread-safe and designed to be called from request handlers.
        The returned event signals when the result is ready, and result dict is populated
        with the inference output.
        
        Args:
            request_data: Dict with keys like 'prompt', 'max_length', 'strategy', etc.
            
        Returns:
            Tuple of (event, result_dict):
            - event: threading.Event that gets set when inference is complete
            - result_dict: Dict that will contain 'response' key with the output
        """
        event = threading.Event()
        result: Dict[str, Any] = {}

        with self.lock:
            self.queue.append((request_data, event, result))

        return event, result

    def process_batch(self) -> None:
        """Background worker that accumulates and processes request batches.
        
        This method is intended to run in a daemon thread. It continuously:
        1. Waits for requests to arrive
        2. Starts accumulation window (wait_time seconds)
        3. Atomically dequeues up to batch_size requests
        4. Processes each request via model.generate()
        5. Stores results and signals waiting clients via threading.Event
        
        The accumulation window creates a natural batching effect: requests arriving
        within wait_time will be batched together if there are remaining slots.
        """
        while True:
            # Wait until at least one request arrives (avoid spinning)
            if not self.queue:
                time.sleep(0.01)
                continue

            # Start accumulation window - give other requests time to arrive
            time.sleep(self.wait_time)

            with self.lock:
                # Atomically dequeue up to batch_size requests
                batch = self.queue[:self.batch_size]
                self.queue = self.queue[self.batch_size:]

            # Log batch processing (useful for debugging)
            print(f"[Batcher] Processing batch of size: {len(batch)}")

            # Process each request in the batch
            # Note: Sequential processing per-request, but batched at API layer
            for request_data, event, result in batch:
                try:
                    output = self.model.generate(**request_data)
                    result["response"] = output
                    result["batch_size"] = len(batch)
                except Exception as e:
                    # Graceful error handling: include error in result
                    result["response"] = f"Error: {str(e)}"
                    result["batch_size"] = len(batch)
                finally:
                    # Signal the waiting client that their result is ready
                    event.set()