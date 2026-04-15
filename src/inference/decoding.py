"""Token-by-token text generation engine with multiple decoding strategies.

Implements greedy, top-k, and top-p (nucleus) sampling strategies for text generation.
Supports KV cache optimization to reduce redundant computation during generation.
"""

import torch
import torch.nn.functional as F
from typing import Callable, Optional


# ----------------------------
# Decoding Strategies
# ----------------------------

def greedy_step(logits: torch.Tensor) -> torch.Tensor:
    """Select the token with highest probability (deterministic).
    
    Args:
        logits: Tensor of shape (batch_size, vocab_size) with model logits.
        
    Returns:
        Next token indices of shape (batch_size, 1).
    """
    return torch.argmax(logits, dim=-1, keepdim=True)


def top_k_step(logits: torch.Tensor, k: int = 50) -> torch.Tensor:
    """Select from top-k most likely tokens with sampling.
    
    Limits probability mass to top-k tokens, then samples from the distribution.
    Helps avoid unlikely low-probability tokens while maintaining diversity.
    
    Args:
        logits: Tensor of shape (batch_size, vocab_size) with model logits.
        k: Number of top tokens to consider (default: 50).
        
    Returns:
        Next token indices of shape (batch_size, 1).
    """
    top_k_logits, top_k_indices = torch.topk(logits, k)

    probs = F.softmax(top_k_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return top_k_indices.gather(-1, next_token)


def top_p_step(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """Select from nucleus (top-p) most likely tokens with sampling.
    
    Accumulates probability mass from the highest probability tokens until
    reaching threshold p. Often produces more natural text than top-k by
    adapting to the shape of the probability distribution.
    
    Args:
        logits: Tensor of shape (batch_size, vocab_size) with model logits.
        p: Cumulative probability threshold (default: 0.9).
        
    Returns:
        Next token indices of shape (batch_size, 1).
    """
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Keep tokens until we exceed probability threshold p
    sorted_indices_to_keep = cumulative_probs <= p
    # Always include at least the top token
    sorted_indices_to_keep[..., 0] = True

    # Zero out filtered probabilities and renormalize
    filtered_probs = sorted_probs * sorted_indices_to_keep
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(filtered_probs, num_samples=1)

    return sorted_indices.gather(-1, next_token)


# ----------------------------
# Unified Generation Engine
# ----------------------------

def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_length: int = 50,
    strategy: str = "top_p",
    use_cache: bool = True,
    k: int = 50,
    p: float = 0.9
) -> str:
    """Generate text token-by-token with specified strategy and optimizations.
    
    This is the core inference function. It:
    1. Tokenizes the input prompt
    2. Loops for max_length iterations
    3. Runs the model forward pass
    4. Selects next token using specified strategy
    5. Optionally uses KV cache to reduce recomputation
    
    Args:
        model: The language model to use for generation.
        tokenizer: Tokenizer for encoding/decoding text.
        prompt: Input text to complete.
        device: Device to run inference on ('cuda' or 'cpu').
        max_length: Maximum tokens to generate (default: 50).
        strategy: Decoding strategy - 'greedy', 'top_k', or 'top_p' (default: 'top_p').
        use_cache: Enable KV cache optimization (default: True).
        k: For top_k strategy, number of top tokens (default: 50).
        p: For top_p strategy, cumulative probability threshold (default: 0.9).
        
    Returns:
        Generated text string combining prompt and generated tokens.
        
    Raises:
        ValueError: If strategy is not recognized.
    """
    # Tokenize input prompt to token IDs and move to device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Initialize sequence with the prompt tokens
    generated = input_ids
    past_key_values = None

    # Generate tokens one at a time until reaching max_length
    for step in range(max_length):
        with torch.no_grad():
            # KV Cache optimization: only use new token after first step
            if use_cache and past_key_values is not None:
                # Pass only the latest token (much faster on subsequent steps)
                outputs = model(
                    input_ids=generated[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )
            else:
                # First step or cache disabled: process entire sequence
                outputs = model(
                    input_ids=generated,
                    use_cache=use_cache
                )

            logits = outputs.logits
            # Save cache for next iteration (if enabled)
            past_key_values = outputs.past_key_values if use_cache else None

            # Extract logits for the last token position
            next_token_logits = logits[:, -1, :]

            # Select next token using specified strategy
            if strategy == "greedy":
                next_token = greedy_step(next_token_logits)

            elif strategy == "top_k":
                next_token = top_k_step(next_token_logits, k)

            elif strategy == "top_p":
                next_token = top_p_step(next_token_logits, p)

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        # Append selected token to sequence
        generated = torch.cat((generated, next_token), dim=1)

    # Decode token IDs back to text
    return tokenizer.decode(generated[0], skip_special_tokens=True)