import torch
import torch.nn.functional as F


# ----------------------------
# Decoding Strategies
# ----------------------------

def greedy_step(logits):
    return torch.argmax(logits, dim=-1, keepdim=True)


def top_k_step(logits, k=50):
    top_k_logits, top_k_indices = torch.topk(logits, k)

    probs = F.softmax(top_k_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return top_k_indices.gather(-1, next_token)


def top_p_step(logits, p=0.9):
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_keep = cumulative_probs <= p
    sorted_indices_to_keep[..., 0] = True

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
    prompt,
    device,
    max_length=50,
    strategy="top_p",
    use_cache=True,
    k=50,
    p=0.9
):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated = input_ids
    past_key_values = None

    for step in range(max_length):
        with torch.no_grad():
            if use_cache and past_key_values is not None:
                outputs = model(
                    input_ids=generated[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )
            else:
                outputs = model(
                    input_ids=generated,
                    use_cache=use_cache
                )

            logits = outputs.logits
            past_key_values = outputs.past_key_values if use_cache else None

            next_token_logits = logits[:, -1, :]

            # 🔥 Select decoding strategy
            if strategy == "greedy":
                next_token = greedy_step(next_token_logits)

            elif strategy == "top_k":
                next_token = top_k_step(next_token_logits, k)

            elif strategy == "top_p":
                next_token = top_p_step(next_token_logits, p)

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        generated = torch.cat((generated, next_token), dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)