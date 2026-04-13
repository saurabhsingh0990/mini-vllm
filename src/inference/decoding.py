import torch
import torch.nn.functional as F

def greedy_decode(model, tokenizer, prompt, device, max_length=50):
    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated = input_ids

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids=generated)

            logits = outputs.logits

            # Get last token logits
            next_token_logits = logits[:, -1, :]
            
            # Greedy: pick highest probability token
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        # Append token
        generated = torch.cat((generated, next_token), dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)



def top_k_decode(model, tokenizer, prompt, device, max_length=50, k=50):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated = input_ids

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids=generated)
            logits = outputs.logits

            next_token_logits = logits[:, -1, :]

            # Get top-k logits and indices
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k)

            # Convert logits to probabilities
            probs = F.softmax(top_k_logits, dim=-1)

            # Sample from top-k
            next_token = torch.multinomial(probs, num_samples=1)

            # Map back to original token indices
            next_token = top_k_indices.gather(-1, next_token)

        generated = torch.cat((generated, next_token), dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def top_p_decode(model, tokenizer, prompt, device, max_length=50, p=0.9):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated = input_ids

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids=generated)
            logits = outputs.logits

            next_token_logits = logits[:, -1, :]

            # Convert to probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # Sort probabilities
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)

            # Compute cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Mask tokens beyond top-p
            sorted_indices_to_keep = cumulative_probs <= p

            # Ensure at least one token is kept
            sorted_indices_to_keep[..., 0] = True

            # Filter probabilities
            filtered_probs = sorted_probs * sorted_indices_to_keep

            # Normalize again
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

            # Sample from filtered distribution
            next_token = torch.multinomial(filtered_probs, num_samples=1)

            # Map back to original token ids
            next_token = sorted_indices.gather(-1, next_token)

        generated = torch.cat((generated, next_token), dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)