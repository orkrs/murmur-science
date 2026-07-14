"""Cache-backed token generation."""

import torch


@torch.no_grad()
def generate(model, input_ids: torch.Tensor, max_new_tokens: int, depth: int, eos_id: int | None = None) -> torch.Tensor:
    model.eval()
    result = input_ids
    cache = None
    for _ in range(max_new_tokens):
        current = result if cache is None else result[:, -1:]
        output = model(current, depths=torch.full((result.shape[0],), depth, device=result.device), cache=cache, use_cache=True)
        cache = output.diagnostics["cache"]
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        result = torch.cat([result, next_token], dim=1)
        if eos_id is not None and bool(torch.all(next_token == eos_id)):
            break
    return result
