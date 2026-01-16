from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import tiktoken


@dataclass(frozen=True)
class Pricing:
    embed_price_per_1k_usd: float  # embeddings input tokens
    chat_in_price_per_1k_usd: float  # prompt tokens
    chat_out_price_per_1k_usd: float  # completion tokens


def get_encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str) -> int:
    enc = get_encoding(model)
    return len(enc.encode(text or ""))


def estimate_embeddings_cost(tokens: int, pricing: Pricing) -> float:
    return (tokens / 1000.0) * pricing.embed_price_per_1k_usd


def estimate_chat_cost(prompt_tokens: int, completion_tokens: int, pricing: Pricing) -> float:
    return (prompt_tokens / 1000.0) * pricing.chat_in_price_per_1k_usd + (completion_tokens / 1000.0) * pricing.chat_out_price_per_1k_usd


def print_cost(
    *,
    embedding_tokens: Optional[int] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    pricing: Pricing,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    if embedding_tokens is not None:
        out["embedding_tokens"] = float(embedding_tokens)
        out["embedding_cost_usd"] = float(estimate_embeddings_cost(embedding_tokens, pricing))

    if prompt_tokens is not None and completion_tokens is not None:
        out["prompt_tokens"] = float(prompt_tokens)
        out["completion_tokens"] = float(completion_tokens)
        out["chat_cost_usd"] = float(estimate_chat_cost(prompt_tokens, completion_tokens, pricing))

    out["total_cost_usd"] = float(out.get("embedding_cost_usd", 0.0) + out.get("chat_cost_usd", 0.0))

    for k in sorted(out):
        print(f"{k}: {out[k]}")
    return out
