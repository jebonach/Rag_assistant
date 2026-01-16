from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: Optional[str]
    model: str
    temperature: float = 0.0
    max_tokens: int = 600


def make_client(api_key: str, base_url: Optional[str]) -> OpenAI:
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def generate_answer(
    question: str,
    context: str,
    system_prompt: str,
    cfg: LLMConfig,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (answer_text, usage_dict).
    usage_dict is provider-dependent; OpenAI typically returns prompt_tokens/completion_tokens/total_tokens.
    """
    client = make_client(cfg.api_key, cfg.base_url)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"КОНТЕКСТ:\n{context}\n\nВОПРОС:\n{question}"},
    ]

    resp = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )

    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    usage_dict = usage.model_dump() if usage is not None else {}
    return text, usage_dict
