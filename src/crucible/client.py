"""Talk to a running llama-server over its OpenAI-compatible chat-completions API.

Determinism by default: temperature=0 + fixed seed. Everything downstream (graders,
regression gates) depends on stable output, so stability is the default, not an option.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class ChatResult:
    text: str
    # llama.cpp returns a `timings` object with prefill/decode speeds. predicted_per_second
    # is the decode (generation) tok/s — we read it straight from the server, no client timing.
    tokens_per_second: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    raw: dict


def chat(
    base_url: str,
    messages: list[dict],
    *,
    temperature: float = 0.0,
    seed: int = 0,
    max_tokens: int = 512,
    timeout_s: float = 300.0,
) -> ChatResult:
    """Send a chat-completions request and return the assistant text + server-reported timings."""
    payload = {
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        # llama.cpp surfaces its timings block when this is set.
        "timings_per_token": True,
    }
    r = httpx.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()

    text = data["choices"][0]["message"]["content"] or ""
    timings = data.get("timings") or {}
    usage = data.get("usage") or {}

    return ChatResult(
        text=text,
        tokens_per_second=timings.get("predicted_per_second"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        raw=data,
    )
