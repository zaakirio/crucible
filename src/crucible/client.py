"""Talk to a running llama-server over its OpenAI-compatible chat-completions API.

Determinism by default: temperature=0 + fixed seed. Everything downstream (graders,
regression gates) depends on stable output, so stability is the default, not an option.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx


@dataclass
class ToolCall:
    name: str
    arguments: dict | None  # parsed JSON args, or None if the model emitted invalid JSON
    raw_arguments: str      # exactly what the server returned, for grading detail


@dataclass
class ChatResult:
    text: str
    # llama.cpp returns a `timings` object with prefill/decode speeds. predicted_per_second
    # is the decode (generation) tok/s - we read it straight from the server, no client timing.
    tokens_per_second: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    raw: dict
    tool_calls: list[ToolCall] = field(default_factory=list)


def chat(
    base_url: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    temperature: float = 0.0,
    seed: int = 0,
    max_tokens: int = 512,
    timeout_s: float = 300.0,
) -> ChatResult:
    """Send a chat-completions request and return the assistant text + server-reported timings.

    When `tools` is given we rely on llama-server's own template-driven tool-call parsing
    (--jinja): the model is evaluated exactly as served, including how its tool-call syntax
    is parsed into the OpenAI tool_calls format.
    """
    payload = {
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        # llama.cpp surfaces its timings block when this is set.
        "timings_per_token": True,
    }
    if tools:
        payload["tools"] = tools
    r = httpx.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()

    message = data["choices"][0]["message"]
    text = message.get("content") or ""
    timings = data.get("timings") or {}
    usage = data.get("usage") or {}

    calls: list[ToolCall] = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments") or ""
        try:
            parsed = json.loads(raw_args) if raw_args else {}
            if not isinstance(parsed, dict):
                parsed = None
        except json.JSONDecodeError:
            parsed = None  # invalid-JSON arguments are a grading outcome, not a crash
        calls.append(ToolCall(name=fn.get("name") or "", arguments=parsed, raw_arguments=raw_args))

    return ChatResult(
        text=text,
        tokens_per_second=timings.get("predicted_per_second"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        raw=data,
        tool_calls=calls,
    )
