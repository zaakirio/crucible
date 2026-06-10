# Crucible

**What survives quantization and abliteration.** A reproducible eval harness for self-hostable
models — capability, tool-calling, agents, and RAG — with first-class tracking of what quantization
and refusal-direction removal actually cost a model.

> Building in public. WIP. See [`../roadmap/`](../roadmap/) for the full plan.

## Why

Everyone benchmarks GPT-5. Crucible benchmarks what you can actually run on your own GPU — including
abliterated and quantized GGUFs — and reports the deltas nobody else publishes.

## Status — Module 00: drive a model over HTTP

Crucible drives `llama-server` over its OpenAI-compatible API (not `llama-cpp-python`), so it
evaluates a model exactly as it's served: same chat template (`--jinja`), same samplers, same
tool-call parsing your published GGUFs' users get.

```bash
cd crucible
uv sync

# list the GGUFs in ../models
uv run crucible models ../models

# spawn a server, run smoke prompts, print responses + tok/s
uv run crucible smoke ../models/LFM2.5-1.2B-Instruct-Uncensored-GGUF/LFM2.5-1.2B-Instruct-Uncensored-Q4_K_M.gguf
```

`crucible smoke` finds `llama-server` automatically (`../llama.cpp/build/bin/` or `$PATH`); override
with `$CRUCIBLE_LLAMA_SERVER`.

## Next

Module 01 adds tests-as-data (YAML), deterministic graders, SQLite result storage, the noise-floor
`--repeat` mode, and the quant-sweep `compare`. See the roadmap.
