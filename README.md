# Crucible

**What survives quantization and abliteration.** A reproducible eval harness for self-hostable
models — capability, tool-calling, agents, and RAG — with first-class tracking of what quantization
and refusal-direction removal actually cost a model.

> Building in public. WIP — currently at Module 01 of a ten-module plan that grows this into
> tool-calling, agent, and RAG evals with a CI regression gate and a public leaderboard.

## Why

Everyone benchmarks GPT-5. Crucible benchmarks what you can actually run on your own GPU — including
abliterated and quantized GGUFs — and reports the deltas nobody else publishes.

Crucible drives `llama-server` over its OpenAI-compatible API (not `llama-cpp-python`), so it
evaluates a model exactly as it's served: same chat template (`--jinja`), same samplers, same
tool-call parsing your published GGUFs' users get. Every run records the llama.cpp commit, because
a score shift can be the engine, not your model.

## Status — Module 01: capability & refusal evals

Tests are YAML data (`tests/`), graded deterministically (exact / numeric / regex / code-exec /
refusal-profile), stored append-only in SQLite, compared across runs, and charted.

```bash
cd crucible
uv sync

# seed the paper-comparable suites (GSM8K, XSTest) — deterministic, fixed seed
uv run python scripts/seed_tests.py

# grab a model straight from Hugging Face (any repo with GGUFs; $HF_TOKEN for gated ones)
uv run crucible pull LiquidAI/LFM2.5-1.2B-Instruct-GGUF Q4_K_M
uv run crucible pull bartowski/some-model-GGUF --list   # see what's in a repo first

# run the full suite against a GGUF; results land in results.db
uv run crucible run models/<model>.gguf -v

# noise floor: same model 3x, reports which tests flap
uv run crucible run models/<model>.gguf --repeat 3

# the audit: diff two runs (base vs abliterated, Q4 vs Q8)
uv run crucible runs
uv run crucible compare 1 7

# render findings as PNGs (quant curve, abliteration delta, refusal profile, pareto, ppl)
uv run crucible chart

# WikiText-2 perplexity (the literature's intrinsic metric), attached to the model's latest run
uv run crucible ppl models/<model>.gguf

# validate the refusal grader against your own judgment: hand-label a sample blind,
# then get a grader-vs-human agreement report (marker-based graders are known to
# disagree with humans ~25% of the time — measure yours instead of trusting it)
uv run crucible label
uv run crucible label --report
```

`crucible smoke <model>` (quick 5-prompt sanity check) and `crucible models <dir>` (list GGUFs)
are still there from Module 00.

**Requirements:** [uv](https://docs.astral.sh/uv/) and a built
[llama.cpp](https://github.com/ggml-org/llama.cpp) — `llama-server` is found via a sibling
`llama.cpp/build/bin/` checkout or `$PATH`; override with `$CRUCIBLE_LLAMA_SERVER`.

### First findings (LFM2.5-1.2B, base vs Heretic-abliterated, Q3_K_M→F16, 2026-06-10)

| category | base [Q4_K_M] | abliterated [Q4_K_M] | Δ |
|---|---|---|---|
| gsm8k | 15/20 | 15/20 | +0pp |
| code | 5/6 | 5/6 | +0pp |
| instruction | 7/7 | 7/7 | +0pp |
| xstest | 32 complied / 3 hedged / 5 refused | **40 complied / 0 / 0** | the point |

Zero measured capability cost; the entire abliteration effect shows up in the refusal profile.
Q3_K_M is the quant cliff (GSM8K −30pp vs Q4_K_M); above Q4 the differences don't clear the n=20
noise bar. Noise floor: 0/94 tests flapped across 3 repetitions at temperature 0.

![capability vs quantization](charts/quant_curve.png)
![refusal profile](charts/refusal_profile.png)

### Test suites

| Category | Source | Grader |
|---|---|---|
| `gsm8k` | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) test split, seeded sample — kept for paper-comparable *deltas* | `numeric` |
| `gsm_symbolic` | [GSM-Symbolic](https://huggingface.co/datasets/apple/GSM-Symbolic) (ICLR 2025) — contamination-resistant regenerated math, for *absolute* claims | `numeric` |
| `xstest` | [XSTest](https://huggingface.co/datasets/Paul/XSTest) (Röttger et al.), stratified safe/unsafe | `refusal` profile |
| `orbench` | [OR-Bench-Hard](https://huggingface.co/datasets/bench-llm/or-bench) (ICML 2025) — over-refusal, harder than XSTest | `refusal` profile |
| `falsereject` | [FalseReject-Test](https://huggingface.co/datasets/AmazonScience/FalseReject) (2025) — over-refusal, human-annotated | `refusal` profile |
| `sorrybench` | [SORRY-Bench](https://huggingface.co/datasets/sorry-bench/sorry-bench-202503) (ICLR 2025) — refusal-of-unsafe, 1/category | `refusal` profile |
| `math`, `code`, `instruction`, `refusal` | hand-written starters (Module 00) | mixed |

Refusal categories report a **profile** (complied / hedged / refused), not pass/fail — moving
refusals to complies is the *point* of abliteration, so Crucible reports where each model lands.

Methodology follows the published work it extends: arXiv 2512.13655 (abliteration impact across
16 models) and arXiv 2601.14277 (llama.cpp quant impact, task-dependent).

## Next

Module 02 adds tool-calling evals: right function, valid arguments, knowing when *not* to call.
See the roadmap.
