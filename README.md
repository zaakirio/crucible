<p align="center"><img src="docs/banner.svg" alt="" width="100%"></p>

# Crucible

**What survives quantization, abliteration, and serving.**
A forensic eval workbench for self-hostable models: capability, refusal behavior,
tool-calling, RAG, and agent-style context, with first-class support for the
abliteration workflow - base vs uncensored delta measurement and model card generation.

## Why

Most leaderboards benchmark remote frontier APIs or unserved model snapshots.
Crucible measures what you can actually run on your own hardware, and reports the
deltas that matter when you abliterate a model: did refusals move to complies?
Did capability survive?

Crucible talks to any running OpenAI-compatible inference server.
It evaluates a model exactly as it's served - same chat template, same samplers,
same tool-call parsing your published GGUFs get.
Every run records provenance hashes (model file, test suite, llama.cpp commit)
so a score shift is attributable.

## Install

```bash
pip install crucible-eval
```

**Requirements:** any running OpenAI-compatible inference server (Ollama, LM Studio, vLLM, or llama-server).
No llama.cpp build required.

## Quickstart

```bash
# single model — generates ornith-9b-eval/model-card.md
crucible eval \
  --server http://localhost:11434/v1 \
  --model-name ornith-9b-uncensored \
  --judge claude \
  --workers 4

# base vs abliterated — delta-focused model card in one command
crucible eval \
  --server http://localhost:11434/v1 \
  --model-name ornith-9b-uncensored \
  --base ornith-9b-base \
  --judge claude \
  --workers 4
```

`--judge` is required and always explicit - Crucible never guesses a judge from whichever
env var happens to be set. Its API key comes from `--api-key`, or from the matching env var
for a named preset: `claude` → `ANTHROPIC_API_KEY` · `openai` → `OPENAI_API_KEY` · `deepseek` → `DEEPSEEK_API_KEY`.

Or from source:

```bash
git clone https://github.com/zaakirio/crucible
cd crucible
uv sync
```

## Running evals

Crucible works in two modes.

### External server (Ollama, LM Studio, vLLM, remote llama.cpp)

Start your server however you normally would, then:

```bash
# Ollama
uv run crucible run --server http://localhost:11434/v1 --model-name llama3 --workers 4

# Any OpenAI-compatible endpoint
uv run crucible run --server http://my-gpu-box:8080/v1 --model-name my-model --workers 4
```

`--workers 4` runs 4 requests concurrently.
On a single GPU, total token throughput stays the same but you get better utilisation
through prefill/decode overlap.

### Managed mode (local GGUF + llama.cpp)

If you have llama.cpp built, Crucible can spawn and manage `llama-server` for you:

```bash
# pull a GGUF from Hugging Face
uv run crucible pull zaakirio/LFM2.5-1.2B-Instruct-Uncensored-GGUF Q4_K_M

# run the full suite (llama-server found via $PATH or sibling llama.cpp/build/bin/)
uv run crucible run models/model.gguf --workers 4 -v
```

Override the binary with `$CRUCIBLE_LLAMA_SERVER` or `--ngl`/`--ctx` flags as needed.

### Preflight check

```bash
uv run crucible doctor
```

## The abliteration workflow

The core use case: prove your abliterated model is more open than the base without
being dumber. Three commands.

```bash
# 1. eval base model
uv run crucible run --server http://localhost:11434/v1 --model-name base-model --workers 4
# note the run id from `crucible runs`

# 2. eval abliterated model
uv run crucible run --server http://localhost:11434/v1 --model-name uncensored-model --workers 4

# 3. compare
uv run crucible compare <base-run-id> <abliterated-run-id>
```

## Crucible Lab (web workbench)

An interactive web UI over the results database: browse runs, drill into per-prompt
transcripts with keyword/judge/human verdicts side by side, diff two runs, and chat
against any OpenAI-compatible server in a playground.

```bash
pip install crucible-eval[lab]     # or: uv sync --extra lab
crucible lab                       # serves http://127.0.0.1:7860 over results.db
```

![Runs overview](docs/lab-runs.png)

![Base vs abliterated diff](docs/lab-compare.png)

![Playground streaming against a live model](docs/lab-playground.png)

Every aggregate is one click away from the raw transcripts that produced it - the
compare view flags any capability drop of 15pp or more, and refusal categories always
render as a complied/hedged/refused profile, never a single collapsed score.

Frontend development lives in `lab-ui/` (Vite + React); `npm run build` outputs to
`src/crucible/lab_static/`, which `crucible lab` serves. Build it once after cloning
from source (release wheels ship it prebuilt).

## LLM judge grading

The keyword refusal grader is fast and fully deterministic, but an LLM judge gives
more accurate complied/hedged/refused classifications - especially on short or ambiguous
responses.
Run it after any eval, bring your own API key:

```bash
# grade refusal categories with Claude, DeepSeek, OpenAI, or any OpenAI-compatible URL
uv run crucible grade <run-id> --judge claude --api-key $ANTHROPIC_API_KEY
uv run crucible grade <run-id> --judge deepseek --api-key $DEEPSEEK_API_KEY
uv run crucible grade <run-id> --judge openai --api-key $OPENAI_API_KEY
uv run crucible grade <run-id> --judge http://localhost:11434/v1 --model-name llama3

# model card shows both keyword and judge results side by side
uv run crucible model-card <run-id> --out model-card.md
```

Judge verdicts are stored in `judge_results` alongside the keyword labels - nothing
is overwritten.

## Configuration

`crucible.yaml` sets project defaults.
The most important section is `suite_defaults`, which controls generation budget
and thinking mode per category:

```yaml
gate:
  max_drop_pp: 5            # fail CI if capability drops more than this
  max_refusal_shift_pp: 5   # fail CI if refusal rate INCREASES (over-refusal creep)

suite_defaults:
  gsm8k:       {max_tokens: 512,  enable_thinking: false}
  sorrybench:  {max_tokens: 128,  enable_thinking: false}
  # ... see crucible.yaml for all categories
```

`enable_thinking` maps to `chat_template_kwargs` in llama.cpp's jinja pipeline.
Models that support a thinking toggle (e.g. Qwen3) respect it; others silently
ignore unknown template kwargs.
No per-model branching needed.

Thinking is off for every suite by default - Crucible grades the model's direct
answer, not its reasoning chain. For thinking models (Qwen3, DeepSeek-R1, etc.)
that support toggling it, `enable_thinking: true` is available as a per-suite
override in `crucible.yaml` if you need it; with thinking enabled, ensure `--ctx`
is large enough to give each parallel slot at least 2048 tokens: `--ctx 8192
--workers 4`.

## Other commands

```bash
# list available GGUFs
uv run crucible models

# quick 5-prompt sanity check (no grading)
uv run crucible smoke models/model.gguf

# run only specific categories
uv run crucible run models/model.gguf --only sorrybench,xstest --workers 4

# resume an interrupted run
uv run crucible run models/model.gguf --resume

# RAG-backed grounded QA (supply a docs directory)
uv run crucible run models/model.gguf --docs docs/rag --only rag_grounded,rag_faithfulness

# noise floor: same model 3x, see which tests flap
uv run crucible run models/model.gguf --repeat 3

# CI gate: nonzero exit if candidate regresses against baseline
uv run crucible gate <base-id> <candidate-id>

# validate keyword grader against your own judgment (blind labeling)
uv run crucible label --run <run-id>
uv run crucible label --report

# WikiText-2 perplexity, attached to the model's latest run
uv run crucible ppl models/model.gguf

# evidence pack and raw artifacts
uv run crucible report <run-id> --out reports/run.md
uv run crucible export <run-id> --out reports/run.jsonl

# charts (quant curve, refusal profile, abliteration delta, pareto, ppl)
uv run crucible chart
```

## Example findings

**[LFM2.5-1.2B — base vs Heretic-abliterated (2026-06-29)](charts/evidence.html)**

Full evidence report with refusal profile shift, capability preservation, and judge-validated grading.
Headline: sorrybench +26 complied, −21 refused, zero capability regression.

![Capability vs quantization](charts/quant_curve.png)

![Refusal profile](charts/refusal_profile.png)

## Test suites

| Category | Source | n | Grader |
|---|---|---|---|
| `gsm8k` | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) test split | 100 | `numeric` |
| `gsm_symbolic` | [GSM-Symbolic](https://huggingface.co/datasets/apple/GSM-Symbolic) (ICLR 2025) | 100 | `numeric` |
| `xstest` | [XSTest](https://huggingface.co/datasets/Paul/XSTest) stratified safe/unsafe | 40 | `refusal` profile |
| `orbench` | [OR-Bench-Hard](https://huggingface.co/datasets/bench-llm/or-bench) (ICML 2025) | 50 | `refusal` profile |
| `falsereject` | [FalseReject-Test](https://huggingface.co/datasets/AmazonScience/FalseReject) (2025) | 50 | `refusal` profile |
| `sorrybench` | [SORRY-Bench](https://huggingface.co/datasets/sorry-bench/sorry-bench-202503) (ICLR 2025) | 45 | `refusal` profile |
| `toolcall_single/multiple/parallel` | [BFCL v4](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) (Apache 2.0) | 40/20/20 | `tool_call` |
| `toolcall_irrelevance/relevance` | BFCL v4 Live | 15/5 | `tool_call` |
| `agent_tool` | hand-authored tool-use loops, deterministic mocked results | 3 | final-answer |
| `rag_grounded` | local retrieval over `docs/rag/` | 3 | `contains` |
| `rag_faithfulness` | local retrieval with citations, abstention, distractors | 4 | grounded |
| `agent_dialogue` | hand-authored multi-turn conversation fixtures | 3 | `exact` |
| `math`, `code`, `instruction`, `refusal` | hand-written starters | 8/6/7/8 | mixed |

All test YAML files are committed - no seed scripts needed.

Refusal categories report a **profile** (complied / hedged / refused), not pass/fail.
The keyword grader is deterministic and instant.
`crucible grade` adds an LLM judge layer for higher accuracy.

## Development

```bash
uv sync
uv run python -m unittest discover tests   # offline, no model needed
```

## Next

- `crucible compare` side-by-side in model card output
- thinking model auto-detection (no manual `enable_thinking` config needed)
- `crucible setup` for guided llama.cpp build
- expand RAG corpora and agent workflows
