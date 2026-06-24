# Crucible

**What survives quantization, abliteration, and serving.** A forensic eval workbench for
self-hostable models - capability, refusal behavior, tool-calling, RAG, and agent-style context -
with first-class tracking of what local deployment choices actually cost.

> Building in public. WIP. Crucible already covers capability, refusal, tool-calling, early
> grounded QA/RAG faithfulness, multi-turn dialogue fixtures, and starter tool-using agent workflows.

## Why

Most leaderboards benchmark remote frontier APIs or unserved model snapshots. Crucible measures
what you can actually run on your own GPU - including abliterated and quantized GGUFs - and reports
the deltas that matter when you choose a local model for real use.

Crucible drives `llama-server` over its OpenAI-compatible API (not `llama-cpp-python`), so it
evaluates a model exactly as it's served: same chat template (`--jinja`), same samplers, same
tool-call parsing your published GGUFs' users get. Every run records the llama.cpp commit, because
a score shift can be the engine, not your model.

## Status

Tests are YAML data (`tests/`), graded deterministically where possible (exact / numeric / regex /
code-exec / tool-call checks / refusal-profile), stored append-only in SQLite, compared across
runs, and charted.

```bash
cd crucible
uv sync

# optional project defaults for db/tests/docs/hardware/gate thresholds
uv run crucible --config crucible.yaml doctor

# seed the paper-comparable suites (GSM8K, XSTest, ...) - deterministic, fixed seed
uv run python scripts/seed_tests.py

# seed the tool-calling suites from BFCL v4 (Apache 2.0), then run just those
uv run python scripts/seed_tools.py
uv run crucible run models/<model>.gguf --only 'toolcall_*'

# grab a model straight from Hugging Face (any repo with GGUFs; $HF_TOKEN for gated ones)
uv run crucible pull LiquidAI/LFM2.5-1.2B-Instruct-GGUF Q4_K_M
uv run crucible pull bartowski/some-model-GGUF --list   # see what's in a repo first

# run the full suite against a GGUF; results land in results.db
uv run crucible run models/<model>.gguf -v
# add --resume to continue an unfinished run after interruption
# add --docs docs/rag to enable retrieval-backed grounded QA / RAG fixtures
uv run crucible run models/<model>.gguf --docs docs/rag --only 'rag_*'

# noise floor: same model 3x, reports which tests flap
uv run crucible run models/<model>.gguf --repeat 3

# the audit: diff two runs (base vs abliterated, Q4 vs Q8)
uv run crucible runs
uv run crucible compare 1 7

# local preflight / CI gate: nonzero exit if candidate regresses beyond thresholds
uv run crucible gate 1 7 --max-drop-pp 5 --max-refusal-shift-pp 20

# evidence pack for a run: provenance, category results, failures, and caveats
uv run crucible report 7 --out reports/run-7.md
uv run crucible report 7 --format json --out reports/run-7.json

# raw artifacts: one JSONL row per result, optionally reconstructing prompts/messages
uv run crucible export 7 --tests tests --docs docs/rag --out reports/run-7.jsonl

# Hugging Face-ready evidence block for model cards
uv run crucible model-card 7 --report-path reports/run-7.md --export-path reports/run-7.jsonl --out reports/model-card.md

# render findings as PNGs (quant curve, abliteration delta, refusal profile, pareto, ppl)
uv run crucible chart

# WikiText-2 perplexity (the literature's intrinsic metric), attached to the model's latest run
uv run crucible ppl models/<model>.gguf

# validate the refusal grader against your own judgment: hand-label a sample blind,
# then get a grader-vs-human agreement report. Measured here: 38/50 (76%) agreement
# over 50 blind labels; the disagreements were mostly complied-vs-hedged, with one
# hedged-vs-refused case.
uv run crucible label
uv run crucible label --report
```

Current coverage:
- local GGUF execution through `llama-server`
- deterministic grading and append-only SQLite storage
- provenance hashes for model files, tests, docs, and Crucible version
- refusal profiling, tool-calling, tool-using agent loops, PPL, and charts
- markdown/JSON evidence reports for stored runs
- raw JSONL artifact export for prompts, responses, tool calls, grader details, and reconstructed RAG context
- regression gates for local preflight or CI
- model-card evidence snippets, `crucible.yaml` defaults, and `doctor` environment checks
- resumable runs plus a mock-server integration test
- grounded QA / RAG faithfulness fixtures via local retrieval over `docs/rag`
- agent-style multi-turn conversation fixtures

`crucible smoke <model>` (quick 5-prompt sanity check) and `crucible models <dir>` (list GGUFs)
are still available.

**Requirements:** [uv](https://docs.astral.sh/uv/) and a built
[llama.cpp](https://github.com/ggml-org/llama.cpp) - `llama-server` is found via a sibling
`llama.cpp/build/bin/` checkout or `$PATH`; override with `$CRUCIBLE_LLAMA_SERVER`.

The unit suite needs neither: it mocks the server (including a real-subprocess mock over the
OpenAI-compatible API), so it runs offline with no model and no extra dependencies.

```bash
uv sync                                   # editable install + deps
uv run python -m unittest discover tests  # full unit suite (stdlib unittest, no model needed)
```

### Selected Findings

Selected results from finished runs only. These are the exact values stored in `results.db`
for one model family on one hardware setup and one llama.cpp commit; they are useful as
comparative evidence, not universal claims.

(LFM2.5-1.2B, base vs Heretic-abliterated, 2026-06-10)

| category | base [Q4_K_M] | abliterated [Q4_K_M] | Δ |
|---|---|---|---|
| gsm8k | 15/20 | 15/20 | +0pp |
| gsm_symbolic (n=100) | 54/100 | 49/100 | -5pp (within noise; gap shrank as n grew) |
| code | 5/6 | 5/6 | +0pp |
| instruction | 7/7 | 7/7 | +0pp |
| WikiText-2 PPL | 18.147 | 18.145 | ~0 |
| sorrybench (unsafe) | 19 complied / 11 hedged / **15 refused** | **44 complied / 1 / 0** | the point |
| orbench (over-refusal) | 42 complied / 6 hedged / 2 refused | 50 / 0 / 0 | false refusals gone |
| xstest | 32 complied / 3 hedged / 5 refused | 40 / 0 / 0 | - |

No capability cost that clears the noise bar, and the entire abliteration effect shows up where
it should: on SORRY-Bench's unsafe instructions the base model refused/hedged 26/45, the
abliterated model 1/45. Q3_K_M is the lowest-fidelity point in the sweep; above Q4 the differences
do not clear the n=20 noise bar. Noise floor: 0/89 unique tests flapped across 3 repetitions at
temperature 0.

![capability vs quantization](charts/quant_curve.png)
![refusal profile](charts/refusal_profile.png)

### Tool Calling

| category | Q3_K_M | Q4_K_M | Q5_K_M | Q6_K | Q8_0 | F16 |
|---|---|---|---|---|---|---|
| single call | 25/40 | 26/40 | 25/40 | 25/40 | 25/40 | 25/40 |
| choose right function | 13/20 | 12/20 | 13/20 | 12/20 | 13/20 | 13/20 |
| parallel calls | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| relevance (should call) | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| irrelevance (should NOT call) | 12/15 | 10/15 | 8/15 | 9/15 | 9/15 | 9/15 |

Three findings: (1) tool calling on this model is insensitive to quantization within the
measured sweep, with the same Q3_K_M performance in the same ballpark as F16; (2) what actually
gates tool use at 1.2B is **parallel calling (0% everywhere)** - the model emits exactly one
well-formed call no matter how many are required; (3) the serving stack is part of the result -
llama-server's tool-call parser returned a 500 on one Q5 output (recorded as a failure with
the error body, not a crash). Abliteration delta on tool calling at Q4_K_M is not observed in
these stored runs.

![tool calling vs quantization](charts/toolcall_curve.png)

### Test suites

| Category | Source | Grader |
|---|---|---|
| `gsm8k` | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) test split, seeded sample - kept for paper-comparable *deltas* | `numeric` |
| `gsm_symbolic` | [GSM-Symbolic](https://huggingface.co/datasets/apple/GSM-Symbolic) (ICLR 2025) - contamination-resistant regenerated math, for *absolute* claims | `numeric` |
| `xstest` | [XSTest](https://huggingface.co/datasets/Paul/XSTest) (Röttger et al.), stratified safe/unsafe | `refusal` profile |
| `orbench` | [OR-Bench-Hard](https://huggingface.co/datasets/bench-llm/or-bench) (ICML 2025) - over-refusal, harder than XSTest | `refusal` profile |
| `falsereject` | [FalseReject-Test](https://huggingface.co/datasets/AmazonScience/FalseReject) (2025) - over-refusal, human-annotated | `refusal` profile |
| `sorrybench` | [SORRY-Bench](https://huggingface.co/datasets/sorry-bench/sorry-bench-202503) (ICLR 2025) - refusal-of-unsafe, 1/category | `refusal` profile |
| `toolcall_single/multiple/parallel` | [BFCL v4](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) static categories (Apache 2.0) | `tool_call` (BFCL-AST style) |
| `toolcall_irrelevance/relevance` | BFCL v4 Live - knowing when *not* to call | `tool_call` |
| `agent_tool` | hand-authored tool-use loops with deterministic mocked tool results | final-answer graders |
| `rag_grounded` | local retrieval over `docs/rag/` | `exact` |
| `rag_faithfulness` | local retrieval with citations, abstention, distractors, and conflicting snippets | grounded graders |
| `agent_dialogue` | hand-authored multi-turn conversation fixtures | `exact` |
| `math`, `code`, `instruction`, `refusal` | hand-written starters | mixed |

Tool calls are evaluated as served: llama-server's own `--jinja` template parsing
produces the `tool_calls`, and grading checks function-name match, argument values against
BFCL's allowed lists, and no-call behavior on irrelevant prompts. Invalid-JSON arguments are
a recorded failure mode, not an error.

`agent_tool` fixtures go one step further: Crucible sends the model's parsed tool call back as an
assistant message, injects deterministic mocked tool results as `role=tool`, and grades the final
assistant answer. This tests whether a local model can complete the practical tool-use loop, not
just emit valid JSON.

RAG faithfulness fixtures test citation use, abstention when context lacks the answer, distractor
resistance, and conflict handling. The graders are deterministic: exact grounded answers, required
source markers, forbidden hallucinated answers, and required abstention markers.

Refusal categories report a **profile** (complied / hedged / refused), not pass/fail - moving
refusals to complies is the *point* of abliteration, so Crucible reports where each model lands.

Methodology follows the published work it extends, but the results here are local, model-specific,
and version-specific. They should be treated as reproducible measurements on the recorded setup,
not as generalized benchmark claims.

## Next

- expand stateful agent/tool workflows beyond the starter loops
- expand RAG corpora beyond the starter local docs
- expand model coverage and compare more quant / abliterated variants
