## Crucible Local Eval Evidence

- model file: `example-model-Q4_K_M.gguf`
- model sha256: `aaaaaaaaaaaa`
- quant / lineage: `Q4_K_M` / `base`
- hardware: `example-local-gpu`
- llama.cpp commit: `abc123`
- Crucible version: `0.0.1`
- context / GPU layers / repeat: `4096` / `99` / `1`
- tests sha256: `bbbbbbbbbbbb`
- docs sha256: `cccccccccccc`
- graded pass rate: `3/4 (75%)`
- refusal profile: `2` complied / `1` hedged / `1` refused

| category | result |
|---|---:|
| `math` | 1/1 (100%) |
| `rag_faithfulness` | 1/2 (50%) |
| `xstest` | 2 complied / 1 hedged / 1 refused |

Caveat: this file is a static example of Crucible's model-card evidence format, not a live model result.
