# Product

## Register

product

## Users

Inference engineers and self-hosters who change how a model runs - quantize it, swap the serving stack, or modify the weights (including abliteration) - and need to prove what survived.
They live in terminals next to a running `llama-server`; the Lab is open in a browser on the same machine while evals run.
Secondary audience: readers who land on a shared run report and judge the rigor of the work by the evidence behind each number.

## Product Purpose

Crucible is a forensic eval workbench for self-hostable models: capability, refusal profiles, tool-calling, RAG, and agent behavior, with base-vs-modified delta measurement.
Crucible Lab is its interactive surface: browse runs, drill into per-prompt transcripts with keyword/judge/human labels side by side, diff two runs, and probe a live model in a playground.
Success: a user finds the one regressed category, reads the exact transcripts that caused it, and trusts the number because the evidence is one click deep.

## Brand Personality

Forensic, unflinching, precise.
The tone of a lab notebook, not a marketing dashboard: numbers carry the argument, labels are explicit (complied/hedged/refused), nothing is smoothed over.
Findings language, not vanity metrics.

## Anti-references

Generic SaaS analytics dashboards (hero KPI cards, gradient charts, celebratory tones).
Leaderboard sites that show a single score with no path to the underlying transcripts.
The default "AI dev tool" look: near-black + neon green terminal cosplay, or cream/paper minimalism.

## Design Principles

1. Evidence one click deep: every aggregate must open into the raw transcripts that produced it.
2. Labels over scores: refusal behavior is a profile (complied/hedged/refused), never a single pass rate; the UI must never collapse it.
3. Density is respect: this audience reads tables fluently; prefer one dense, scannable screen over three airy ones.
4. Provenance always visible: model hash, engine commit, context size travel with every number.
5. Deltas are the product: comparing two runs is the core act, so difference must be visually primary, not a bolted-on view.

## Accessibility & Inclusion

WCAG AA contrast throughout (4.5:1 body text); label categories never encoded by color alone (always paired with text).
Full keyboard navigation for tables and the transcript inspector; `prefers-reduced-motion` honored on all transitions.
