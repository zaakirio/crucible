# Design - Crucible Lab

Visual system for the Lab web UI (`lab-ui/`).
Register: product (dense forensic workbench); see PRODUCT.md for strategy.

## Theme

Dark only, by scene: the Lab runs next to a terminal with `llama-server` logs, usually at night.
Field is warm charcoal (OKLCH hue 55, chroma ~0.01), never pure black, never neon-terminal cosplay.

## Tokens (source of truth: `lab-ui/src/index.css`)

Surfaces: `--bg` 0.17L, `--surface` 0.205L, `--surface-2` 0.24L, `--surface-3` 0.28L, borders at 0.31/0.38L.
Ink: `--ink` 0.94L, `--ink-mid` 0.8L, `--ink-muted` 0.7L (all pass 4.5:1 on their surfaces).
Accent: ember orange OKLCH(0.68 0.17 45), reserved for primary actions, selection, and the abliterated lineage pill.

Data hues are semantic and reserved for data only:
complied = green (0.78 0.15 155), hedged = amber (0.82 0.13 90), refused = slate blue (0.72 0.09 265), fail = red (0.7 0.17 25).
Refused is deliberately not red: a refusal is a behavior, not an error.
Labels are never color-only; every hue is paired with text.

## Typography

System sans stack for everything; `ui-monospace` for test ids, hashes, categories, transcripts.
Base 13.5px, tight scale (h1 1.35rem, h2 1.05rem); `tabular-nums` via `.num` on all numeric cells.

## Components

Pills (`.pill.<label>`) for verdicts and lineage; chips (`.chip`) for provenance key-values.
`LabelBar` renders complied/hedged/refused as a stacked profile bar; `PassBar` renders pass rate with low/mid/high color steps.
Tables are the primary layout: `.table-wrap` owns horizontal scroll so the page never scrolls sideways.
Transcript inspector: list left, sticky detail right, arrow-key navigation, collapses to one column under 980px.

## Motion

150-250ms, `--ease` (ease-out-quint feel), state changes only: `.fade-in` on view/transcript swap, shimmer skeletons while loading, blinking caret while streaming.
All animation collapses under `prefers-reduced-motion`.
