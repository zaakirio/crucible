"""crucible eval — end-to-end pipeline: run → grade → report → model card.

Replaces the four-command manual workflow with a single entry point:

    crucible eval --server http://localhost:11434/v1 \\
                  --model-name ornith-9b-uncensored \\
                  [--base ornith-9b-base] \\
                  [--judge claude]

Judge is required. Auto-detected from environment in priority order:
  ANTHROPIC_API_KEY → OPENAI_API_KEY → DEEPSEEK_API_KEY
"""

from __future__ import annotations

import contextlib
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from . import db
from .config import apply_config_defaults, load_config
from .judge import REFUSAL_CATEGORIES, _resolve_preset, grade_run
from .model_card import render_delta_model_card, render_model_card, write_model_card
from .report import build_run_report, render_markdown, write_report
from .runner import load_tests, run_suite

console = Console()

# ── judge auto-detection ─────────────────────────────────────────────────────

_JUDGE_PRIORITY = [
    ("claude",    "ANTHROPIC_API_KEY"),
    ("openai",    "OPENAI_API_KEY"),
    ("deepseek",  "DEEPSEEK_API_KEY"),
]


def detect_judge(explicit_judge: str | None = None,
                 explicit_key: str | None = None) -> tuple[str, str]:
    """Return (judge_name, api_key). Raises if nothing can be resolved."""
    if explicit_judge and explicit_key:
        return explicit_judge, explicit_key

    # Explicit judge name without key → look up its env var
    if explicit_judge:
        for name, env_var in _JUDGE_PRIORITY:
            if name == explicit_judge:
                key = explicit_key or os.environ.get(env_var, "")
                if key:
                    return name, key
                raise ValueError(
                    f"Judge '{explicit_judge}' requires an API key. "
                    f"Set {env_var} or pass --api-key."
                )
        # Treat as URL judge
        if explicit_judge.startswith("http"):
            if not explicit_key:
                raise ValueError(
                    "URL judge requires --api-key (or set any supported env var)."
                )
            return explicit_judge, explicit_key

    # Auto-detect from environment
    for name, env_var in _JUDGE_PRIORITY:
        key = os.environ.get(env_var, "")
        if key:
            return name, key

    raise ValueError(
        "No judge API key found. Set one of:\n"
        "  ANTHROPIC_API_KEY  → uses Claude (recommended)\n"
        "  OPENAI_API_KEY     → uses gpt-4o-mini\n"
        "  DEEPSEEK_API_KEY   → uses deepseek-chat\n"
        "Or pass --judge <name> --api-key <key>."
    )


# ── model name / directory helpers ───────────────────────────────────────────

_SIZE_RE = re.compile(r"(\d+\.?\d*)\s*[bB](?=[^a-zA-Z]|$)")
_ABLIT_MARKERS = ("uncensored", "abliterat", "heretic", "decensored", "deccp")


def _parse_size(name: str) -> str:
    """Extract '9b', '1.2b' etc. from a model name string."""
    m = _SIZE_RE.search(name)
    if not m:
        return ""
    n = float(m.group(1))
    return f"{n:g}b"


def _clean_model_slug(name: str) -> str:
    """Convert a model name to a filesystem-safe slug without quant/size/markers."""
    slug = name.split("/")[-1].lower()
    # Strip abliteration markers
    for marker in _ABLIT_MARKERS:
        slug = re.sub(rf"[-_]?{marker}[-_]?", "-", slug, flags=re.IGNORECASE)
    # Strip quant suffixes (Q4_K_M, F16, etc.)
    slug = re.sub(r"[-_]?(q\d+[_k]*[sml]?|f16|bf16|f32|q\d+_\d+)[-_]?", "-",
                  slug, flags=re.IGNORECASE)
    # Strip size token
    slug = _SIZE_RE.sub("-", slug)
    # Strip version tokens (v1.0, 1.0, etc.)
    slug = re.sub(r"[-_]?v?\d+\.\d+[-_]?", "-", slug)
    slug = re.sub(r"[-_]{2,}", "-", slug).strip("-")
    return slug or "model"


def output_dir_name(model_name: str, out: str | Path | None = None) -> Path:
    """Resolve the output directory path."""
    if out:
        return Path(out).expanduser().resolve()
    size = _parse_size(model_name)
    slug = _clean_model_slug(model_name)
    parts = [slug, size, "eval"] if size else [slug, "eval"]
    return Path("-".join(parts))


# ── rich progress factory ─────────────────────────────────────────────────────

def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=36, style="steel_blue1", complete_style="bright_green"),
        MofNCompleteColumn(),
        TextColumn("[dim]·"),
        TimeElapsedColumn(),
        TextColumn("[dim]·"),
        TextColumn("[dim]{task.fields[detail]}"),
        console=console,
        transient=False,
    )


# ── single-model eval pipeline ────────────────────────────────────────────────

def _run_one(
    *,
    server_url: str,
    model_name: str,
    judge: str,
    api_key: str,
    db_path: Path,
    tests_dir: Path,
    docs_dir: Path | None,
    hardware: str,
    workers: int,
    suite_defaults: dict,
    progress: Progress,
    phase_prefix: str,
    total_phases: int,
) -> int:
    """Run eval + grade for one model. Returns run_id."""
    # Count total tests for the progress bar
    tests = load_tests(tests_dir, suite_defaults=suite_defaults)
    n_tests = len(tests)
    n_refusal = sum(
        1 for _, t in tests
        if t.get("grader") == "refusal"
        or _category_from_tests(tests_dir, t.get("id", "")) in REFUSAL_CATEGORIES
    )
    # Simpler: count after we know category labels
    refusal_cats = REFUSAL_CATEGORIES

    # ── Phase: run ──
    run_desc = f"[{phase_prefix}] running {n_tests} tests"
    run_task = progress.add_task(run_desc, total=n_tests, detail="starting…")

    def on_test(category, test, rep, g):
        label = g.label or ("pass" if g.passed else "fail")
        progress.update(run_task, advance=1,
                        detail=f"{category}  {test['id']}  {label}")

    conn = db.connect(str(db_path))
    conn.close()

    run_id = run_suite(
        server_url=server_url,
        model_name=model_name,
        tests_dir=tests_dir,
        db_path=db_path,
        hardware=hardware,
        workers=workers,
        docs_dir=docs_dir,
        suite_defaults=suite_defaults,
        on_progress=on_test,
    )
    progress.update(run_task, completed=n_tests, detail="done")

    # ── Phase: grade ──
    conn2 = db.connect(str(db_path))
    refusal_rows = db.refusal_results_for_run(conn2, run_id)
    n_grade = len(refusal_rows)
    conn2.close()

    grade_prefix = phase_prefix.replace("run", "grade").replace("1/", "").replace("3/", "")
    grade_desc = f"[{phase_prefix.split(']')[0].replace('[', '')} grade] grading {n_grade} refusal responses"
    grade_task = progress.add_task(grade_desc, total=n_grade, detail="starting…")

    def on_grade(i, total, category, test_id, label):
        progress.update(grade_task, advance=1,
                        detail=f"{category}  {test_id}  → {label}")

    conn3 = db.connect(str(db_path))
    grade_run(conn3, run_id, judge=judge, api_key=api_key, on_progress=on_grade)
    conn3.close()

    progress.update(grade_task, completed=n_grade, detail="done")
    return run_id


def _category_from_tests(tests_dir: Path, test_id: str) -> str:
    """Best-effort: extract category from test_id prefix (e.g. 'sorrybench-001' → 'sorrybench')."""
    parts = test_id.rsplit("-", 1)
    return parts[0] if len(parts) == 2 else ""


# ── main eval entry point ─────────────────────────────────────────────────────

def run_eval(
    *,
    server_url: str,
    model_name: str,
    base_model_name: str | None = None,
    judge: str | None = None,
    api_key: str | None = None,
    out: str | Path | None = None,
    tests_dir: str | Path = "tests",
    docs_dir: str | Path | None = None,
    hardware: str = "unknown",
    workers: int = 1,
    db_path: str | Path = "results.db",
    suite_defaults: dict | None = None,
    config_path: str = "crucible.yaml",
) -> Path:
    """Run the full eval pipeline and return the output directory path."""
    # Resolve judge
    judge_name, resolved_key = detect_judge(judge, api_key)

    # Resolve output dir
    out_dir = output_dir_name(model_name, out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use a run-local DB inside the output dir for clean isolation
    run_db = out_dir / "results.db"

    tests_path = Path(tests_dir)
    docs_path = Path(docs_dir) if docs_dir else None
    suite_def = suite_defaults or {}

    has_base = bool(base_model_name)
    total_phases = 4 if has_base else 3  # run[+base], grade[+base], report

    console.print()
    console.print(f"[bold]crucible eval[/bold]  →  [cyan]{out_dir}/[/cyan]")
    console.print(
        f"  model   [bold]{model_name}[/bold]"
        + (f"\n  base    [dim]{base_model_name}[/dim]" if has_base else "")
    )
    console.print(f"  judge   [dim]{judge_name}[/dim] (auto-detected)")
    console.print()

    ablit_run_id: int
    base_run_id: int | None = None

    with _make_progress() as progress:
        # ── Eval abliterated model ──
        phase = f"[1/{total_phases}]"
        ablit_run_id = _run_one(
            server_url=server_url,
            model_name=model_name,
            judge=judge_name,
            api_key=resolved_key,
            db_path=run_db,
            tests_dir=tests_path,
            docs_dir=docs_path,
            hardware=hardware,
            workers=workers,
            suite_defaults=suite_def,
            progress=progress,
            phase_prefix=phase,
            total_phases=total_phases,
        )

        # ── Eval base model (if --base) ──
        if has_base:
            phase = f"[2/{total_phases}]"
            base_run_id = _run_one(
                server_url=server_url,
                model_name=base_model_name,
                judge=judge_name,
                api_key=resolved_key,
                db_path=run_db,
                tests_dir=tests_path,
                docs_dir=docs_path,
                hardware=hardware,
                workers=workers,
                suite_defaults=suite_def,
                progress=progress,
                phase_prefix=phase,
                total_phases=total_phases,
            )

        # ── Generate outputs ──
        report_phase = f"[{total_phases}/{total_phases}]"
        output_task = progress.add_task(
            f"{report_phase} generating report", total=3, detail=""
        )

        conn = db.connect(str(run_db))
        try:
            ablit_report = build_run_report(conn, ablit_run_id)
            progress.update(output_task, advance=1, detail="report built")

            # Report
            report_path = out_dir / "report.md"
            write_report(render_markdown(ablit_report), report_path)
            progress.update(output_task, advance=1, detail="report.md")

            # Model card
            card_path = out_dir / "model-card.md"
            if base_run_id is not None:
                base_report = build_run_report(conn, base_run_id)
                ablit_judge = db.get_judge_results(conn, ablit_run_id)
                base_judge  = db.get_judge_results(conn, base_run_id)
                card_text = render_delta_model_card(
                    base_report=base_report,
                    ablit_report=ablit_report,
                    base_judge=base_judge,
                    ablit_judge=ablit_judge,
                    report_path=str(report_path),
                )
            else:
                card_text = render_model_card(
                    ablit_report,
                    report_path=str(report_path),
                    conn=conn,
                )
            write_model_card(card_text, card_path)
            progress.update(output_task, advance=1, detail="model-card.md")
        finally:
            conn.close()

    console.print()
    console.print(f"[bold green]✓[/bold green]  eval complete")
    console.print(f"   [cyan]{card_path}[/cyan]  ← paste this into your HuggingFace model card")
    console.print(f"   [dim]{report_path}[/dim]  ← full evidence report")
    console.print()

    return out_dir
