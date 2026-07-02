"""crucible eval — end-to-end pipeline: run → grade → report → model card.

Replaces the four-command manual workflow with a single entry point:

    crucible eval --server http://localhost:11434/v1 \\
                  --model-name ornith-9b-uncensored \\
                  [--base ornith-9b-base] \\
                  --judge claude

Judge is required and always explicit - there is no default judge picked from
whichever env var happens to be set. Its API key comes from --api-key, or from
the matching env var for a named preset (ANTHROPIC_API_KEY for claude, etc.).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

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
from .judge import _PRESETS as _JUDGE_PRESETS
from .judge import grade_run
from .model_card import render_delta_model_card, render_model_card, write_model_card
from .report import build_run_report, render_markdown, write_report
from .runner import load_tests, run_suite

# ── judge resolution (no defaults) ──────────────────────────────────────────

def detect_judge(explicit_judge: str | None = None,
                 explicit_key: str | None = None) -> tuple[str, str]:
    """Return (judge_name, api_key) for an explicitly named judge or URL.

    Never guesses a judge from whichever env var happens to be set - --judge is
    always required. Only the *key* for a named judge may come from its env var.
    """
    if not explicit_judge:
        raise ValueError(
            f"No judge specified. Pass --judge <{'|'.join(_JUDGE_PRESETS)}|URL> --api-key <key>, "
            "or set the matching env var for a named preset "
            f"({', '.join(p['env_key'] for p in _JUDGE_PRESETS.values())})."
        )
    if explicit_key:
        return explicit_judge, explicit_key

    if explicit_judge in _JUDGE_PRESETS:
        env_var = _JUDGE_PRESETS[explicit_judge]["env_key"]
        key = os.environ.get(env_var, "")
        if key:
            return explicit_judge, key
        raise ValueError(
            f"Judge '{explicit_judge}' requires an API key. "
            f"Set {env_var} or pass --api-key."
        )

    if explicit_judge.startswith("http"):
        raise ValueError("URL judge requires --api-key (or set any supported env var).")

    raise ValueError(
        f"Unknown judge '{explicit_judge}'. Use a preset ({', '.join(_JUDGE_PRESETS)}) or a full URL."
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


# ── progress reporting ───────────────────────────────────────────────────────
#
# run_eval() drives its progress through an EvalReporter instead of talking to Rich
# directly, so a different front end (e.g. the Textual app in tui.py) can observe the
# exact same run→grade→report pipeline without duplicating its orchestration.

class EvalReporter:
    """No-op base: every hook is optional to override."""

    def __enter__(self) -> "EvalReporter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass

    def header(self, *, out_dir: Path, model_name: str,
               base_model_name: str | None, judge_name: str) -> None:
        pass

    def phase_start(self, label: str, total: int) -> None:
        pass

    def phase_tick(self, detail: str) -> None:
        pass

    def phase_done(self) -> None:
        pass

    def footer(self, *, card_path: Path, report_path: Path) -> None:
        pass


def _make_progress(console: Console) -> Progress:
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


class RichEvalReporter(EvalReporter):
    """Default reporter: renders exactly what `crucible eval` has always printed."""

    def __init__(self) -> None:
        self._console = Console()
        self._progress: Progress | None = None
        self._task: TaskID | None = None
        self._task_total = 0

    def __enter__(self) -> "RichEvalReporter":
        self._progress = _make_progress(self._console)
        self._progress.__enter__()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._progress is not None:
            self._progress.__exit__(*exc_info)

    def header(self, *, out_dir: Path, model_name: str,
               base_model_name: str | None, judge_name: str) -> None:
        self._console.print()
        self._console.print(f"[bold]crucible eval[/bold]  →  [cyan]{out_dir}/[/cyan]")
        self._console.print(
            f"  model   [bold]{model_name}[/bold]"
            + (f"\n  base    [dim]{base_model_name}[/dim]" if base_model_name else "")
        )
        self._console.print(f"  judge   [dim]{judge_name}[/dim]")
        self._console.print()

    def phase_start(self, label: str, total: int) -> None:
        assert self._progress is not None
        self._task_total = total
        self._task = self._progress.add_task(label, total=total, detail="starting…")

    def phase_tick(self, detail: str) -> None:
        assert self._progress is not None and self._task is not None
        self._progress.update(self._task, advance=1, detail=detail)

    def phase_done(self) -> None:
        assert self._progress is not None and self._task is not None
        self._progress.update(self._task, completed=self._task_total, detail="done")

    def footer(self, *, card_path: Path, report_path: Path) -> None:
        self._console.print()
        self._console.print("[bold green]✓[/bold green]  eval complete")
        self._console.print(f"   [cyan]{card_path}[/cyan]  ← paste this into your HuggingFace model card")
        self._console.print(f"   [dim]{report_path}[/dim]  ← full evidence report")
        self._console.print()


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
    reporter: EvalReporter,
    phase_label: str,
) -> int:
    """Run eval + grade for one model. Returns run_id."""
    tests = load_tests(tests_dir, suite_defaults=suite_defaults)
    n_tests = len(tests)

    conn = db.connect(str(db_path))
    conn.close()

    # ── Phase: run ──
    reporter.phase_start(f"{phase_label} running {n_tests} tests", n_tests)

    def on_test(category, test, rep, g):
        label = g.label or ("pass" if g.passed else "fail")
        reporter.phase_tick(f"{category}  {test['id']}  {label}")

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
    reporter.phase_done()

    # ── Phase: grade ──
    conn2 = db.connect(str(db_path))
    refusal_rows = db.refusal_results_for_run(conn2, run_id)
    n_grade = len(refusal_rows)
    conn2.close()

    reporter.phase_start(f"{phase_label} grading {n_grade} refusal responses", n_grade)

    def on_grade(i, total, category, test_id, label):
        reporter.phase_tick(f"{category}  {test_id}  → {label}")

    conn3 = db.connect(str(db_path))
    grade_run(conn3, run_id, judge=judge, api_key=api_key, on_progress=on_grade)
    conn3.close()

    reporter.phase_done()
    return run_id


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
    suite_defaults: dict | None = None,
    reporter: EvalReporter | None = None,
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

    reporter = reporter or RichEvalReporter()

    ablit_run_id: int
    base_run_id: int | None = None

    with reporter:
        reporter.header(
            out_dir=out_dir, model_name=model_name,
            base_model_name=base_model_name, judge_name=judge_name,
        )

        # ── Eval abliterated model ──
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
            reporter=reporter,
            phase_label=f"[1/{total_phases}]",
        )

        # ── Eval base model (if --base) ──
        if has_base:
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
                reporter=reporter,
                phase_label=f"[2/{total_phases}]",
            )

        # ── Generate outputs ──
        report_phase = f"[{total_phases}/{total_phases}]"
        reporter.phase_start(f"{report_phase} generating report", 3)

        conn = db.connect(str(run_db))
        try:
            ablit_report = build_run_report(conn, ablit_run_id)
            reporter.phase_tick("report built")

            # Report
            report_path = out_dir / "report.md"
            write_report(render_markdown(ablit_report), report_path)
            reporter.phase_tick("report.md")

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
            reporter.phase_tick("model-card.md")
        finally:
            conn.close()

        reporter.footer(card_path=card_path, report_path=report_path)

    return out_dir
