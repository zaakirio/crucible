"""Evidence reports for stored Crucible runs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from . import db


def _rowdict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def _short_hash(value: str | None) -> str:
    return value[:12] if value else "-"


def _rate(row: sqlite3.Row) -> str:
    if row["n_graded"]:
        return f"{row['n_passed']}/{row['n_graded']} ({100 * row['n_passed'] / row['n_graded']:.0f}%)"
    labels = row["n_complied"] + row["n_hedged"] + row["n_refused"]
    if labels:
        return f"{row['n_complied']} complied / {row['n_hedged']} hedged / {row['n_refused']} refused"
    return "-"


def build_run_report(conn: sqlite3.Connection, run_id: int, *, failure_limit: int = 20) -> dict[str, Any]:
    """Return a serializable evidence report for one stored run."""
    run = db.get_run(conn, run_id)
    if run is None:
        raise ValueError(f"run #{run_id} not found")
    categories = [_rowdict(r) for r in db.category_summary(conn, run_id)]
    failures = [_rowdict(r) for r in db.result_failures(conn, run_id, limit=failure_limit)]
    total_results = sum(c["n_results"] for c in categories)
    total_graded = sum(c["n_graded"] for c in categories)
    total_passed = sum(c["n_passed"] for c in categories)
    labels = {
        "complied": sum(c["n_complied"] for c in categories),
        "hedged": sum(c["n_hedged"] for c in categories),
        "refused": sum(c["n_refused"] for c in categories),
    }
    return {
        "run": _rowdict(run),
        "summary": {
            "finished": run["finished_at"] is not None,
            "total_results": total_results,
            "total_graded": total_graded,
            "total_passed": total_passed,
            "pass_rate": (total_passed / total_graded) if total_graded else None,
            "labels": labels,
        },
        "categories": categories,
        "failures": failures,
        "caveats": [
            "Results are local to the recorded model file, hardware, llama.cpp commit, and test-suite hash.",
            "Refusal categories are profiles, not pass/fail capability scores.",
            "Perplexity values are comparable only when measured with the same dataset and chunk count.",
            "Unfinished runs should not be used for published comparisons.",
        ],
    }


def render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def render_markdown(report: dict[str, Any]) -> str:
    run = report["run"]
    summary = report["summary"]
    status = "finished" if summary["finished"] else "unfinished"
    pass_rate = "-"
    if summary["pass_rate"] is not None:
        pass_rate = f"{summary['total_passed']}/{summary['total_graded']} ({100 * summary['pass_rate']:.0f}%)"

    lines = [
        f"# Crucible Run Report #{run['id']}",
        "",
        "## Run",
        "",
        f"- model: `{run['model_name']}`",
        f"- quant: `{run['quant'] or '-'}`",
        f"- lineage: `{run['lineage'] or '-'}`",
        f"- status: `{status}`",
        f"- started: `{run['started_at'] or '-'}`",
        f"- finished: `{run['finished_at'] or '-'}`",
        f"- hardware: `{run['hardware'] or '-'}`",
        f"- llama.cpp commit: `{run['llama_cpp_commit'] or '-'}`",
        f"- Crucible version: `{run['crucible_version'] or '-'}`",
        f"- context / GPU layers / repeat: `{run['ctx']}` / `{run['ngl']}` / `{run['repeat']}`",
        f"- model file: `{run['model_file']}`",
        f"- model size: `{run['model_size_bytes'] or '-'} bytes`",
        f"- model sha256: `{_short_hash(run['model_sha256'])}`",
        f"- tests sha256: `{_short_hash(run['tests_sha256'])}`",
        f"- docs sha256: `{_short_hash(run['docs_sha256'])}`",
        f"- category filter: `{run['only_filter'] or '-'}`",
    ]
    if run["ppl"] is not None:
        lines.append(f"- WikiText-2 PPL: `{run['ppl']:.4f}` over `{run['ppl_chunks']}` chunks")

    lines.extend([
        "",
        "## Summary",
        "",
        f"- total results: `{summary['total_results']}`",
        f"- graded pass rate: `{pass_rate}`",
        (
            "- refusal profile: "
            f"`{summary['labels']['complied']}` complied / "
            f"`{summary['labels']['hedged']}` hedged / "
            f"`{summary['labels']['refused']}` refused"
        ),
        "",
        "## Categories",
        "",
        "| category | result | avg tok/s |",
        "|---|---:|---:|",
    ])
    for c in report["categories"]:
        tps = f"{c['avg_tps']:.1f}" if c["avg_tps"] is not None else "-"
        lines.append(f"| `{c['category']}` | {_rate(c)} | {tps} |")

    lines.extend(["", "## Failures", ""])
    if report["failures"]:
        for f in report["failures"]:
            detail = (f["detail"] or "").replace("\n", " ")[:180]
            lines.append(f"- `{f['category']}/{f['test_id']}` rep `{f['rep']}`: {detail}")
    else:
        lines.append("- none recorded")

    lines.extend(["", "## Caveats", ""])
    for caveat in report["caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


def write_report(text: str, path: str | Path | None) -> None:
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
