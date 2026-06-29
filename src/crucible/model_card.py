"""Hugging Face model-card evidence snippets."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def _rate(category: dict[str, Any]) -> str:
    if category["n_graded"]:
        return f"{category['n_passed']}/{category['n_graded']} ({100 * category['n_passed'] / category['n_graded']:.0f}%)"
    total = category["n_complied"] + category["n_hedged"] + category["n_refused"]
    if total:
        return f"{category['n_complied']} complied / {category['n_hedged']} hedged / {category['n_refused']} refused"
    return "-"


def _judge_summary(conn: sqlite3.Connection, run_id: int) -> dict[str, dict] | None:
    """Load per-category judge verdict counts, if any judge results exist for this run."""
    rows = conn.execute(
        """
        SELECT r.category, j.label, COUNT(*) as n
        FROM judge_results j
        JOIN results r ON r.id = j.result_id
        WHERE r.run_id = ?
        GROUP BY r.category, j.label
        ORDER BY r.category, j.label
        """,
        (run_id,),
    ).fetchall()
    if not rows:
        return None
    out: dict[str, dict] = {}
    for row in rows:
        cat = row["category"]
        out.setdefault(cat, {"complied": 0, "hedged": 0, "refused": 0, "judge_model": ""})
        out[cat][row["label"]] = row["n"]
    # Fetch judge model name used
    meta = conn.execute(
        "SELECT judge_model FROM judge_results j JOIN results r ON r.id=j.result_id WHERE r.run_id=? LIMIT 1",
        (run_id,),
    ).fetchone()
    judge_model = meta["judge_model"] if meta else "unknown"
    for cat in out:
        out[cat]["judge_model"] = judge_model
    return out


def render_model_card(
    report: dict[str, Any],
    *,
    report_path: str | None = None,
    export_path: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> str:
    """Render a concise markdown block suitable for a model card."""
    run = report["run"]
    summary = report["summary"]
    run_id = run.get("id")

    pass_rate = "-"
    if summary["pass_rate"] is not None:
        pass_rate = f"{summary['total_passed']}/{summary['total_graded']} ({100 * summary['pass_rate']:.0f}%)"

    judge = _judge_summary(conn, run_id) if (conn and run_id) else None

    lines = [
        "## Crucible Local Eval Evidence",
        "",
        f"- model file: `{Path(run['model_file']).name}`",
        f"- model sha256: `{(run['model_sha256'] or '-')[:12]}`",
        f"- quant / lineage: `{run['quant'] or '-'}` / `{run['lineage'] or '-'}`",
        f"- hardware: `{run['hardware'] or '-'}`",
        f"- llama.cpp commit: `{run['llama_cpp_commit'] or '-'}`",
        f"- Crucible version: `{run['crucible_version'] or '-'}`",
        f"- context / GPU layers / repeat: `{run['ctx']}` / `{run['ngl']}` / `{run['repeat']}`",
        f"- tests sha256: `{(run['tests_sha256'] or '-')[:12]}`",
        f"- graded pass rate: `{pass_rate}`",
        (
            "- refusal profile (keyword grader): "
            f"`{summary['labels']['complied']}` complied / "
            f"`{summary['labels']['hedged']}` hedged / "
            f"`{summary['labels']['refused']}` refused"
        ),
    ]

    if judge:
        judge_model = next(iter(judge.values()))["judge_model"]
        total_j = {k: sum(v[k] for v in judge.values()) for k in ("complied", "hedged", "refused")}
        lines.append(
            f"- refusal profile (LLM judge — {judge_model}): "
            f"`{total_j['complied']}` complied / "
            f"`{total_j['hedged']}` hedged / "
            f"`{total_j['refused']}` refused"
        )

    lines.extend([
        "",
        "| category | keyword grader | LLM judge |" if judge else "| category | result |",
        "|---|---:|---:|" if judge else "|---|---:|",
    ])

    for category in report["categories"]:
        cat_name = category["category"]
        kw = _rate(category)
        if judge and cat_name in judge:
            j = judge[cat_name]
            jstr = f"{j['complied']}c / {j['hedged']}h / {j['refused']}r"
            lines.append(f"| `{cat_name}` | {kw} | {jstr} |")
        elif judge:
            lines.append(f"| `{cat_name}` | {kw} | - |")
        else:
            lines.append(f"| `{cat_name}` | {kw} |")

    lines.extend([
        "",
        "Caveat: these are local, served-runtime measurements for the recorded model file, hardware, "
        "llama.cpp commit, and test-suite hash. They are evidence for this setup, not universal claims.",
        "Refusal profile is a distribution, not a score - the delta between base and abliterated is the signal.",
    ])
    if report_path or export_path:
        lines.append("")
        if report_path:
            lines.append(f"- full report: `{report_path}`")
        if export_path:
            lines.append(f"- raw JSONL artifacts: `{export_path}`")
    lines.append("")
    return "\n".join(lines)


def write_model_card(text: str, path: str | Path | None) -> None:
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
