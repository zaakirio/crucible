"""Hugging Face model-card evidence snippets."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _rate(category: dict[str, Any]) -> str:
    if category["n_graded"]:
        return f"{category['n_passed']}/{category['n_graded']} ({100 * category['n_passed'] / category['n_graded']:.0f}%)"
    total = category["n_complied"] + category["n_hedged"] + category["n_refused"]
    if total:
        return f"{category['n_complied']} complied / {category['n_hedged']} hedged / {category['n_refused']} refused"
    return "-"


def render_model_card(report: dict[str, Any], *, report_path: str | None = None, export_path: str | None = None) -> str:
    """Render a concise markdown block suitable for a model card."""
    run = report["run"]
    summary = report["summary"]
    pass_rate = "-"
    if summary["pass_rate"] is not None:
        pass_rate = f"{summary['total_passed']}/{summary['total_graded']} ({100 * summary['pass_rate']:.0f}%)"

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
        f"- docs sha256: `{(run['docs_sha256'] or '-')[:12]}`",
        f"- graded pass rate: `{pass_rate}`",
        (
            "- refusal profile: "
            f"`{summary['labels']['complied']}` complied / "
            f"`{summary['labels']['hedged']}` hedged / "
            f"`{summary['labels']['refused']}` refused"
        ),
        "",
        "| category | result |",
        "|---|---:|",
    ]
    for category in report["categories"]:
        lines.append(f"| `{category['category']}` | {_rate(category)} |")
    lines.extend([
        "",
        "Caveat: these are local, served-runtime measurements for the recorded model file, hardware, "
        "llama.cpp commit, and test-suite hash. They are evidence for this setup, not universal claims.",
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
