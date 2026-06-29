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


def _judge_rows_to_dict(judge_rows: list) -> dict[str, dict]:
    """Convert judge_results DB rows (from db.get_judge_results) to {category: {c,h,r}} dict."""
    out: dict[str, dict] = {}
    judge_model = ""
    for row in judge_rows:
        cat = row["category"]
        out.setdefault(cat, {"complied": 0, "hedged": 0, "refused": 0})
        out[cat][row["judge_label"]] = out[cat].get(row["judge_label"], 0) + 1
        if not judge_model:
            judge_model = row["judge_model"]
    for cat in out:
        out[cat]["judge_model"] = judge_model
    return out


def render_delta_model_card(
    *,
    base_report: dict,
    ablit_report: dict,
    base_judge: list,
    ablit_judge: list,
    report_path: str | None = None,
) -> str:
    """Render a delta-focused model card comparing base vs abliterated.

    The delta is the centrepiece: refusal shift first, capability regression check second.
    Scores are not claims of intelligence — they are evidence that abliteration preserved capability.
    """
    base_run  = base_report["run"]
    ablit_run = ablit_report["run"]

    base_judge_dict  = _judge_rows_to_dict(base_judge)
    ablit_judge_dict = _judge_rows_to_dict(ablit_judge)
    judge_model = next(
        (v["judge_model"] for v in ablit_judge_dict.values() if v.get("judge_model")),
        "LLM judge",
    )

    # Build category maps
    base_cats  = {c["category"]: c for c in base_report["categories"]}
    ablit_cats = {c["category"]: c for c in ablit_report["categories"]}

    from . import db as _db  # local import to avoid circular
    from datetime import datetime, timezone as _tz
    date_str = datetime.now(_tz.utc).strftime("%Y-%m-%d")

    lines = [
        "## Crucible Abliteration Evidence",
        "",
        "> Scores below are **regression checks**, not capability claims.",
        "> The signal is the Δ column — did abliteration shift refusals without breaking anything?",
        "",
        f"- **abliterated**: `{Path(ablit_run['model_file']).name}`",
        f"- **base**: `{Path(base_run['model_file']).name}`",
        f"- judge: `{judge_model}`",
        f"- hardware: `{ablit_run['hardware'] or '-'}`",
        f"- date: `{date_str}`",
        f"- Crucible: `{ablit_run.get('crucible_version') or '-'}`",
        "",
    ]

    # ── Refusal profile delta ──
    refusal_order = ["sorrybench", "orbench", "xstest", "falsereject", "refusal"]
    refusal_cats = [c for c in refusal_order
                    if c in base_judge_dict or c in ablit_judge_dict]

    if refusal_cats:
        lines += [
            "### Refusal profile — did abliteration open the model?",
            "",
            "| benchmark | base | abliterated | Δ complied |",
            "|---|---|---|---|",
        ]
        for cat in refusal_cats:
            b = base_judge_dict.get(cat, {"complied": 0, "hedged": 0, "refused": 0})
            a = ablit_judge_dict.get(cat, {"complied": 0, "hedged": 0, "refused": 0})
            bn = b["complied"] + b["hedged"] + b["refused"]
            an = a["complied"] + a["hedged"] + a["refused"]
            b_str = f"{b['complied']}c / {b['hedged']}h / {b['refused']}r" if bn else "-"
            a_str = f"{a['complied']}c / {a['hedged']}h / {a['refused']}r" if an else "-"
            delta_c = a["complied"] - b["complied"]
            delta_str = (f"**+{delta_c}**" if delta_c > 0
                         else (f"-{abs(delta_c)}" if delta_c < 0 else "±0"))
            lines.append(f"| `{cat}` | {b_str} | {a_str} | {delta_str} |")
        lines.append("")

    # ── Capability delta ──
    cap_exclude = set(refusal_cats) | {"math", "agent_dialogue", "agent_tool",
                                        "rag_grounded", "rag_faithfulness"}
    cap_cats = [c for c in sorted(set(base_cats) | set(ablit_cats))
                if c not in cap_exclude
                and (base_cats.get(c, {}).get("n_graded", 0) or
                     ablit_cats.get(c, {}).get("n_graded", 0))]

    if cap_cats:
        lines += [
            "### Capability — did abliteration break anything?",
            "",
            "| category | n | base | abliterated | Δ |",
            "|---|---|---|---|---|",
        ]
        for cat in cap_cats:
            bc = base_cats.get(cat)
            ac = ablit_cats.get(cat)
            if bc and bc["n_graded"]:
                b_pct = 100 * bc["n_passed"] / bc["n_graded"]
                b_str = f"{b_pct:.0f}%"
                n = bc["n_graded"]
            else:
                b_str, n = "-", "-"
            if ac and ac["n_graded"]:
                a_pct = 100 * ac["n_passed"] / ac["n_graded"]
                a_str = f"{a_pct:.0f}%"
                if bc and bc["n_graded"]:
                    delta = a_pct - b_pct
                    if abs(delta) < 1:
                        d_str = "±0"
                    elif delta > 0:
                        d_str = f"**+{delta:.0f}pp**"
                    else:
                        d_str = f"{delta:.0f}pp"
                else:
                    d_str = "-"
            else:
                a_str, d_str = "-", "-"
            lines.append(f"| `{cat}` | {n} | {b_str} | {a_str} | {d_str} |")
        lines.append("")

    lines += [
        "---",
        f"*Generated by [crucible-eval](https://github.com/zaakirio/crucible).*",
    ]
    if report_path:
        lines.append(f"*Full evidence report: `{report_path}`*")
    lines.append("")
    return "\n".join(lines)


def write_model_card(text: str, path: str | Path | None) -> None:
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
