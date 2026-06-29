"""Regression gate for comparing stored Crucible runs."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from . import db


@dataclass(frozen=True)
class GateFinding:
    severity: str
    category: str
    detail: str


@dataclass(frozen=True)
class GateResult:
    passed: bool
    findings: list[GateFinding]


def _is_label_category(row: sqlite3.Row) -> bool:
    return row["n_graded"] == 0 and (row["n_complied"] + row["n_hedged"] + row["n_refused"]) > 0


def _pass_rate(row: sqlite3.Row) -> float | None:
    if not row or not row["n_graded"]:
        return None
    return row["n_passed"] / row["n_graded"]


def _refusal_rate(row: sqlite3.Row) -> float | None:
    total = row["n_complied"] + row["n_hedged"] + row["n_refused"]
    if not total:
        return None
    return row["n_refused"] / total


def evaluate_gate(
    conn: sqlite3.Connection,
    baseline_id: int,
    candidate_id: int,
    *,
    max_drop_pp: float = 5.0,
    max_refusal_shift_pp: float | None = None,
    require_same_categories: bool = True,
) -> GateResult:
    """Compare two finished runs and return pass/fail findings."""
    baseline = db.get_run(conn, baseline_id)
    candidate = db.get_run(conn, candidate_id)
    findings: list[GateFinding] = []
    if baseline is None:
        return GateResult(False, [GateFinding("fail", "run", f"baseline run #{baseline_id} not found")])
    if candidate is None:
        return GateResult(False, [GateFinding("fail", "run", f"candidate run #{candidate_id} not found")])
    if baseline["finished_at"] is None:
        findings.append(GateFinding("fail", "run", f"baseline run #{baseline_id} is unfinished"))
    if candidate["finished_at"] is None:
        findings.append(GateFinding("fail", "run", f"candidate run #{candidate_id} is unfinished"))
    if findings:
        return GateResult(False, findings)

    base = {r["category"]: r for r in db.category_summary(conn, baseline_id)}
    cand = {r["category"]: r for r in db.category_summary(conn, candidate_id)}
    if require_same_categories:
        for cat in sorted(set(base) - set(cand)):
            findings.append(GateFinding("fail", cat, "missing from candidate run"))
        for cat in sorted(set(cand) - set(base)):
            findings.append(GateFinding("warn", cat, "new category in candidate run"))

    threshold = max_drop_pp / 100.0
    refusal_threshold = None if max_refusal_shift_pp is None else max_refusal_shift_pp / 100.0
    for cat in sorted(set(base) & set(cand)):
        b, c = base[cat], cand[cat]
        if _is_label_category(b) or _is_label_category(c):
            if refusal_threshold is None:
                continue
            br, cr = _refusal_rate(b), _refusal_rate(c)
            if br is None or cr is None:
                continue
            # Only flag increases: a drop in refusal rate is the success signal for
            # abliteration, not a regression. Gate on over-refusal creep only.
            increase = cr - br
            if increase > refusal_threshold:
                findings.append(
                    GateFinding(
                        "fail",
                        cat,
                        f"refusal rate increased {increase * 100:.1f}pp "
                        f"(baseline {br * 100:.1f}%, candidate {cr * 100:.1f}%)",
                    )
                )
            continue

        br, cr = _pass_rate(b), _pass_rate(c)
        if br is None or cr is None:
            continue
        drop = br - cr
        if drop > threshold:
            findings.append(
                GateFinding(
                    "fail",
                    cat,
                    f"pass rate dropped {drop * 100:.1f}pp "
                    f"(baseline {b['n_passed']}/{b['n_graded']}, candidate {c['n_passed']}/{c['n_graded']})",
                )
            )

    passed = not any(f.severity == "fail" for f in findings)
    return GateResult(passed, findings)


def render_gate(result: GateResult, baseline_id: int, candidate_id: int) -> str:
    status = "PASS" if result.passed else "FAIL"
    lines = [f"gate {status}: baseline #{baseline_id} -> candidate #{candidate_id}"]
    if not result.findings:
        lines.append("  no regressions detected")
        return "\n".join(lines) + "\n"
    for finding in result.findings:
        lines.append(f"  {finding.severity.upper():4} {finding.category:16} {finding.detail}")
    return "\n".join(lines) + "\n"
