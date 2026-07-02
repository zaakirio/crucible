"""Pure base-vs-candidate comparison logic, shared by `crucible compare` and the TUI.

Kept separate from cli.py so the delta table is one implementation, not two: the CLI
renders these rows as text, the TUI renders the same rows into a DataTable.
"""

from __future__ import annotations

from dataclasses import dataclass


def is_label_category(c) -> bool:
    """Refusal-style categories report labels, not pass/fail - detect by data, not name."""
    return c["n_graded"] == 0 and (c["n_complied"] + c["n_hedged"] + c["n_refused"]) > 0


@dataclass
class ComparisonRow:
    category: str
    is_label: bool
    value_a: str
    value_b: str
    delta: str
    flagged: bool  # capability dropped >= 15pp - the one thing worth a visual flag either way


def build_comparison_rows(summary_a: dict, summary_b: dict) -> list[ComparisonRow]:
    """summary_a/summary_b: {category: row} from db.category_summary(), keyed by category.

    A missing category on either side renders as '-' rather than being dropped, so a
    partial `--only` run doesn't silently disappear from the comparison.
    """
    rows = []
    for cat in sorted(set(summary_a) | set(summary_b)):
        ca, cb = summary_a.get(cat), summary_b.get(cat)
        if (ca and is_label_category(ca)) or (cb and is_label_category(cb)):
            va = f"{ca['n_complied']}c/{ca['n_refused']}r" if ca else "-"
            vb = f"{cb['n_complied']}c/{cb['n_refused']}r" if cb else "-"
            delta = ""
            if ca and cb:
                delta = f"{cb['n_complied'] - ca['n_complied']:+d} complied"
            rows.append(ComparisonRow(cat, True, va, vb, delta, flagged=False))
        else:
            pa = ca["n_passed"] / ca["n_graded"] if ca and ca["n_graded"] else None
            pb = cb["n_passed"] / cb["n_graded"] if cb and cb["n_graded"] else None
            va = f"{ca['n_passed']}/{ca['n_graded']}" if ca else "-"
            vb = f"{cb['n_passed']}/{cb['n_graded']}" if cb else "-"
            delta = f"{(pb - pa) * 100:+.0f}%" if (pa is not None and pb is not None) else ""
            flagged = bool(delta) and (pb - pa) <= -0.15
            rows.append(ComparisonRow(cat, False, va, vb, delta, flagged=flagged))
    return rows
