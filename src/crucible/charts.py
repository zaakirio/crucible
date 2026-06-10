"""Charts — render the findings from results.db as PNGs.

The table is the product; these are the table's visual form. Four charts, each answering
one question someone running local models actually asks:

  quant_curve      — where does quality fall off as you quantize?  (pass-rate vs quant)
  ablit_delta      — what did abliteration cost?                   (base vs abliterated bars)
  refusal_profile  — did abliteration actually work?               (complied/hedged/refused)
  pareto           — which quant is the knee?                      (pass-rate vs tok/s vs size)

Every chart degrades gracefully: if the runs it needs aren't in the DB yet, it's skipped
with a reason instead of failing the command.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; never require a display
import matplotlib.pyplot as plt

# Ascending fidelity. Unknown quants sort to the end rather than erroring.
QUANT_ORDER = [
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "IQ4_XS", "Q4_K_S", "Q4_K_M",
    "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16", "F32",
]

# Categories graded pass/fail (refusal categories report labels instead).
_CAPABILITY_SQL = "passed IS NOT NULL"

_FIG_KW = {"dpi": 150, "bbox_inches": "tight"}


def quant_rank(quant: str | None) -> int:
    try:
        return QUANT_ORDER.index((quant or "").upper())
    except ValueError:
        return len(QUANT_ORDER)


def latest_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Newest run per (model_name, quant, lineage) — reruns supersede, never average."""
    return conn.execute(
        """
        SELECT * FROM runs WHERE id IN (
          SELECT MAX(id) FROM runs GROUP BY model_name, quant, lineage
        ) ORDER BY id
        """
    ).fetchall()


def _category_rates(conn: sqlite3.Connection, run_id: int) -> dict[str, float]:
    """category -> pass rate (capability categories only), averaged over reps."""
    rows = conn.execute(
        f"""
        SELECT category,
               AVG(CASE WHEN passed = 1 THEN 1.0 ELSE 0.0 END) AS rate
        FROM results WHERE run_id = ? AND {_CAPABILITY_SQL}
        GROUP BY category
        """,
        (run_id,),
    ).fetchall()
    return {r["category"]: r["rate"] for r in rows}


def _refusal_counts(conn: sqlite3.Connection, run_id: int) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT label, COUNT(*) AS n FROM results
        WHERE run_id = ? AND label IS NOT NULL GROUP BY label
        """,
        (run_id,),
    ).fetchall()
    return {r["label"]: r["n"] for r in rows}


def _overall(conn: sqlite3.Connection, run_id: int) -> tuple[float | None, float | None]:
    """(overall capability pass-rate, avg generation tok/s) for a run."""
    row = conn.execute(
        f"""
        SELECT AVG(CASE WHEN passed = 1 THEN 1.0 ELSE 0.0 END) AS rate,
               AVG(tok_per_sec) AS tps
        FROM results WHERE run_id = ? AND {_CAPABILITY_SQL}
        """,
        (run_id,),
    ).fetchone()
    return row["rate"], row["tps"]


def _sweep_runs(runs: list[sqlite3.Row]) -> list[sqlite3.Row]:
    """The (model_name, lineage) group with the most distinct quants = the sweep."""
    groups: dict[tuple, list] = {}
    for r in runs:
        if r["quant"]:
            groups.setdefault((r["model_name"], r["lineage"]), []).append(r)
    if not groups:
        return []
    best = max(groups.values(), key=lambda g: len({r["quant"] for r in g}))
    return sorted(best, key=lambda r: quant_rank(r["quant"]))


def chart_quant_curve(conn, out_dir: Path) -> Path | str:
    sweep = _sweep_runs(latest_runs(conn))
    if len(sweep) < 3:
        return "needs >=3 quants of one model (run the sweep)"
    quants = [r["quant"] for r in sweep]
    categories = sorted({c for r in sweep for c in _category_rates(conn, r["id"])})

    fig, ax = plt.subplots(figsize=(8, 5))
    for cat in categories:
        ys = [_category_rates(conn, r["id"]).get(cat) for r in sweep]
        ax.plot(quants, [y * 100 if y is not None else None for y in ys],
                marker="o", label=cat)
    ax.set_ylabel("pass rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"{sweep[0]['model_name']} ({sweep[0]['lineage']}) — capability vs quantization")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = out_dir / "quant_curve.png"
    fig.savefig(path, **_FIG_KW)
    plt.close(fig)
    return path


def _matched_pair(runs: list[sqlite3.Row]) -> tuple[sqlite3.Row, sqlite3.Row] | None:
    """A (base, abliterated) pair at the same quant, preferring higher-fidelity quants."""
    base = {r["quant"]: r for r in runs if r["lineage"] == "base"}
    ablit = {r["quant"]: r for r in runs if r["lineage"] == "abliterated"}
    shared = sorted(set(base) & set(ablit), key=quant_rank)
    if not shared:
        return None
    q = shared[-1]
    return base[q], ablit[q]


def chart_ablit_delta(conn, out_dir: Path) -> Path | str:
    pair = _matched_pair(latest_runs(conn))
    if not pair:
        return "needs a base + abliterated run at the same quant"
    b, a = pair
    rb, ra = _category_rates(conn, b["id"]), _category_rates(conn, a["id"])
    categories = sorted(set(rb) & set(ra))
    if not categories:
        return "matched runs share no graded categories"

    x = range(len(categories))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - w / 2 for i in x], [rb[c] * 100 for c in categories], w,
           label=f"base [{b['quant']}]", color="#4878a8")
    ax.bar([i + w / 2 for i in x], [ra[c] * 100 for c in categories], w,
           label=f"abliterated [{a['quant']}]", color="#c44e52")
    for i, c in enumerate(categories):
        d = (ra[c] - rb[c]) * 100
        ax.annotate(f"{d:+.0f}pp", (i, max(rb[c], ra[c]) * 100 + 2),
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(list(x), categories)
    ax.set_ylabel("pass rate (%)")
    ax.set_ylim(0, 112)
    ax.set_title(f"{b['model_name']} — what abliteration cost, by category")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    path = out_dir / "ablit_delta.png"
    fig.savefig(path, **_FIG_KW)
    plt.close(fig)
    return path


def chart_refusal_profile(conn, out_dir: Path) -> Path | str:
    runs = [r for r in latest_runs(conn) if _refusal_counts(conn, r["id"])]
    if not runs:
        return "no refusal-graded results yet"
    runs = sorted(runs, key=lambda r: (r["lineage"] != "base", quant_rank(r["quant"])))

    labels = [f"{r['lineage']}\n[{r['quant']}]" for r in runs]
    order = ["complied", "hedged", "refused"]
    colors = {"complied": "#55a868", "hedged": "#dd8452", "refused": "#c44e52"}
    counts = [_refusal_counts(conn, r["id"]) for r in runs]
    totals = [sum(c.values()) for c in counts]

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(runs)), 5))
    bottom = [0.0] * len(runs)
    for lab in order:
        vals = [100 * c.get(lab, 0) / t for c, t in zip(counts, totals)]
        ax.bar(labels, vals, bottom=bottom, label=lab, color=colors[lab])
        for i, v in enumerate(vals):
            if v >= 6:
                ax.annotate(f"{v:.0f}%", (i, bottom[i] + v / 2),
                            ha="center", va="center", fontsize=9, color="white")
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("share of refusal-eval prompts (%)")
    ax.set_title("Refusal profile — complied / hedged / refused")
    ax.legend(loc="lower right")
    path = out_dir / "refusal_profile.png"
    fig.savefig(path, **_FIG_KW)
    plt.close(fig)
    return path


def chart_pareto(conn, out_dir: Path) -> Path | str:
    runs = [r for r in latest_runs(conn) if r["quant"]]
    points = []
    for r in runs:
        rate, tps = _overall(conn, r["id"])
        if rate is not None and tps:
            points.append((r, rate, tps))
    if len(points) < 3:
        return "needs >=3 runs with capability results"

    fig, ax = plt.subplots(figsize=(8, 5))
    for r, rate, tps in points:
        gb = (r["model_size_bytes"] or 0) / 1e9
        color = "#c44e52" if r["lineage"] == "abliterated" else "#4878a8"
        ax.scatter(tps, rate * 100, s=120 * max(gb, 0.3), color=color, alpha=0.75, zorder=3)
        ax.annotate(f" {r['quant']} ({gb:.1f} GB)", (tps, rate * 100), fontsize=9)
    ax.set_xlabel("generation speed (tok/s, server-reported)")
    ax.set_ylabel("overall capability pass rate (%)")
    ax.set_title("Quality vs speed — where the knee is (marker size = file size)")
    ax.grid(True, alpha=0.3)
    path = out_dir / "pareto.png"
    fig.savefig(path, **_FIG_KW)
    plt.close(fig)
    return path


CHARTS = {
    "quant_curve": chart_quant_curve,
    "ablit_delta": chart_ablit_delta,
    "refusal_profile": chart_refusal_profile,
    "pareto": chart_pareto,
}


def render_all(conn, out_dir: str | Path) -> dict[str, Path | str]:
    """Render every chart that has data. Returns name -> Path (written) or str (skip reason)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {name: fn(conn, out) for name, fn in CHARTS.items()}
