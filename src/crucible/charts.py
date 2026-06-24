"""Charts - render the findings from results.db as PNGs.

The table is the product; these are the table's visual form. Six charts, each answering
one question someone running local models actually asks:

  quant_curve      - where does quality fall off as you quantize?  (pass-rate vs quant)
  toolcall_curve   - does tool calling survive quantization?       (per toolcall category)
  ablit_delta      - what did abliteration cost?                   (base vs abliterated bars)
  refusal_profile  - did abliteration actually work?               (complied/hedged/refused)
  pareto           - which quant is the knee?                      (pass-rate vs tok/s vs size)
  ppl_curve        - the intrinsic metric                          (perplexity vs quant)

Stats are merged at the CATEGORY level: for each (model, quant, lineage, category) the
newest run containing that category wins. Partial runs (e.g. `run --only toolcall_*`)
update their categories without shadowing a full run's other categories.

Every chart degrades gracefully: if the runs it needs aren't in the DB yet, it's skipped
with a reason instead of failing the command.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; never require a display
import matplotlib.pyplot as plt

# Ascending fidelity. Unknown quants sort to the end rather than erroring.
QUANT_ORDER = [
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "IQ4_XS", "Q4_K_S", "Q4_K_M",
    "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16", "F32",
]

_FIG_KW = {"dpi": 150, "bbox_inches": "tight"}


def quant_rank(quant: str | None) -> int:
    try:
        return QUANT_ORDER.index((quant or "").upper())
    except ValueError:
        return len(QUANT_ORDER)


@dataclass
class CatStats:
    rate: float | None = None      # pass rate, None for label-graded categories
    tps: float | None = None
    labels: dict = field(default_factory=dict)  # complied/hedged/refused counts


@dataclass
class GroupStats:
    """All known results for one (model_name, quant, lineage), newest-run-per-category."""
    model_name: str
    quant: str | None
    lineage: str
    categories: dict[str, CatStats] = field(default_factory=dict)
    ppl: float | None = None
    ppl_chunks: int | None = None
    model_size_bytes: int | None = None

    def capability_rate(self) -> float | None:
        rates = [c.rate for c in self.categories.values() if c.rate is not None]
        return sum(rates) / len(rates) if rates else None

    def avg_tps(self) -> float | None:
        tps = [c.tps for c in self.categories.values() if c.tps]
        return sum(tps) / len(tps) if tps else None

    def label_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.categories.values():
            for k, v in c.labels.items():
                out[k] = out.get(k, 0) + v
        return out


def merged_stats(conn: sqlite3.Connection) -> dict[tuple, GroupStats]:
    """(model_name, quant, lineage) -> GroupStats, newest run winning per category."""
    groups: dict[tuple, GroupStats] = {}
    # Only finished runs belong in aggregate reporting. Unfinished runs may be useful for
    # resuming a local session, but they are not scientifically valid inputs to charts.
    for run in conn.execute("SELECT * FROM runs WHERE finished_at IS NOT NULL ORDER BY id").fetchall():
        key = (run["model_name"], run["quant"], run["lineage"])
        g = groups.setdefault(key, GroupStats(run["model_name"], run["quant"], run["lineage"]))
        if run["model_size_bytes"]:
            g.model_size_bytes = run["model_size_bytes"]
        if run["ppl"] is not None:
            g.ppl, g.ppl_chunks = run["ppl"], run["ppl_chunks"]
        rows = conn.execute(
            """
            SELECT category,
                   AVG(CASE WHEN passed = 1 THEN 1.0 ELSE 0.0 END) AS rate,
                   SUM(CASE WHEN passed IS NOT NULL THEN 1 ELSE 0 END) AS n_graded,
                   AVG(tok_per_sec) AS tps,
                   SUM(CASE WHEN label = 'complied' THEN 1 ELSE 0 END) AS complied,
                   SUM(CASE WHEN label = 'hedged'   THEN 1 ELSE 0 END) AS hedged,
                   SUM(CASE WHEN label = 'refused'  THEN 1 ELSE 0 END) AS refused
            FROM results WHERE run_id = ? GROUP BY category
            """,
            (run["id"],),
        ).fetchall()
        for r in rows:  # this run is newer than anything stored: overwrite its categories
            labels = {k: r[k] for k in ("complied", "hedged", "refused") if r[k]}
            g.categories[r["category"]] = CatStats(
                rate=r["rate"] if r["n_graded"] else None, tps=r["tps"], labels=labels)
    return groups


def _sweep(groups: dict[tuple, GroupStats]) -> list[GroupStats]:
    """The (model_name, lineage) family with the most distinct quants = the sweep."""
    fams: dict[tuple, list[GroupStats]] = {}
    for (model, quant, lineage), g in groups.items():
        if quant:
            fams.setdefault((model, lineage), []).append(g)
    if not fams:
        return []
    best = max(fams.values(), key=lambda f: len({g.quant for g in f}))
    return sorted(best, key=lambda g: quant_rank(g.quant))


def _curve(sweep: list[GroupStats], categories: list[str], title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    quants = [g.quant for g in sweep]
    for cat in categories:
        ys = [g.categories[cat].rate * 100 if cat in g.categories
              and g.categories[cat].rate is not None else None for g in sweep]
        ax.plot(quants, ys, marker="o", label=cat)
    ax.set_ylabel("pass rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, **_FIG_KW)
    plt.close(fig)
    return path


def chart_quant_curve(conn, out_dir: Path) -> Path | str:
    sweep = _sweep(merged_stats(conn))
    if len(sweep) < 3:
        return "needs >=3 quants of one model (run the sweep)"
    cats = sorted({c for g in sweep for c, s in g.categories.items()
                   if s.rate is not None and not c.startswith("toolcall")})
    if not cats:
        return "no capability categories yet"
    g0 = sweep[0]
    return _curve(sweep, cats,
                  f"{g0.model_name} ({g0.lineage}) - capability vs quantization",
                  out_dir / "quant_curve.png")


def chart_toolcall_curve(conn, out_dir: Path) -> Path | str:
    sweep = _sweep(merged_stats(conn))
    sweep = [g for g in sweep if any(c.startswith("toolcall") for c in g.categories)]
    if len(sweep) < 3:
        return "needs >=3 quants with toolcall results (run --only 'toolcall_*' across quants)"
    cats = sorted({c for g in sweep for c, s in g.categories.items()
                   if s.rate is not None and c.startswith("toolcall")})
    g0 = sweep[0]
    return _curve(sweep, cats,
                  f"{g0.model_name} ({g0.lineage}) - tool calling vs quantization",
                  out_dir / "toolcall_curve.png")


def _matched_pair(groups: dict[tuple, GroupStats]) -> tuple[GroupStats, GroupStats] | None:
    """A (base, abliterated) pair at the same quant, preferring higher-fidelity quants."""
    base = {g.quant: g for g in groups.values() if g.lineage == "base" and g.quant}
    ablit = {g.quant: g for g in groups.values() if g.lineage == "abliterated" and g.quant}
    shared = sorted(set(base) & set(ablit), key=quant_rank)
    if not shared:
        return None
    q = shared[-1]
    return base[q], ablit[q]


def chart_ablit_delta(conn, out_dir: Path) -> Path | str:
    pair = _matched_pair(merged_stats(conn))
    if not pair:
        return "needs a base + abliterated run at the same quant"
    b, a = pair
    cats = sorted(c for c in set(b.categories) & set(a.categories)
                  if b.categories[c].rate is not None and a.categories[c].rate is not None)
    if not cats:
        return "matched runs share no graded categories"

    x = range(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(cats)), 5))
    rb = [b.categories[c].rate * 100 for c in cats]
    ra = [a.categories[c].rate * 100 for c in cats]
    ax.bar([i - w / 2 for i in x], rb, w, label=f"base [{b.quant}]", color="#4878a8")
    ax.bar([i + w / 2 for i in x], ra, w, label=f"abliterated [{a.quant}]", color="#c44e52")
    for i, c in enumerate(cats):
        ax.annotate(f"{ra[i] - rb[i]:+.0f}pp", (i, max(rb[i], ra[i]) + 2),
                    ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(list(x), cats, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("pass rate (%)")
    ax.set_ylim(0, 112)
    ax.set_title(f"{b.model_name} - what abliteration cost, by category")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    path = out_dir / "ablit_delta.png"
    fig.savefig(path, **_FIG_KW)
    plt.close(fig)
    return path


def chart_refusal_profile(conn, out_dir: Path) -> Path | str:
    groups = [g for g in merged_stats(conn).values() if g.label_counts()]
    if not groups:
        return "no refusal-graded results yet"
    groups = sorted(groups, key=lambda g: (g.lineage != "base", quant_rank(g.quant)))

    labels = [f"{g.lineage}\n[{g.quant}]" for g in groups]
    order = ["complied", "hedged", "refused"]
    colors = {"complied": "#55a868", "hedged": "#dd8452", "refused": "#c44e52"}
    counts = [g.label_counts() for g in groups]
    totals = [sum(c.values()) for c in counts]

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(groups)), 5))
    bottom = [0.0] * len(groups)
    for lab in order:
        vals = [100 * c.get(lab, 0) / t for c, t in zip(counts, totals)]
        ax.bar(labels, vals, bottom=bottom, label=lab, color=colors[lab])
        for i, v in enumerate(vals):
            if v >= 6:
                ax.annotate(f"{v:.0f}%", (i, bottom[i] + v / 2),
                            ha="center", va="center", fontsize=9, color="white")
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("share of refusal-eval prompts (%)")
    ax.set_title("Refusal profile - complied / hedged / refused")
    ax.legend(loc="lower right")
    path = out_dir / "refusal_profile.png"
    fig.savefig(path, **_FIG_KW)
    plt.close(fig)
    return path


def chart_pareto(conn, out_dir: Path) -> Path | str:
    points = []
    for g in merged_stats(conn).values():
        rate, tps = g.capability_rate(), g.avg_tps()
        if g.quant and rate is not None and tps:
            points.append((g, rate, tps))
    if len(points) < 3:
        return "needs >=3 runs with capability results"

    fig, ax = plt.subplots(figsize=(8, 5))
    for g, rate, tps in points:
        gb = (g.model_size_bytes or 0) / 1e9
        color = "#c44e52" if g.lineage == "abliterated" else "#4878a8"
        ax.scatter(tps, rate * 100, s=120 * max(gb, 0.3), color=color, alpha=0.75, zorder=3)
        ax.annotate(f" {g.quant} ({gb:.1f} GB)", (tps, rate * 100), fontsize=9)
    ax.set_xlabel("generation speed (tok/s, server-reported)")
    ax.set_ylabel("overall capability pass rate (%)")
    ax.set_title("Quality vs speed - where the knee is (marker size = file size)")
    ax.grid(True, alpha=0.3)
    path = out_dir / "pareto.png"
    fig.savefig(path, **_FIG_KW)
    plt.close(fig)
    return path


def chart_ppl_curve(conn, out_dir: Path) -> Path | str:
    """Perplexity vs quant - the intrinsic metric; moves smoothly where task scores jump."""
    sweep = [g for g in _sweep(merged_stats(conn)) if g.ppl]
    if len(sweep) < 3:
        return "needs >=3 runs with stored ppl (run `crucible ppl <model>`)"
    chunk_counts = {g.ppl_chunks for g in sweep}
    if len(chunk_counts) > 1:
        return f"mixed ppl_chunks {sorted(chunk_counts)} - values not comparable; re-measure"

    fig, ax = plt.subplots(figsize=(8, 5))
    quants = [g.quant for g in sweep]
    ax.plot(quants, [g.ppl for g in sweep], marker="o", color="#4878a8")
    for g in sweep:
        ax.annotate(f" {g.ppl:.2f}", (g.quant, g.ppl), fontsize=9)
    ax.set_ylabel(f"WikiText-2 perplexity ({sweep[0].ppl_chunks} chunks, lower = better)")
    ax.set_title(f"{sweep[0].model_name} ({sweep[0].lineage}) - perplexity vs quantization")
    ax.grid(True, alpha=0.3)
    path = out_dir / "ppl_curve.png"
    fig.savefig(path, **_FIG_KW)
    plt.close(fig)
    return path


CHARTS = {
    "quant_curve": chart_quant_curve,
    "toolcall_curve": chart_toolcall_curve,
    "ablit_delta": chart_ablit_delta,
    "refusal_profile": chart_refusal_profile,
    "pareto": chart_pareto,
    "ppl_curve": chart_ppl_curve,
}


def render_all(conn, out_dir: str | Path) -> dict[str, Path | str]:
    """Render every chart that has data. Returns name -> Path (written) or str (skip reason)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {name: fn(conn, out) for name, fn in CHARTS.items()}
