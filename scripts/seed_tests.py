"""Seed tests/ from the standard datasets the papers use — reproducibly.

Pulls via the Hugging Face datasets-server API (no auth, no `datasets` dependency):

  - GSM8K (openai/gsm8k, test split)            -> tests/gsm8k.yaml        [numeric]
  - GSM-Symbolic (apple/GSM-Symbolic, 5000 pre-generated instances)
                                                -> tests/gsm_symbolic.yaml [numeric]
    Contamination-resistant math (ICLR 2025): regenerated variants of GSM8K-style
    templates. Use for ABSOLUTE capability claims; gsm8k stays for paper-comparable deltas.
  - XSTest (Paul/XSTest, 450 prompts)           -> tests/xstest.yaml       [refusal]
  - OR-Bench-Hard (bench-llm/or-bench, ICML 2025, hard-1k subset)
                                                -> tests/orbench.yaml      [refusal]
    Seemingly-toxic-but-safe prompts: the over-refusal axis, harder than XSTest.
  - FalseReject-Test (AmazonScience/FalseReject, ~1.1k human-annotated)
                                                -> tests/falsereject.yaml  [refusal]
  - SORRY-Bench (sorry-bench/sorry-bench-202503; ICLR 2025)
                                                -> tests/sorrybench.yaml   [refusal]
    Class-balanced UNSAFE instructions (44/45-topic taxonomy) — the refusal-of-unsafe
    axis XSTest doesn't cover; where the abliteration delta should be largest. The
    official repo is gated: set $HF_TOKEN after accepting its terms, else this falls
    back to the ungated SillyTilly/SorryBench mirror of the 2024-06 version.

Sampling is deterministic (fixed SEED) so the suite is reproducible: re-running this
script always produces the same files. Hand-written starter suites (math/code/
instruction/refusal) are left in place as separate categories.

Usage:  uv run python scripts/seed_tests.py [--gsm8k-n 20] [--xstest-n 40]
          [--gsm-symbolic-n 20] [--orbench-n 50] [--falsereject-n 50] [--sorrybench-n 45]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

SEED = 20260610  # fixed: sampling must be reproducible across machines
API = "https://datasets-server.huggingface.co/rows"
TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"

_GSM8K_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def fetch_rows(dataset: str, config: str, split: str, total: int) -> list[dict]:
    """Page through datasets-server (max 100 rows/page) and return `total` rows.

    `total` may overshoot the dataset; it's clamped to num_rows_total after page one.
    $HF_TOKEN (if set) is sent for gated datasets.
    """
    headers = {}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    rows: list[dict] = []
    while len(rows) < total:
        qs = urllib.parse.urlencode({
            "dataset": dataset, "config": config, "split": split,
            "offset": len(rows), "length": min(100, total - len(rows)),
        })
        req = urllib.request.Request(f"{API}?{qs}", headers=headers)
        for attempt in range(9):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    page = json.load(resp)
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 8:
                    raise
                time.sleep(min(60, 2 ** attempt))  # 429: back off up to ~3.5min total
        time.sleep(0.5)  # stay friendly to the shared API
        batch = [r["row"] for r in page["rows"]]
        if not batch:
            break
        rows.extend(batch)
        total = min(total, page["num_rows_total"])
    return rows


def seed_gsm8k(n: int) -> int:
    # Fetch a fixed-size pool from the front of the (stable-ordered) test split,
    # then sample deterministically from it.
    pool = fetch_rows("openai/gsm8k", "main", "test", 200)
    rng = random.Random(SEED)
    sample = rng.sample(pool, n)
    tests = []
    for i, row in enumerate(sample, 1):
        m = _GSM8K_ANSWER_RE.search(row["answer"])
        if not m:
            continue
        expected = float(m.group(1).replace(",", ""))
        tests.append({
            "id": f"gsm8k-{i:03d}",
            "prompt": row["question"].strip()
                      + " Give the final answer as just the number on the last line.",
            "grader": "numeric",
            "expected": int(expected) if expected.is_integer() else expected,
            "tolerance": 0.001,
        })
    header = (
        "# GSM8K (openai/gsm8k, test split) — grade-school math word problems.\n"
        f"# Deterministic sample of {len(tests)} via scripts/seed_tests.py (seed {SEED}).\n"
        "# Same dataset as arXiv 2512.13655 / 2601.14277, so deltas are paper-comparable.\n"
        "# grader: numeric (extracts the LAST number; prompt asks for it on the last line).\n"
    )
    (TESTS_DIR / "gsm8k.yaml").write_text(header + yaml.safe_dump(tests, sort_keys=False, width=100))
    return len(tests)


def seed_xstest(n: int) -> int:
    rows = fetch_rows("Paul/XSTest", "default", "train", 450)
    safe = [r for r in rows if r["label"] == "safe"]
    unsafe = [r for r in rows if r["label"] != "safe"]
    rng = random.Random(SEED)
    # Stratified half/half: safe prompts measure FALSE refusal (over-safety);
    # unsafe ones are where base models refuse — the abliteration delta rows.
    sample = rng.sample(safe, n // 2) + rng.sample(unsafe, n - n // 2)
    tests = []
    for i, row in enumerate(sample, 1):
        tests.append({
            "id": f"xstest-{i:03d}",
            "prompt": row["prompt"].strip(),
            "grader": "refusal",
            "expect": "comply" if row["label"] == "safe" else "comply_if_abliterated",
            "xstest_type": row["type"],
        })
    header = (
        "# XSTest (Paul/XSTest, 450 prompts; Röttger et al.) — exaggerated-safety / refusal profile.\n"
        f"# Deterministic stratified sample of {len(tests)} ({n // 2} safe / {n - n // 2} unsafe) "
        f"via scripts/seed_tests.py (seed {SEED}).\n"
        "# Reported as a complied/hedged/refused PROFILE, not pass/fail. `expect` is context:\n"
        "#   comply                -> safe prompt; refusing it is a FALSE refusal (over-safety)\n"
        "#   comply_if_abliterated -> unsafe prompt; base models refuse, abliterated should comply.\n"
    )
    (TESTS_DIR / "xstest.yaml").write_text(header + yaml.safe_dump(tests, sort_keys=False, width=100, allow_unicode=True))
    return len(tests)


def seed_gsm_symbolic(n: int) -> int:
    """One instance per original template, so n tests cover n distinct problems."""
    rows = fetch_rows("apple/GSM-Symbolic", "main", "test", 5000)
    rng = random.Random(SEED)
    by_original: dict = {}
    for r in rows:
        by_original.setdefault(r["original_id"], []).append(r)
    originals = rng.sample(sorted(by_original), min(n, len(by_original)))
    tests = []
    for i, oid in enumerate(originals, 1):
        row = rng.choice(sorted(by_original[oid], key=lambda r: r["instance"]))
        m = _GSM8K_ANSWER_RE.search(row["answer"])
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", row["answer"])
        raw = m.group(1) if m else (nums[-1] if nums else None)
        if raw is None:
            continue
        expected = float(raw.replace(",", ""))
        tests.append({
            "id": f"gsmsym-{i:03d}",
            "prompt": row["question"].strip()
                      + " Give the final answer as just the number on the last line.",
            "grader": "numeric",
            "expected": int(expected) if expected.is_integer() else expected,
            "tolerance": 0.001,
        })
    header = (
        "# GSM-Symbolic (apple/GSM-Symbolic; Mirzadeh et al., ICLR 2025) — contamination-resistant\n"
        "# math: regenerated instances of GSM8K-style templates. Use these for ABSOLUTE capability\n"
        f"# claims; gsm8k.yaml stays for paper-comparable deltas. One instance per original template,\n"
        f"# deterministic sample of {len(tests)} via scripts/seed_tests.py (seed {SEED}).\n"
    )
    (TESTS_DIR / "gsm_symbolic.yaml").write_text(header + yaml.safe_dump(tests, sort_keys=False, width=100))
    return len(tests)


def seed_orbench(n: int) -> int:
    rows = fetch_rows("bench-llm/or-bench", "or-bench-hard-1k", "train", 1400)
    rng = random.Random(SEED)
    sample = rng.sample(rows, min(n, len(rows)))
    tests = [{
        "id": f"orbench-{i:03d}",
        "prompt": row["prompt"].strip(),
        "grader": "refusal",
        "expect": "comply",
        "orbench_category": row["category"],
    } for i, row in enumerate(sample, 1)]
    header = (
        "# OR-Bench-Hard-1K (bench-llm/or-bench; Cui et al., ICML 2025) — over-refusal.\n"
        "# Seemingly-toxic-but-SAFE prompts that trip cautious models; harder than XSTest.\n"
        f"# All rows expect comply: a refusal here is a false refusal. Deterministic sample of\n"
        f"# {len(tests)} via scripts/seed_tests.py (seed {SEED}). Prompts are LLM-generated (known noise).\n"
    )
    (TESTS_DIR / "orbench.yaml").write_text(header + yaml.safe_dump(tests, sort_keys=False, width=100, allow_unicode=True))
    return len(tests)


def seed_falsereject(n: int) -> int:
    rows = fetch_rows("AmazonScience/FalseReject", "default", "test", 1200)
    rng = random.Random(SEED)
    sample = rng.sample(rows, min(n, len(rows)))
    tests = [{
        "id": f"falsereject-{i:03d}",
        "prompt": row["prompt"].strip(),
        "grader": "refusal",
        "expect": "comply",
        "falsereject_category": row["category_text"],
    } for i, row in enumerate(sample, 1)]
    header = (
        "# FalseReject-Test (AmazonScience/FalseReject, 2025) — over-refusal, human-annotated.\n"
        "# Seemingly-toxic-but-SAFE prompts across 44 categories; elicits more false refusals\n"
        f"# than XSTest on SOTA models. All rows expect comply. Deterministic sample of {len(tests)}\n"
        f"# via scripts/seed_tests.py (seed {SEED}).\n"
    )
    (TESTS_DIR / "falsereject.yaml").write_text(header + yaml.safe_dump(tests, sort_keys=False, width=100, allow_unicode=True))
    return len(tests)


def seed_sorrybench(n: int) -> int:
    """Official gated repo with $HF_TOKEN; ungated 2024-06 mirror otherwise. 1 prompt/category."""
    source = "sorry-bench/sorry-bench-202503 (official, 2025-03)"
    try:
        rows = fetch_rows("sorry-bench/sorry-bench-202503", "default", "train", 1000)
        rows = [r for r in rows if r.get("prompt_style", "base") == "base"]
    except urllib.error.HTTPError:
        source = "SillyTilly/SorryBench (ungated mirror of 2024-06)"
        print("  sorrybench: official repo gated (set $HF_TOKEN after accepting its terms);"
              " falling back to the mirror")
        rows = fetch_rows("SillyTilly/SorryBench", "default", "train", 1000)
        rows = [r for r in rows if r.get("prompt_style") == "base"]
    rng = random.Random(SEED)
    by_cat: dict = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    cats = sorted(by_cat, key=str)[:n]
    tests = []
    for i, cat in enumerate(cats, 1):
        row = rng.choice(sorted(by_cat[cat], key=lambda r: r["question_id"]))
        prompt = row["turns"][0] if isinstance(row.get("turns"), list) else row.get("prompt", "")
        tests.append({
            "id": f"sorrybench-{i:03d}",
            "prompt": prompt.strip(),
            "grader": "refusal",
            "expect": "comply_if_abliterated",
            "sorrybench_category": str(row["category"]),
        })
    header = (
        f"# SORRY-Bench ({source}; Xie et al., ICLR 2025) — refusal-of-UNSAFE instructions,\n"
        "# class-balanced taxonomy, one prompt per category (deterministic, seed "
        f"{SEED}).\n"
        "# This is the axis XSTest barely covers and where the abliteration delta should be\n"
        "# largest: base models refuse these; an abliterated model complies. As ever this is a\n"
        "# PROFILE — the base-vs-abliterated shift is the measurement, not a pass bar.\n"
    )
    (TESTS_DIR / "sorrybench.yaml").write_text(header + yaml.safe_dump(tests, sort_keys=False, width=100, allow_unicode=True))
    return len(tests)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gsm8k-n", type=int, default=20)
    ap.add_argument("--xstest-n", type=int, default=40)
    ap.add_argument("--gsm-symbolic-n", type=int, default=20)
    ap.add_argument("--orbench-n", type=int, default=50)
    ap.add_argument("--falsereject-n", type=int, default=50)
    ap.add_argument("--sorrybench-n", type=int, default=45)
    args = ap.parse_args()
    for name, fn, n in [
        ("gsm8k", seed_gsm8k, args.gsm8k_n),
        ("gsm_symbolic", seed_gsm_symbolic, args.gsm_symbolic_n),
        ("xstest", seed_xstest, args.xstest_n),
        ("orbench", seed_orbench, args.orbench_n),
        ("falsereject", seed_falsereject, args.falsereject_n),
        ("sorrybench", seed_sorrybench, args.sorrybench_n),
    ]:
        wrote = fn(n)
        print(f"wrote tests/{name}.yaml  ({wrote} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
