"""Seed tests/ from the standard datasets the papers use — reproducibly.

Pulls via the Hugging Face datasets-server API (no auth, no `datasets` dependency):

  - GSM8K (openai/gsm8k, test split)  -> tests/gsm8k.yaml   [numeric grader]
  - XSTest (Paul/XSTest, 450 prompts) -> tests/xstest.yaml  [refusal grader]

Sampling is deterministic (fixed SEED) so the suite is reproducible: re-running this
script always produces the same files. Hand-written starter suites (math/code/
instruction/refusal) are left in place as separate categories.

Usage:  uv run python scripts/seed_tests.py [--gsm8k-n 20] [--xstest-n 40]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

SEED = 20260610  # fixed: sampling must be reproducible across machines
API = "https://datasets-server.huggingface.co/rows"
TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"

_GSM8K_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def fetch_rows(dataset: str, config: str, split: str, total: int) -> list[dict]:
    """Page through datasets-server (max 100 rows/page) and return `total` rows."""
    rows: list[dict] = []
    while len(rows) < total:
        qs = urllib.parse.urlencode({
            "dataset": dataset, "config": config, "split": split,
            "offset": len(rows), "length": min(100, total - len(rows)),
        })
        with urllib.request.urlopen(f"{API}?{qs}", timeout=30) as resp:
            page = json.load(resp)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gsm8k-n", type=int, default=20)
    ap.add_argument("--xstest-n", type=int, default=40)
    args = ap.parse_args()
    n1 = seed_gsm8k(args.gsm8k_n)
    print(f"wrote tests/gsm8k.yaml   ({n1} tests)")
    n2 = seed_xstest(args.xstest_n)
    print(f"wrote tests/xstest.yaml  ({n2} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
