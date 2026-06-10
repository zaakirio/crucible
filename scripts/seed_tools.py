"""Seed tests/toolcall_*.yaml from BFCL (Berkeley Function Calling Leaderboard) v4 data.

Source: the gorilla GitHub repo (the canonical source; the HF mirror is stale at V3).
Data is Apache 2.0 (see repo LICENSE); sampled items are vendored with attribution.

Categories sampled (the static, AST-gradable subset; agentic/multi-turn excluded):

  toolcall_single       BFCL_v4_simple_python + live_simple    one function, one call
  toolcall_multiple     BFCL_v4_multiple + live_multiple       pick the right function (2-4 offered)
  toolcall_parallel     BFCL_v4_parallel + parallel_multiple   several calls in one turn
  toolcall_irrelevance  BFCL_v4_live_irrelevance               correct answer is NO call
  toolcall_relevance    BFCL_v4_live_relevance                 correct answer is to call

Grading (graders.grade_tool_call, BFCL-AST style, on llama-server's parsed tool_calls):
function-name exact match, argument values within BFCL's per-parameter allowed lists
("" in a list marks the parameter optional), no unexpected arguments, all-or-nothing on
parallel, order-insensitive. Irrelevance passes iff no call is emitted.

BFCL function declarations use Python-flavored JSON schema; they're converted to OpenAI
tools format here (dict->object, float->number, tuple->array) and function names with
dots are sanitized to underscores (ground truth is renamed to match).

Usage:  uv run python scripts/seed_tools.py [--single-n 40] [--multiple-n 20]
          [--parallel-n 20] [--irrelevance-n 15] [--relevance-n 5]
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

SEED = 20260610  # same fixed seed as seed_tests.py: reproducible across machines
RAW = ("https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
       "berkeley-function-call-leaderboard/bfcl_eval/data")
TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"

_TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


def fetch_jsonl(name: str) -> list[dict]:
    url = f"{RAW}/{name}.json"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return [json.loads(line) for line in resp.read().decode().splitlines() if line.strip()]
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _convert_schema(node):
    """BFCL's Python-flavored schema -> OpenAI JSON schema, recursively."""
    if isinstance(node, list):
        return [_convert_schema(n) for n in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k == "type" and isinstance(v, str):
            out[k] = _TYPE_MAP.get(v, v)
        else:
            out[k] = _convert_schema(v)
    return out


def _sanitize(name: str) -> str:
    return name.replace(".", "_")


def to_openai_tools(functions: list[dict]) -> list[dict]:
    tools = []
    for fn in functions:
        tools.append({
            "type": "function",
            "function": {
                "name": _sanitize(fn["name"]),
                "description": fn.get("description", ""),
                "parameters": _convert_schema(fn.get("parameters", {"type": "object", "properties": {}})),
            },
        })
    return tools


def first_user_prompt(row: dict) -> str | None:
    """BFCL question shape: [[{role, content}, ...]]. We take the single-turn user content."""
    turns = row.get("question") or []
    if not turns or not turns[0]:
        return None
    users = [m["content"] for m in turns[0] if m.get("role") == "user"]
    return users[0].strip() if users else None


def build_items(rows: list[dict], answers: dict[str, list] | None, *,
                expect_call: bool, prefix: str, n: int, rng: random.Random) -> list[dict]:
    pool = []
    for row in rows:
        prompt = first_user_prompt(row)
        if not prompt or not row.get("function"):
            continue
        if answers is not None and row["id"] not in answers:
            continue
        pool.append((row, prompt))
    sample = rng.sample(pool, min(n, len(pool)))
    items = []
    for i, (row, prompt) in enumerate(sample, 1):
        item = {
            "id": f"{prefix}-{i:03d}",
            "bfcl_id": row["id"],
            "prompt": prompt,
            "grader": "tool_call",
            "expect_call": expect_call,
            "tools": to_openai_tools(row["function"]),
        }
        if answers is not None:
            item["expected_calls"] = [
                {_sanitize(fname): params for fname, params in gt.items()}
                for gt in answers[row["id"]]
            ]
        items.append(item)
    return items


def write_suite(filename: str, what: str, items: list[dict]) -> None:
    header = (
        f"# {what}\n"
        "# Sampled from BFCL v4 (Berkeley Function Calling Leaderboard; Patil et al.),\n"
        "# github.com/ShishirPatil/gorilla, Apache 2.0. Deterministic sample via\n"
        f"# scripts/seed_tools.py (seed {SEED}). Graded BFCL-AST style on llama-server's\n"
        "# parsed tool_calls: name match, args within allowed value lists, all-or-nothing.\n"
    )
    (TESTS_DIR / filename).write_text(
        header + yaml.safe_dump(items, sort_keys=False, width=110, allow_unicode=True))
    print(f"wrote tests/{filename}  ({len(items)} tests)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--single-n", type=int, default=40)
    ap.add_argument("--multiple-n", type=int, default=20)
    ap.add_argument("--parallel-n", type=int, default=20)
    ap.add_argument("--irrelevance-n", type=int, default=15)
    ap.add_argument("--relevance-n", type=int, default=5)
    args = ap.parse_args()
    rng = random.Random(SEED)

    def answers_for(*names: str) -> dict[str, list]:
        out: dict[str, list] = {}
        for name in names:
            for row in fetch_jsonl(f"possible_answer/{name}"):
                out[row["id"]] = row["ground_truth"]
        return out

    # single: one function offered, one call expected
    rows = fetch_jsonl("BFCL_v4_simple_python") + fetch_jsonl("BFCL_v4_live_simple")
    ans = answers_for("BFCL_v4_simple_python", "BFCL_v4_live_simple")
    write_suite("toolcall_single.yaml", "Tool calling: single function, single call.",
                build_items(rows, ans, expect_call=True, prefix="tc-single",
                            n=args.single_n, rng=rng))

    # multiple: several functions offered, pick the right one
    rows = fetch_jsonl("BFCL_v4_multiple") + fetch_jsonl("BFCL_v4_live_multiple")
    ans = answers_for("BFCL_v4_multiple", "BFCL_v4_live_multiple")
    write_suite("toolcall_multiple.yaml", "Tool calling: choose the right function (2-4 offered).",
                build_items(rows, ans, expect_call=True, prefix="tc-multiple",
                            n=args.multiple_n, rng=rng))

    # parallel: several calls expected in one turn
    rows = fetch_jsonl("BFCL_v4_parallel") + fetch_jsonl("BFCL_v4_parallel_multiple")
    ans = answers_for("BFCL_v4_parallel", "BFCL_v4_parallel_multiple")
    write_suite("toolcall_parallel.yaml", "Tool calling: parallel calls in one turn.",
                build_items(rows, ans, expect_call=True, prefix="tc-parallel",
                            n=args.parallel_n, rng=rng))

    # irrelevance: tools offered, none applies; correct behavior is NO call
    rows = fetch_jsonl("BFCL_v4_live_irrelevance")
    write_suite("toolcall_irrelevance.yaml",
                "Tool calling: irrelevance detection (correct answer is NOT to call).",
                build_items(rows, None, expect_call=False, prefix="tc-irrel",
                            n=args.irrelevance_n, rng=rng))

    # relevance: a tool applies; correct behavior is to call (any well-formed call passes)
    rows = fetch_jsonl("BFCL_v4_live_relevance")
    write_suite("toolcall_relevance.yaml",
                "Tool calling: relevance detection (correct answer is to make a call).",
                build_items(rows, None, expect_call=True, prefix="tc-rel",
                            n=args.relevance_n, rng=rng))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
