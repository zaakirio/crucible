"""Raw JSONL artifact export for stored Crucible runs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from . import db
from .runner import load_tests, test_messages


def _rowdict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def _test_index(tests_dir: str | Path | None) -> dict[tuple[str, str], dict]:
    if tests_dir is None:
        return {}
    return {(category, test["id"]): test for category, test in load_tests(Path(tests_dir))}


def _parse_stored_response(response: str | None) -> tuple[str | None, list[dict] | None, dict | None]:
    if not response:
        return response, None, None
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return response, None, None
    if isinstance(payload, dict) and "tool_calls" in payload:
        final = payload.get("final")
        content = final if isinstance(final, str) else payload.get("content")
        tool_calls = payload.get("tool_calls") if isinstance(payload.get("tool_calls"), list) else None
        return content, tool_calls, payload
    return response, None, payload if isinstance(payload, dict) else None


def export_rows(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    tests_dir: str | Path | None = None,
    docs_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return JSON-serializable raw artifact rows for one run."""
    run = db.get_run(conn, run_id)
    if run is None:
        raise ValueError(f"run #{run_id} not found")
    tests = _test_index(tests_dir)
    rows: list[dict[str, Any]] = []
    for result in db.results_for_run(conn, run_id):
        test = tests.get((result["category"], result["test_id"]))
        messages = None
        fixture = None
        if test is not None:
            fixture = {
                k: v for k, v in test.items()
                if k not in {"tools", "messages", "conversation"}
            }
            messages = test_messages(test, docs_dir=docs_dir)
        response_text, tool_calls, response_json = _parse_stored_response(result["response"])
        rows.append({
            "run": {
                "id": run["id"],
                "model_file": run["model_file"],
                "model_name": run["model_name"],
                "quant": run["quant"],
                "lineage": run["lineage"],
                "hardware": run["hardware"],
                "llama_cpp_commit": run["llama_cpp_commit"],
                "ctx": run["ctx"],
                "ngl": run["ngl"],
                "repeat": run["repeat"],
                "model_sha256": run["model_sha256"],
                "tests_sha256": run["tests_sha256"],
                "docs_sha256": run["docs_sha256"],
                "crucible_version": run["crucible_version"],
            },
            "result": _rowdict(result),
            "fixture": fixture,
            "messages": messages,
            "response_text": response_text,
            "tool_calls": tool_calls,
            "response_json": response_json,
        })
    return rows


def render_jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def write_export(text: str, path: str | Path | None) -> None:
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
