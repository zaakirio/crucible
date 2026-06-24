"""Run a test suite against one model and store every result.

A run spawns llama-server once, sends each test prompt (optionally `repeat` times for the
noise check), grades the response, and appends rows to SQLite.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

from . import db
from . import __version__
from .client import ChatResult, ToolCall, chat
from .graders import GradeResult, grade, grade_tool_call
from .server import llama_cpp_commit, llama_server
from .retrieval import retrieve_context

_QUANT_RE = re.compile(
    r"(IQ\d+_[A-Z]+|Q\d+_K_[SML]|Q\d+_K|Q\d+_\d+|BF16|F16|F32)", re.IGNORECASE
)
_ABLITERATED_MARKERS = ("uncensored", "abliterat", "heretic", "decensored", "deccp")
_HASH_EXTS = {".yaml", ".yml", ".md", ".markdown", ".txt", ".rst"}


def parse_model_meta(model_path: Path) -> tuple[str, str | None, str]:
    """(model_name, quant, lineage) parsed from the filename."""
    stem = model_path.stem
    quant_match = None
    for m in _QUANT_RE.finditer(stem):
        quant_match = m
    quant = quant_match.group(1) if quant_match else None
    name = stem
    if quant_match:
        # Remove the matched quant token anywhere in the stem, then normalize separators.
        name = (stem[:quant_match.start()] + stem[quant_match.end():]).strip("._- ")
        name = re.sub(r"[-_.]{2,}", "-", name)
        name = name.strip("._- ")
    lineage = "abliterated" if any(k in stem.lower() for k in _ABLITERATED_MARKERS) else "base"
    return name, quant, lineage


def load_tests(tests_dir: Path, only: set[str] | None = None) -> list[tuple[str, dict]]:
    """Load every tests/*.yaml file. Category = filename stem. Returns [(category, test), ...].

    `only` restricts to the named categories; a name with a trailing * is a prefix match
    (e.g. "toolcall_*" selects every tool-calling suite).
    """
    def wanted(category: str) -> bool:
        if only is None:
            return True
        return any(category == o or (o.endswith("*") and category.startswith(o[:-1]))
                   for o in only)

    out: list[tuple[str, dict]] = []
    for path in sorted(tests_dir.glob("*.yaml")):
        category = path.stem
        if not wanted(category):
            continue
        items = yaml.safe_load(path.read_text()) or []
        for t in items:
            out.append((category, t))
    return out


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_sha256(root: Path, *, exts: set[str] = _HASH_EXTS) -> str | None:
    if not root.exists():
        return None
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts):
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def test_messages(test: dict, *, docs_dir: str | Path | None = None) -> list[dict]:
    """Return the chat payload messages for one test.

    Most fixtures are still single-turn prompts, but grounded QA / RAG-style fixtures can
    supply a full message list, and agent-style fixtures can use an explicit conversation
    transcript so the harness can evaluate context-aware answers directly.
    """
    if test.get("messages"):
        return test["messages"]
    if test.get("conversation"):
        return test["conversation"]
    if test.get("retrieval"):
        if docs_dir is None:
            raise ValueError(f"test {test.get('id')!r} requests retrieval but no docs_dir was set")
        query = test.get("prompt", "")
        context = retrieve_context(query, docs_dir, top_k=int(test.get("top_k", 3)))
        system = test.get(
            "system",
            "Answer using only the retrieved context. Cite sources using their bracketed source markers, "
            "such as [file.md#0], when the question asks for a citation. If the answer is not in the "
            "context, say you don't know.",
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Retrieved context:\n{context}\n\nQuestion: {query}"},
        ]
    return [{"role": "user", "content": test["prompt"]}]


def _tool_call_id(call: ToolCall, index: int) -> str:
    return call.id or f"call_{index}"


def _assistant_tool_call_message(res: ChatResult) -> dict:
    calls = []
    for i, call in enumerate(res.tool_calls):
        calls.append({
            "id": _tool_call_id(call, i),
            "type": "function",
            "function": {"name": call.name, "arguments": call.raw_arguments},
        })
    return {"role": "assistant", "content": res.text, "tool_calls": calls}


def _tool_result_messages(test: dict, calls: list[ToolCall]) -> list[dict]:
    results = test.get("tool_results") or {}
    out: list[dict] = []
    for i, call in enumerate(calls):
        if call.arguments is None:
            raise ValueError(f"tool {call.name!r} arguments are not valid JSON")
        if call.name not in results:
            raise ValueError(f"no mocked result configured for tool {call.name!r}")
        content = results[call.name]
        if not isinstance(content, str):
            content = json.dumps(content, sort_keys=True)
        out.append({
            "role": "tool",
            "tool_call_id": _tool_call_id(call, i),
            "name": call.name,
            "content": content,
        })
    return out


def _combine_counts(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


def _stored_tool_loop_response(first: ChatResult, final: ChatResult) -> str:
    calls = [
        {"id": _tool_call_id(c, i), "name": c.name, "arguments": c.raw_arguments}
        for i, c in enumerate(first.tool_calls)
    ]
    return json.dumps({"tool_calls": calls, "final": final.text}, sort_keys=True)


def _run_agent_tool_test(base_url: str, test: dict, *, docs_dir: str | Path | None = None) -> tuple[ChatResult, GradeResult]:
    messages = test_messages(test, docs_dir=docs_dir)
    first = chat(base_url, messages, tools=test.get("tools"))
    call_grade = grade_tool_call(
        {
            "grader": "tool_call",
            "expect_call": test.get("expect_call", True),
            "expected_calls": test.get("expected_calls"),
        },
        first.text,
        first.tool_calls,
    )
    if not call_grade.passed:
        return first, call_grade
    if test.get("expect_call") is False:
        return first, call_grade
    if not first.tool_calls:
        return first, GradeResult(passed=False, detail="agent_tool test made no tool call")

    tool_messages = _tool_result_messages(test, first.tool_calls)
    final = chat(base_url, [*messages, _assistant_tool_call_message(first), *tool_messages])
    final.prompt_tokens = _combine_counts(first.prompt_tokens, final.prompt_tokens)
    final.completion_tokens = _combine_counts(first.completion_tokens, final.completion_tokens)
    final.text = _stored_tool_loop_response(first, final)

    final_answer = json.loads(final.text)["final"]
    g = grade(test, final_answer, final.tool_calls)
    if g.passed:
        g.detail = f"tool loop ok; {g.detail}"
    return final, g


def run_suite(
    model_path: str | Path,
    *,
    tests_dir: str | Path = "tests",
    db_path: str | Path = "results.db",
    hardware: str = "unknown",
    repeat: int = 1,
    ngl: int = 99,
    ctx: int = 4096,
    only: set[str] | None = None,
    resume: bool = False,
    docs_dir: str | Path | None = None,
    on_progress=None,
) -> int:
    model_path = Path(model_path).expanduser().resolve()
    tests_dir = Path(tests_dir)
    docs_path = Path(docs_dir) if docs_dir is not None else None
    tests = load_tests(tests_dir, only)
    if not tests:
        raise SystemExit(f"No tests found under {tests_dir}/")

    name, quant, lineage = parse_model_meta(model_path)
    conn = db.connect(db_path)
    llama_commit = llama_cpp_commit()
    model_sha256 = _file_sha256(model_path)
    tests_sha256 = _tree_sha256(tests_dir)
    docs_sha256 = _tree_sha256(docs_path) if docs_path else None
    only_filter = ",".join(sorted(only)) if only else None

    # This is the production execution path: swap in a temporary mock server for tests,
    # or let this context manager spawn a real llama.cpp `llama-server` binary for local runs.
    with llama_server(model_path, ngl=ngl, ctx=ctx) as srv:
        run = (
            db.find_resumeable_run(
                conn,
                model_file=str(model_path),
                hardware=hardware,
                ctx=ctx,
                ngl=ngl,
                repeat=repeat,
                llama_cpp_commit=llama_commit,
                model_sha256=model_sha256,
                tests_sha256=tests_sha256,
                docs_sha256=docs_sha256,
                only_filter=only_filter,
                crucible_version=__version__,
            )
            if resume
            else None
        )
        if run:
            run_id = run["id"]
        else:
            run_id = db.create_run(
                conn,
                model_file=str(model_path),
                model_name=name,
                quant=quant,
                lineage=lineage,
                hardware=hardware,
                llama_cpp_commit=llama_commit,
                ctx=ctx,
                ngl=ngl,
                repeat=repeat,
                started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                load_time_s=srv.load_time_s,
                model_size_bytes=model_path.stat().st_size,
                model_sha256=model_sha256,
                tests_sha256=tests_sha256,
                docs_sha256=docs_sha256,
                only_filter=only_filter,
                crucible_version=__version__,
            )
        seen = db.result_keys(conn, run_id)
        for category, test in tests:
            for rep in range(repeat):
                if (test["id"], rep) in seen:
                    if on_progress:
                        on_progress(category, test, rep, GradeResult(passed=True, detail="skipped (resumed)"))
                    continue
                t0 = time.perf_counter()
                try:
                    if test.get("agent_tool"):
                        res, g = _run_agent_tool_test(srv.base_url, test, docs_dir=docs_dir)
                    else:
                        res = chat(
                            srv.base_url,
                            test_messages(test, docs_dir=docs_dir),
                            tools=test.get("tools"),
                        )
                        g = None
                except httpx.HTTPStatusError as e:
                    # The server answered but errored (e.g. its tool-call parser choked on
                    # the model's output). That's a finding about this model/quant, not a
                    # harness crash: record it as a failure and keep going.
                    latency_ms = int((time.perf_counter() - t0) * 1000)
                    body = e.response.text[:200]
                    g = grade_error = GradeResult(
                        passed=None if test.get("grader") == "refusal" else False,
                        detail=f"server returned {e.response.status_code}: {body[:120]}",
                    )
                    db.insert_result(
                        conn, run_id=run_id, test_id=test["id"], category=category, rep=rep,
                        response=f"<server error {e.response.status_code}> {body}",
                        passed=None if grade_error.passed is None else int(grade_error.passed),
                        label=None, detail=grade_error.detail, latency_ms=latency_ms,
                        tok_per_sec=None, prompt_tokens=None, completion_tokens=None,
                    )
                    if on_progress:
                        on_progress(category, test, rep, g)
                    continue
                except Exception as e:
                    latency_ms = int((time.perf_counter() - t0) * 1000)
                    g = GradeResult(
                        passed=False,
                        detail=f"unexpected {type(e).__name__}: {str(e)[:120]}",
                    )
                    db.insert_result(
                        conn, run_id=run_id, test_id=test["id"], category=category, rep=rep,
                        response=f"<unexpected {type(e).__name__}> {str(e)[:200]}",
                        passed=0, label=None, detail=g.detail, latency_ms=latency_ms,
                        tok_per_sec=None, prompt_tokens=None, completion_tokens=None,
                    )
                    if on_progress:
                        on_progress(category, test, rep, g)
                    continue
                latency_ms = int((time.perf_counter() - t0) * 1000)
                if g is None:
                    try:
                        g = grade(test, res.text, res.tool_calls)
                    except Exception as e:
                        g = GradeResult(
                            passed=False,
                            detail=f"grade error {type(e).__name__}: {str(e)[:120]}",
                        )
                        db.insert_result(
                            conn, run_id=run_id, test_id=test["id"], category=category, rep=rep,
                            response=res.text,
                            passed=0, label=None, detail=g.detail, latency_ms=latency_ms,
                            tok_per_sec=res.tokens_per_second, prompt_tokens=res.prompt_tokens,
                            completion_tokens=res.completion_tokens,
                        )
                        if on_progress:
                            on_progress(category, test, rep, g)
                        continue
                stored = res.text
                if res.tool_calls:  # tool-call turns have empty text; store what was called
                    calls = [{"name": c.name, "arguments": c.raw_arguments} for c in res.tool_calls]
                    stored = json.dumps({"tool_calls": calls, "content": res.text})
                db.insert_result(
                    conn,
                    run_id=run_id,
                    test_id=test["id"],
                    category=category,
                    rep=rep,
                    response=stored,
                    passed=None if g.passed is None else int(g.passed),
                    label=g.label,
                    detail=g.detail,
                    latency_ms=latency_ms,
                    tok_per_sec=res.tokens_per_second,
                    prompt_tokens=res.prompt_tokens,
                    completion_tokens=res.completion_tokens,
                )
                seen.add((test["id"], rep))
                if on_progress:
                    on_progress(category, test, rep, g)
        db.finish_run(conn, run_id, datetime.now(timezone.utc).isoformat(timespec="seconds"))

    conn.close()
    return run_id
