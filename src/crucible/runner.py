"""Run a test suite against one model and store every result.

A run spawns llama-server once, sends each test prompt (optionally `repeat` times for the
noise check), grades the response, and appends rows to SQLite.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import db
from . import __version__
from .client import ChatResult, ServerError, ToolCall, chat
from .graders import GradeResult, grade, grade_tool_call
from .server import llama_cpp_commit, llama_server, external_server
from .retrieval import retrieve_context

# Abort a run after this many back-to-back transport/server errors: once llama-server's
# Metal backend OOMs it stays wedged, so grinding out hundreds of identical error rows
# helps nobody. The partial run stays resumable.
_MAX_CONSECUTIVE_ERRORS = 8

_DEFAULT_MAX_TOKENS = 512


class RunAborted(RuntimeError):
    """A run was stopped because the server became unusable (died / repeated errors)."""


def _server_dead(srv) -> bool:
    proc = getattr(srv, "proc", None)
    return proc is not None and proc.poll() is not None


def _abort_check(srv, errors_in_a_row: int) -> None:
    """Stop the run if the server has died or keeps erroring. Raises RunAborted."""
    if _server_dead(srv):
        code = getattr(getattr(srv, "proc", None), "returncode", None)
        raise RunAborted(
            f"llama-server exited mid-run (exit code {code}). The usual cause is GPU "
            "out-of-memory. Free memory or lower --ngl/--ctx, then re-run with --resume "
            "to continue this run."
        )
    if errors_in_a_row >= _MAX_CONSECUTIVE_ERRORS:
        raise RunAborted(
            f"{errors_in_a_row} consecutive server errors - aborting. Check llama-server, "
            "then re-run with --resume to continue this run."
        )

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


def load_tests(
    tests_dir: Path,
    only: set[str] | None = None,
    suite_defaults: dict[str, dict] | None = None,
) -> list[tuple[str, dict]]:
    """Load every tests/*.yaml file. Category = filename stem. Returns [(category, test), ...].

    `only` restricts to the named categories; a name with a trailing * is a prefix match
    (e.g. "toolcall_*" selects every tool-calling suite).

    `suite_defaults` is a {category: {key: value}} mapping sourced from crucible.yaml's
    `suite_defaults` block. Priority order (lowest to highest):
      suite_defaults config -> YAML-file suite-level keys -> individual test keys.
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
        raw = yaml.safe_load(path.read_text()) or []
        # Support both a plain list and a {max_tokens: N, tests: [...]} dict.
        if isinstance(raw, dict):
            file_suite_level = {k: v for k, v in raw.items() if k != "tests"}
            items = raw.get("tests", [])
        else:
            file_suite_level = {}
            items = raw
        config_cat_defaults = (suite_defaults or {}).get(category, {})
        for t in items:
            merged = {**config_cat_defaults, **file_suite_level, **t}
            out.append((category, merged))
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


def _run_agent_tool_test(
    base_url: str, test: dict, *, docs_dir: str | Path | None = None,
    max_tokens: int = _DEFAULT_MAX_TOKENS, model: str | None = None,
) -> tuple[ChatResult, GradeResult]:
    messages = test_messages(test, docs_dir=docs_dir)
    first = chat(base_url, messages, tools=test.get("tools"), max_tokens=max_tokens, model=model)
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
    final = chat(base_url, [*messages, _assistant_tool_call_message(first), *tool_messages],
                 max_tokens=max_tokens, model=model)
    final.prompt_tokens = _combine_counts(first.prompt_tokens, final.prompt_tokens)
    final.completion_tokens = _combine_counts(first.completion_tokens, final.completion_tokens)
    final.text = _stored_tool_loop_response(first, final)

    final_answer = json.loads(final.text)["final"]
    g = grade(test, final_answer, final.tool_calls)
    if g.passed:
        g.detail = f"tool loop ok; {g.detail}"
    return final, g


def _execute_test(
    base_url: str, category: str, test: dict, rep: int, docs_dir: str | Path | None,
    model: str | None = None,
) -> tuple[str, dict, int, ChatResult | None, GradeResult, int, str, bool]:
    """Run one test against the server. Returns (category, test, rep, res, g, latency_ms, stored, is_error).

    Never raises - errors are captured into GradeResult so the pool can always collect results.
    Thread-safe: touches no shared mutable state.
    """
    max_tokens = test.get("max_tokens", _DEFAULT_MAX_TOKENS)
    t0 = time.perf_counter()
    try:
        if test.get("agent_tool"):
            res, g = _run_agent_tool_test(base_url, test, docs_dir=docs_dir,
                                          max_tokens=max_tokens, model=model)
        else:
            res = chat(
                base_url,
                test_messages(test, docs_dir=docs_dir),
                tools=test.get("tools"),
                max_tokens=max_tokens,
                model=model,
            )
            g = None
    except ServerError as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        body = e.body[:200]
        g = GradeResult(
            passed=None if test.get("grader") == "refusal" else False,
            detail=f"server returned {e.status_code}: {body[:120]}",
        )
        return (category, test, rep, None, g, latency_ms, f"<server error {e.status_code}> {body}", True)
    except Exception as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        g = GradeResult(passed=False, detail=f"unexpected {type(e).__name__}: {str(e)[:120]}")
        return (category, test, rep, None, g, latency_ms, f"<unexpected {type(e).__name__}> {str(e)[:200]}", True)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    if g is None:
        try:
            g = grade(test, res.text, res.tool_calls)
        except Exception as e:
            g = GradeResult(passed=False, detail=f"grade error {type(e).__name__}: {str(e)[:120]}")
            return (category, test, rep, res, g, latency_ms, res.text, False)

    stored = res.text
    if res.tool_calls:
        calls = [{"name": c.name, "arguments": c.raw_arguments} for c in res.tool_calls]
        stored = json.dumps({"tool_calls": calls, "content": res.text})

    return (category, test, rep, res, g, latency_ms, stored, False)


def _result_flags(test: dict, res: ChatResult | None) -> str | None:
    if res is None or res.completion_tokens is None:
        return None
    flags = []
    max_tok = test.get("max_tokens", _DEFAULT_MAX_TOKENS)
    if res.completion_tokens >= max_tok - 2:
        flags.append("truncated")
    # short_response only applies to code_exec: a 2-token response can't contain
    # a runnable code block. For numeric graders the model may correctly return
    # just "7" or "42"; for refusal the grader handles brevity directly.
    if res.completion_tokens < 15 and test.get("grader") == "code_exec":
        flags.append("short_response")
    return ",".join(flags) or None


def _prompt_text(test: dict) -> str:
    """User-facing prompt stored alongside the response for downstream judge/report use."""
    if test.get("prompt"):
        return test["prompt"]
    # multi-turn: extract content of the last user message
    for source in ("messages", "conversation"):
        msgs = test.get(source)
        if msgs:
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("role") == "user":
                    return m.get("content", "")
    return ""


def _resolve_tok_per_sec(res: ChatResult | None, latency_ms: int) -> tuple[float | None, str | None]:
    """Return (tok_per_sec, timing_source). Falls back to client-side measurement when the
    server doesn't expose its internal timings (e.g. Ollama, vLLM in external mode)."""
    if res is None:
        return None, None
    if res.tokens_per_second is not None:
        return res.tokens_per_second, "server"
    if res.completion_tokens and latency_ms > 0:
        return res.completion_tokens / (latency_ms / 1000.0), "client"
    return None, None


def _store_result(conn, run_id: int, category: str, test: dict, rep: int,
                  res: ChatResult | None, g: GradeResult, latency_ms: int, stored: str) -> None:
    tok_per_sec, timing_source = _resolve_tok_per_sec(res, latency_ms)
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
        tok_per_sec=tok_per_sec,
        prompt_tokens=res.prompt_tokens if res else None,
        completion_tokens=res.completion_tokens if res else None,
        flags=_result_flags(test, res),
        prompt_text=_prompt_text(test),
        timing_source=timing_source,
    )


def run_suite(
    model_path: str | Path | None = None,
    *,
    server_url: str | None = None,
    model_name: str | None = None,
    engine_tag: str | None = None,
    tests_dir: str | Path = "tests",
    db_path: str | Path = "results.db",
    hardware: str = "unknown",
    repeat: int = 1,
    ngl: int = 99,
    ctx: int = 4096,
    workers: int = 1,
    only: set[str] | None = None,
    resume: bool = False,
    docs_dir: str | Path | None = None,
    suite_defaults: dict[str, dict] | None = None,
    on_progress=None,
) -> int:
    """Run a test suite against a model.

    Two modes:
      Managed (default): pass model_path. Crucible spawns llama-server, loads the GGUF,
        runs tests, and tears down cleanly.
      External: pass server_url + model_name. Crucible connects to an already-running
        OpenAI-compatible server (Ollama, LM Studio, vLLM, etc.). No spawn/kill.
        tok/s is measured client-side and stored with timing_source='client'.
    """
    if server_url and model_path:
        raise ValueError("Pass either model_path (managed) or server_url (external), not both.")
    if server_url and not model_name:
        raise ValueError("--model is required when using --server (no GGUF to infer name from).")
    if not server_url and not model_path:
        raise ValueError("Either model_path or server_url must be provided.")

    external = bool(server_url)
    tests_dir = Path(tests_dir)
    docs_path = Path(docs_dir) if docs_dir is not None else None
    tests = load_tests(tests_dir, only, suite_defaults=suite_defaults)
    if not tests:
        raise SystemExit(f"No tests found under {tests_dir}/")

    if external:
        name = model_name
        quant = None
        lineage = "unknown"
        llama_commit = None
        model_sha256 = None
        model_size_bytes = None
        resolved_model_path = server_url  # store URL as the "model file" identifier
    else:
        resolved_path = Path(model_path).expanduser().resolve()
        name, quant, lineage = parse_model_meta(resolved_path)
        llama_commit = llama_cpp_commit()
        model_sha256 = _file_sha256(resolved_path)
        model_size_bytes = resolved_path.stat().st_size
        resolved_model_path = str(resolved_path)

    tests_sha256 = _tree_sha256(tests_dir)
    docs_sha256 = _tree_sha256(docs_path) if docs_path else None
    only_filter = ",".join(sorted(only)) if only else None

    server_ctx = (
        external_server(server_url)
        if external
        else llama_server(Path(model_path).expanduser().resolve(), ngl=ngl, ctx=ctx, n_parallel=workers)
    )

    with (
        server_ctx as srv,
        contextlib.closing(db.connect(db_path)) as conn,
    ):
        run = (
            db.find_resumeable_run(
                conn,
                model_file=resolved_model_path,
                hardware=hardware,
                ctx=None if external else ctx,
                ngl=None if external else ngl,
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
                model_file=resolved_model_path,
                model_name=name,
                quant=quant,
                lineage=lineage,
                hardware=hardware,
                llama_cpp_commit=llama_commit,
                ctx=None if external else ctx,
                ngl=None if external else ngl,
                repeat=repeat,
                started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                load_time_s=srv.load_time_s,
                model_size_bytes=model_size_bytes,
                model_sha256=model_sha256,
                tests_sha256=tests_sha256,
                docs_sha256=docs_sha256,
                only_filter=only_filter,
                crucible_version=__version__,
                server_url=server_url,
                engine_tag=engine_tag,
            )

        seen = db.result_keys(conn, run_id)

        # Emit progress for already-completed tests (resume case).
        for category, test in tests:
            for rep in range(repeat):
                if (test["id"], rep) in seen and on_progress:
                    on_progress(category, test, rep, GradeResult(passed=True, detail="skipped (resumed)"))

        pending = [
            (category, test, rep)
            for category, test in tests
            for rep in range(repeat)
            if (test["id"], rep) not in seen
        ]

        # In external mode, pass the model name to the API payload. In managed mode,
        # llama-server doesn't need it (it serves exactly one model).
        api_model = model_name if external else None

        errors_in_a_row = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_execute_test, srv.base_url, cat, test, rep, docs_path, api_model): (cat, test, rep)
                for cat, test, rep in pending
            }
            for future in as_completed(futures):
                _abort_check(srv, errors_in_a_row)
                category, test, rep, res, g, latency_ms, stored, is_error = future.result()
                _store_result(conn, run_id, category, test, rep, res, g, latency_ms, stored)
                seen.add((test["id"], rep))
                if is_error:
                    errors_in_a_row += 1
                    _abort_check(srv, errors_in_a_row)
                else:
                    errors_in_a_row = 0
                if on_progress:
                    on_progress(category, test, rep, g)

        db.finish_run(conn, run_id, datetime.now(timezone.utc).isoformat(timespec="seconds"))

    return run_id
