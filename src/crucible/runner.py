"""Run a test suite against one model and store every result.

A run spawns llama-server once, sends each test prompt (optionally `repeat` times for the
noise check), grades the response, and appends rows to SQLite.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import db
from .client import chat
from .graders import grade
from .server import llama_cpp_commit, llama_server

_QUANT_RE = re.compile(
    r"(IQ\d+_[A-Z]+|Q\d+_K_[SML]|Q\d+_K|Q\d+_\d+|BF16|F16|F32)", re.IGNORECASE
)
_ABLITERATED_MARKERS = ("uncensored", "abliterat", "heretic", "decensored", "deccp")


def parse_model_meta(model_path: Path) -> tuple[str, str | None, str]:
    """(model_name, quant, lineage) parsed from the filename."""
    stem = model_path.stem
    m = _QUANT_RE.search(stem)
    quant = m.group(1) if m else None
    name = stem
    if quant:
        # strip the quant token and any trailing separators
        name = re.sub(r"[-_.]?" + re.escape(quant) + r"$", "", stem, flags=re.IGNORECASE)
    lineage = "abliterated" if any(k in stem.lower() for k in _ABLITERATED_MARKERS) else "base"
    return name, quant, lineage


def load_tests(tests_dir: Path) -> list[tuple[str, dict]]:
    """Load every tests/*.yaml file. Category = filename stem. Returns [(category, test), ...]."""
    out: list[tuple[str, dict]] = []
    for path in sorted(tests_dir.glob("*.yaml")):
        category = path.stem
        items = yaml.safe_load(path.read_text()) or []
        for t in items:
            out.append((category, t))
    return out


def run_suite(
    model_path: str | Path,
    *,
    tests_dir: str | Path = "tests",
    db_path: str | Path = "results.db",
    hardware: str = "unknown",
    repeat: int = 1,
    ngl: int = 99,
    ctx: int = 4096,
    on_progress=None,
) -> int:
    model_path = Path(model_path).expanduser().resolve()
    tests_dir = Path(tests_dir)
    tests = load_tests(tests_dir)
    if not tests:
        raise SystemExit(f"No tests found under {tests_dir}/")

    name, quant, lineage = parse_model_meta(model_path)
    conn = db.connect(db_path)

    with llama_server(model_path, ngl=ngl, ctx=ctx) as srv:
        run_id = db.create_run(
            conn,
            model_file=str(model_path),
            model_name=name,
            quant=quant,
            lineage=lineage,
            hardware=hardware,
            llama_cpp_commit=llama_cpp_commit(),
            ctx=ctx,
            ngl=ngl,
            repeat=repeat,
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            load_time_s=srv.load_time_s,
            model_size_bytes=model_path.stat().st_size,
        )
        for category, test in tests:
            for rep in range(repeat):
                t0 = time.perf_counter()
                res = chat(srv.base_url, [{"role": "user", "content": test["prompt"]}])
                latency_ms = int((time.perf_counter() - t0) * 1000)
                g = grade(test, res.text)
                db.insert_result(
                    conn,
                    run_id=run_id,
                    test_id=test["id"],
                    category=category,
                    rep=rep,
                    response=res.text,
                    passed=None if g.passed is None else int(g.passed),
                    label=g.label,
                    detail=g.detail,
                    latency_ms=latency_ms,
                    tok_per_sec=res.tokens_per_second,
                    prompt_tokens=res.prompt_tokens,
                    completion_tokens=res.completion_tokens,
                )
                if on_progress:
                    on_progress(category, test, rep, g)
        db.finish_run(conn, run_id, datetime.now(timezone.utc).isoformat(timespec="seconds"))

    conn.close()
    return run_id
