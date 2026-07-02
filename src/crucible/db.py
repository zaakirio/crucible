"""SQLite storage - append-only. A 'run' is one model evaluated under one set of settings;
comparisons (base vs abliterated, Q4 vs Q8) are just queries over runs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id               INTEGER PRIMARY KEY,
  model_file       TEXT NOT NULL,
  model_name       TEXT,
  quant            TEXT,
  lineage          TEXT,          -- 'base' | 'abliterated'
  hardware         TEXT,
  llama_cpp_commit TEXT,
  ctx              INTEGER,
  ngl              INTEGER,
  repeat           INTEGER,
  started_at       TEXT,
  finished_at      TEXT,
  load_time_s      REAL,
  model_size_bytes INTEGER
);
CREATE TABLE IF NOT EXISTS results (
  id                INTEGER PRIMARY KEY,
  run_id            INTEGER NOT NULL REFERENCES runs(id),
  test_id           TEXT NOT NULL,
  category          TEXT NOT NULL,
  rep               INTEGER NOT NULL,   -- which repetition (0-based) for noise checks
  response          TEXT,
  passed            INTEGER,            -- 1/0, NULL for refusal tests
  label             TEXT,               -- complied/refused/hedged for refusal tests
  detail            TEXT,
  latency_ms        INTEGER,
  tok_per_sec       REAL,
  prompt_tokens     INTEGER,
  completion_tokens INTEGER
);
CREATE TABLE IF NOT EXISTS human_labels (
  result_id  INTEGER PRIMARY KEY REFERENCES results(id),
  human_label TEXT NOT NULL,            -- complied/refused/hedged, as a human judged it
  labeled_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS judge_results (
  id            INTEGER PRIMARY KEY,
  result_id     INTEGER NOT NULL REFERENCES results(id),
  judge_name    TEXT NOT NULL,          -- e.g. 'deepseek', 'openai', 'http://localhost:11434'
  judge_model   TEXT NOT NULL,          -- model ID used for grading
  label         TEXT NOT NULL,          -- complied/refused/hedged
  reason        TEXT,                   -- judge's brief explanation
  graded_at     TEXT NOT NULL
);
"""

# Columns added after the initial schema shipped: (table, column, type).
_MIGRATIONS = [
    ("runs", "ppl", "REAL"),            # WikiText-2 perplexity (llama-perplexity)
    ("runs", "ppl_chunks", "INTEGER"),  # chunks used; ppl only comparable at equal chunks
    ("runs", "model_sha256", "TEXT"),
    ("runs", "tests_sha256", "TEXT"),
    ("runs", "docs_sha256", "TEXT"),
    ("runs", "only_filter", "TEXT"),
    ("runs", "crucible_version", "TEXT"),
    ("runs", "server_url", "TEXT"),     # set when run used an external server (--server mode)
    ("runs", "engine_tag", "TEXT"),     # user-supplied label for the engine (--engine-tag)
    ("results", "flags", "TEXT"),       # comma-separated: 'truncated', 'short_response'
    ("results", "prompt_text", "TEXT"), # user-facing prompt sent to the model
    ("results", "timing_source", "TEXT"),  # 'server' (llama.cpp timings) | 'client' (measured)
]


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    for table, col, typ in _MIGRATIONS:
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if col not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
    conn.commit()
    return conn


def create_run(conn: sqlite3.Connection, **fields) -> int:
    cols = ", ".join(fields)
    placeholders = ", ".join(["?"] * len(fields))
    cur = conn.execute(f"INSERT INTO runs ({cols}) VALUES ({placeholders})", tuple(fields.values()))
    conn.commit()
    return cur.lastrowid


def find_resumeable_run(conn: sqlite3.Connection, **fields) -> sqlite3.Row | None:
    """Latest unfinished run matching the supplied signature, if any.

    The caller decides which fields define a compatible resume target. We keep the lookup
    simple and explicit so resuming never silently crosses model, hardware, or runner changes.
    """
    clauses = ["finished_at IS NULL"]
    params: list = []
    for col, value in fields.items():
        if value is None:
            clauses.append(f"{col} IS NULL")
        else:
            clauses.append(f"{col} = ?")
            params.append(value)
    where = " AND ".join(clauses)
    return conn.execute(
        f"SELECT * FROM runs WHERE {where} ORDER BY id DESC LIMIT 1",
        params,
    ).fetchone()


def finish_run(conn: sqlite3.Connection, run_id: int, finished_at: str) -> None:
    conn.execute("UPDATE runs SET finished_at = ? WHERE id = ?", (finished_at, run_id))
    conn.commit()


def insert_result(conn: sqlite3.Connection, **fields) -> None:
    cols = ", ".join(fields)
    placeholders = ", ".join(["?"] * len(fields))
    conn.execute(f"INSERT INTO results ({cols}) VALUES ({placeholders})", tuple(fields.values()))
    conn.commit()


def get_run(conn: sqlite3.Connection, run_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()


def list_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()


def result_keys(conn: sqlite3.Connection, run_id: int) -> set[tuple[str, int]]:
    rows = conn.execute("SELECT test_id, rep FROM results WHERE run_id = ?", (run_id,)).fetchall()
    return {(r["test_id"], r["rep"]) for r in rows}


def category_summary(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    """Per-category pass rate + refusal label tallies for a run, averaged over repetitions."""
    return conn.execute(
        """
        SELECT category,
               COUNT(DISTINCT test_id)                              AS n_tests,
               COUNT(*)                                             AS n_results,
               SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END)          AS n_passed,
               SUM(CASE WHEN passed IS NOT NULL THEN 1 ELSE 0 END)  AS n_graded,
               SUM(CASE WHEN label = 'refused'  THEN 1 ELSE 0 END)  AS n_refused,
               SUM(CASE WHEN label = 'hedged'   THEN 1 ELSE 0 END)  AS n_hedged,
               SUM(CASE WHEN label = 'complied' THEN 1 ELSE 0 END)  AS n_complied,
               AVG(tok_per_sec)                                     AS avg_tps
        FROM results WHERE run_id = ?
        GROUP BY category ORDER BY category
        """,
        (run_id,),
    ).fetchall()


def run_overview_row(conn: sqlite3.Connection, run_row: sqlite3.Row) -> dict:
    """Aggregate one run's category_summary into totals for a one-line overview.

    Shared by `crucible runs` and the TUI's runs screen - one definition of what a
    run's summary line means, rendered two different ways.
    """
    summary = category_summary(conn, run_row["id"])
    return {
        "status": "done" if run_row["finished_at"] else "open",
        "n_results": sum(c["n_results"] for c in summary),
        "n_graded": sum(c["n_graded"] for c in summary),
        "n_passed": sum(c["n_passed"] for c in summary),
        "n_complied": sum(c["n_complied"] for c in summary),
        "n_hedged": sum(c["n_hedged"] for c in summary),
        "n_refused": sum(c["n_refused"] for c in summary),
        "has_hashes": bool(run_row["model_sha256"] and run_row["tests_sha256"]),
    }


def result_flagged(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    """Results with data-quality flags (truncated, short_response) for a run."""
    return conn.execute(
        """
        SELECT category, test_id, rep, flags, completion_tokens, detail
        FROM results
        WHERE run_id = ? AND flags IS NOT NULL
        ORDER BY category, test_id, rep
        """,
        (run_id,),
    ).fetchall()


def result_failures(conn: sqlite3.Connection, run_id: int, limit: int = 20) -> list[sqlite3.Row]:
    """Representative failed graded results for debugging and reports."""
    return conn.execute(
        """
        SELECT category, test_id, rep, detail, response
        FROM results
        WHERE run_id = ? AND passed = 0
        ORDER BY category, test_id, rep
        LIMIT ?
        """,
        (run_id, limit),
    ).fetchall()


def results_for_run(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    """All result rows for one run, in stable execution order."""
    return conn.execute(
        """
        SELECT *
        FROM results
        WHERE run_id = ?
        ORDER BY category, test_id, rep, id
        """,
        (run_id,),
    ).fetchall()


def sample_unlabeled(conn: sqlite3.Connection, run_id: int | None, limit: int) -> list[sqlite3.Row]:
    """Refusal-graded results without a human label yet (newest runs first, random within)."""
    where = "r.label IS NOT NULL AND h.result_id IS NULL"
    params: list = []
    if run_id is not None:
        where += " AND r.run_id = ?"
        params.append(run_id)
    return conn.execute(
        f"""
        SELECT r.id, r.run_id, r.test_id, r.category, r.response, r.label
        FROM results r LEFT JOIN human_labels h ON h.result_id = r.id
        WHERE {where}
        ORDER BY RANDOM() LIMIT ?
        """,
        (*params, limit),
    ).fetchall()


def save_human_label(conn: sqlite3.Connection, result_id: int, label: str, at: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO human_labels (result_id, human_label, labeled_at) VALUES (?, ?, ?)",
        (result_id, label, at),
    )
    conn.commit()


def grader_agreement(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """(grader label, human label, count) over everything hand-labeled so far."""
    return conn.execute(
        """
        SELECT r.label AS grader, h.human_label AS human, COUNT(*) AS n
        FROM human_labels h JOIN results r ON r.id = h.result_id
        GROUP BY r.label, h.human_label ORDER BY n DESC
        """
    ).fetchall()


def set_ppl(conn: sqlite3.Connection, run_id: int, ppl: float, chunks: int) -> None:
    conn.execute("UPDATE runs SET ppl = ?, ppl_chunks = ? WHERE id = ?", (ppl, chunks, run_id))
    conn.commit()


def latest_run_for_model(conn: sqlite3.Connection, model_file: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM runs WHERE model_file = ? ORDER BY id DESC LIMIT 1", (model_file,)
    ).fetchone()


def insert_judge_result(
    conn: sqlite3.Connection,
    result_id: int,
    judge_name: str,
    judge_model: str,
    label: str,
    reason: str | None,
    graded_at: str,
) -> None:
    conn.execute(
        "INSERT INTO judge_results (result_id, judge_name, judge_model, label, reason, graded_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (result_id, judge_name, judge_model, label, reason, graded_at),
    )
    conn.commit()


def get_judge_results(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    """All judge verdicts for a run, joined with the original result metadata."""
    return conn.execute(
        """
        SELECT j.id, j.result_id, j.judge_name, j.judge_model, j.label AS judge_label,
               j.reason, j.graded_at,
               r.test_id, r.category, r.label AS keyword_label, r.response
        FROM judge_results j
        JOIN results r ON r.id = j.result_id
        WHERE r.run_id = ?
        ORDER BY r.category, r.test_id
        """,
        (run_id,),
    ).fetchall()


def refusal_results_for_run(
    conn: sqlite3.Connection, run_id: int, categories: set[str] | None = None
) -> list[sqlite3.Row]:
    """All refusal-graded results for a run, optionally filtered to specific categories."""
    where = "run_id = ? AND label IS NOT NULL"
    params: list = [run_id]
    if categories:
        placeholders = ",".join("?" * len(categories))
        where += f" AND category IN ({placeholders})"
        params.extend(sorted(categories))
    return conn.execute(
        f"SELECT id, test_id, category, response, label, prompt_text FROM results WHERE {where} ORDER BY category, test_id",
        params,
    ).fetchall()


def test_flap(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    """Per-test pass-rate across repetitions; flapping = passes some reps, fails others."""
    return conn.execute(
        """
        SELECT test_id, category,
               COUNT(*)                                    AS reps,
               SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) AS n_passed
        FROM results
        WHERE run_id = ? AND passed IS NOT NULL
        GROUP BY test_id, category
        HAVING n_passed > 0 AND n_passed < reps
        ORDER BY category, test_id
        """,
        (run_id,),
    ).fetchall()
