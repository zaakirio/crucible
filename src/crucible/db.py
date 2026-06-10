"""SQLite storage — append-only. A 'run' is one model evaluated under one set of settings;
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
"""


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def create_run(conn: sqlite3.Connection, **fields) -> int:
    cols = ", ".join(fields)
    placeholders = ", ".join(["?"] * len(fields))
    cur = conn.execute(f"INSERT INTO runs ({cols}) VALUES ({placeholders})", tuple(fields.values()))
    conn.commit()
    return cur.lastrowid


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
