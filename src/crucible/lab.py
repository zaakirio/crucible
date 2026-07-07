"""Crucible Lab - interactive web workbench over the results database.

A read-mostly FastAPI app: browse runs, drill into per-prompt transcripts
(keyword + judge + human labels side by side), diff two runs, and chat against
any OpenAI-compatible server in a playground. Serves the built frontend from
lab_static/ at the root path.

FastAPI/uvicorn are optional: `pip install crucible-eval[lab]`.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from . import db
from .compare import build_comparison_rows

STATIC_DIR = Path(__file__).parent / "lab_static"


def _require_fastapi():
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "crucible lab needs the 'lab' extra: pip install crucible-eval[lab] "
            "(or: uv sync --extra lab)"
        ) from e


def _row(r) -> dict:
    return dict(r) if r is not None else {}


def create_app(db_path: str | Path):
    _require_fastapi()
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Crucible Lab", docs_url="/api/docs", openapi_url="/api/openapi.json")

    def conn():
        return db.connect(db_path)

    @app.get("/api/meta")
    def meta():
        c = conn()
        try:
            runs = db.list_runs(c)
            models = sorted({r["model_name"] for r in runs if r["model_name"]})
            n_results = c.execute("SELECT COUNT(*) AS n FROM results").fetchone()["n"]
            n_judged = c.execute(
                "SELECT COUNT(DISTINCT result_id) AS n FROM judge_results"
            ).fetchone()["n"]
            return {
                "db_path": str(db_path),
                "n_runs": len(runs),
                "n_results": n_results,
                "n_judged": n_judged,
                "models": models,
            }
        finally:
            c.close()

    @app.get("/api/runs")
    def runs():
        c = conn()
        try:
            out = []
            for r in db.list_runs(c):
                overview = db.run_overview_row(c, r)
                out.append({**_row(r), **overview})
            return out
        finally:
            c.close()

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: int):
        c = conn()
        try:
            r = db.get_run(c, run_id)
            if r is None:
                raise HTTPException(404, f"run {run_id} not found")
            return {
                **_row(r),
                **db.run_overview_row(c, r),
                "categories": [_row(x) for x in db.category_summary(c, run_id)],
                "flapping": [_row(x) for x in db.test_flap(c, run_id)],
                "flagged": [_row(x) for x in db.result_flagged(c, run_id)],
            }
        finally:
            c.close()

    @app.get("/api/runs/{run_id}/results")
    def run_results(
        run_id: int,
        category: str | None = None,
        label: str | None = None,
        status: str | None = Query(None, pattern="^(passed|failed)$"),
        q: str | None = None,
    ):
        c = conn()
        try:
            if db.get_run(c, run_id) is None:
                raise HTTPException(404, f"run {run_id} not found")
            where, params = ["r.run_id = ?"], [run_id]
            if category:
                where.append("r.category = ?")
                params.append(category)
            if label:
                where.append("r.label = ?")
                params.append(label)
            if status:
                where.append("r.passed = ?")
                params.append(1 if status == "passed" else 0)
            if q:
                # Escape LIKE wildcards so searching for "100%" matches literally.
                esc = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                like = "LIKE ? ESCAPE '\\'"
                where.append(f"(r.response {like} OR r.prompt_text {like} OR r.test_id {like})")
                params.extend([f"%{esc}%"] * 3)
            rows = c.execute(
                f"""
                SELECT r.*, h.human_label,
                       j.label AS judge_label, j.reason AS judge_reason, j.judge_model
                FROM results r
                LEFT JOIN human_labels h ON h.result_id = r.id
                LEFT JOIN judge_results j ON j.id =
                  (SELECT id FROM judge_results WHERE result_id = r.id ORDER BY graded_at DESC LIMIT 1)
                WHERE {" AND ".join(where)}
                ORDER BY r.category, r.test_id, r.rep, r.id
                """,
                params,
            ).fetchall()
            return [_row(r) for r in rows]
        finally:
            c.close()

    @app.get("/api/compare")
    def compare(a: int, b: int):
        c = conn()
        try:
            run_a, run_b = db.get_run(c, a), db.get_run(c, b)
            if run_a is None or run_b is None:
                raise HTTPException(404, "one of the runs was not found")
            sa = {r["category"]: r for r in db.category_summary(c, a)}
            sb = {r["category"]: r for r in db.category_summary(c, b)}
            rows = build_comparison_rows(sa, sb)
            return {
                "a": {**_row(run_a), **db.run_overview_row(c, run_a)},
                "b": {**_row(run_b), **db.run_overview_row(c, run_b)},
                "rows": [
                    {
                        "category": r.category,
                        "is_label": r.is_label,
                        "value_a": r.value_a,
                        "value_b": r.value_b,
                        "delta": r.delta,
                        "flagged": r.flagged,
                    }
                    for r in rows
                ],
            }
        finally:
            c.close()

    @app.post("/api/playground/chat")
    async def playground_chat(payload: dict):
        """Streaming proxy to any OpenAI-compatible /chat/completions endpoint.

        The browser can't call llama-server cross-origin; the lab process can.
        """
        server = (payload.get("server") or "").rstrip("/")
        if not server.startswith(("http://", "https://")):
            raise HTTPException(400, "payload.server must be an http(s) URL")
        body = {
            "model": payload.get("model") or "default",
            "messages": payload.get("messages") or [],
            "temperature": payload.get("temperature", 0.7),
            "max_tokens": payload.get("max_tokens", 1024),
            "stream": True,
        }

        async def stream():
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5)) as client:
                    async with client.stream(
                        "POST", f"{server}/chat/completions", json=body
                    ) as resp:
                        if resp.status_code != 200:
                            detail = (await resp.aread()).decode(errors="replace")[:500]
                            yield f"data: {json.dumps({'error': detail})}\n\n"
                            return
                        async for line in resp.aiter_lines():
                            if line:
                                yield line + "\n"
                            else:
                                yield "\n"
            except httpx.HTTPError as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    if STATIC_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        static_root = STATIC_DIR.resolve()

        @app.get("/{path:path}")
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404, f"no such API route: /{path}")
            file = (static_root / path).resolve()
            if path and file.is_relative_to(static_root) and file.is_file():
                return FileResponse(file)
            return FileResponse(static_root / "index.html")
    else:

        @app.get("/")
        def no_ui():
            return {
                "error": "frontend not built",
                "hint": "cd lab-ui && npm install && npm run build",
                "api": "/api/docs",
            }

    return app


def serve(db_path: str | Path, host: str, port: int) -> int:
    _require_fastapi()
    import uvicorn

    app = create_app(db_path)
    print(f"Crucible Lab → http://{host}:{port}  (db: {db_path})")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0
