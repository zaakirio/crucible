from __future__ import annotations

import json
import os
import tempfile
import textwrap
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from crucible import db
from crucible.charts import merged_stats
from crucible.client import ChatResult, ToolCall
from crucible.graders import grade_refusal, grade_tool_call
from crucible.report import build_run_report, render_json, render_markdown
from crucible.runner import parse_model_meta, run_suite
from crucible.retrieval import retrieve_context


class CoreTests(unittest.TestCase):
    def test_parse_model_meta_strips_quant_and_detects_lineage(self) -> None:
        name, quant, lineage = parse_model_meta(Path("Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"))
        self.assertEqual(name, "Meta-Llama-3-8B-Instruct")
        self.assertEqual(quant, "Q4_K_M")
        self.assertEqual(lineage, "base")

        name, quant, lineage = parse_model_meta(Path("heretic-model.Q5_K_M.gguf"))
        self.assertEqual(name, "heretic-model")
        self.assertEqual(quant, "Q5_K_M")
        self.assertEqual(lineage, "abliterated")

    def test_refusal_grader_distinguishes_refuse_hedge_and_comply(self) -> None:
        self.assertEqual(grade_refusal({}, "I'm sorry, but I can't help with that.").label, "refused")
        self.assertEqual(
            grade_refusal({}, "I can't help with that, but I can suggest a safe alternative.").label,
            "hedged",
        )
        self.assertEqual(grade_refusal({}, "Here is a concise summary of safe options.").label, "complied")

    def test_tool_call_grader_handles_parallel_and_irrelevance(self) -> None:
        tool_calls = [
            ToolCall(name="calc_distance", arguments={"start_loc": "New York", "end_loc": "Washington DC", "shortest_route": True}, raw_arguments='{"start_loc":"New York","end_loc":"Washington DC","shortest_route":true}'),
            ToolCall(name="calc_distance", arguments={"start_loc": "Los Angeles", "end_loc": "San Francisco", "shortest_route": True}, raw_arguments='{"start_loc":"Los Angeles","end_loc":"San Francisco","shortest_route":true}'),
        ]
        test = {
            "grader": "tool_call",
            "expect_call": True,
            "expected_calls": [
                {"calc_distance": {"start_loc": ["New York"], "end_loc": ["Washington DC"], "shortest_route": [True]}},
                {"calc_distance": {"start_loc": ["Los Angeles"], "end_loc": ["San Francisco"], "shortest_route": [True]}},
            ],
        }
        self.assertTrue(grade_tool_call(test, "", tool_calls).passed)
        self.assertTrue(grade_tool_call({"grader": "tool_call", "expect_call": False}, "", []).passed)

    def test_run_suite_can_resume_an_unfinished_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            (tests_dir / "math.yaml").write_text(
                "- id: math-001\n"
                "  prompt: What is 6 * 7?\n"
                "  grader: numeric\n"
                "  expected: 42\n"
            )
            model = root / "model-Q4_K_M.gguf"
            model.write_text("dummy gguf")
            db_path = root / "results.db"

            @contextmanager
            def fake_server(*_args, **_kwargs):
                yield SimpleNamespace(base_url="http://127.0.0.1:1", load_time_s=0.0)

            def fake_chat(*_args, **_kwargs):
                return ChatResult(
                    text="42",
                    tokens_per_second=1.0,
                    prompt_tokens=1,
                    completion_tokens=1,
                    raw={},
                    tool_calls=[],
                )

            with (
                patch("crucible.runner.llama_server", fake_server),
                patch("crucible.runner.chat", fake_chat),
                patch("crucible.runner.db.finish_run", lambda *args, **kwargs: None),
            ):
                first = run_suite(model, tests_dir=tests_dir, db_path=db_path, hardware="test-box", resume=True)
                second = run_suite(model, tests_dir=tests_dir, db_path=db_path, hardware="test-box", resume=True)

            conn = db.connect(db_path)
            try:
                self.assertEqual(first, second)
                self.assertEqual(len(db.result_keys(conn, first)), 1)
                self.assertEqual(len(db.list_runs(conn)), 1)
                self.assertIsNone(db.get_run(conn, first)["finished_at"])
            finally:
                conn.close()

    def test_run_suite_through_mock_llama_server_subprocess(self) -> None:
        # This exercises the real subprocess path used in production:
        # runner -> server.llama_server -> HTTP health check -> client.chat -> grader -> DB.
        # To swap in an actual llama.cpp build, point CRUCIBLE_LLAMA_SERVER at a real
        # `llama-server` binary and keep the rest of the pipeline unchanged.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            (tests_dir / "math.yaml").write_text(
                "- id: math-001\n"
                "  prompt: What is 6 * 7?\n"
                "  grader: numeric\n"
                "  expected: 42\n"
            )
            (tests_dir / "toolcall_mock.yaml").write_text(
                "- id: toolcall-001\n"
                "  prompt: Use calc_distance for New York and Washington DC.\n"
                "  grader: tool_call\n"
                "  expect_call: true\n"
                "  tools:\n"
                "  - type: function\n"
                "    function:\n"
                "      name: calc_distance\n"
                "      description: Calculate the driving distance between two locations.\n"
                "      parameters:\n"
                "        type: object\n"
                "        properties:\n"
                "          start_loc:\n"
                "            type: string\n"
                "          end_loc:\n"
                "            type: string\n"
                "          shortest_route:\n"
                "            type: boolean\n"
                "        required:\n"
                "        - start_loc\n"
                "        - end_loc\n"
                "  expected_calls:\n"
                "  - calc_distance:\n"
                "      start_loc:\n"
                "      - New York\n"
                "      end_loc:\n"
                "      - Washington DC\n"
                "      shortest_route:\n"
                "      - true\n"
            )
            model = root / "model-Q4_K_M.gguf"
            model.write_text("dummy gguf")
            db_path = root / "results.db"

            server_script = root / "mock_llama_server"
            server_script.write_text(textwrap.dedent("""\
                #!/usr/bin/env python3
                from __future__ import annotations

                import json
                import sys
                from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

                class Handler(BaseHTTPRequestHandler):
                    protocol_version = "HTTP/1.1"

                    def _send_json(self, payload, status=200):
                        body = json.dumps(payload).encode()
                        self.send_response(status)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)

                    def log_message(self, *_args, **_kwargs):
                        return

                    def do_GET(self):
                        if self.path == "/health":
                            self._send_json({"status": "ok"})
                        else:
                            self._send_json({"error": "not found"}, status=404)

                    def do_POST(self):
                        if self.path != "/v1/chat/completions":
                            self._send_json({"error": "not found"}, status=404)
                            return
                        length = int(self.headers.get("Content-Length", "0"))
                        body = json.loads(self.rfile.read(length) or b"{}")
                        prompt = body["messages"][-1]["content"].lower()
                        if body.get("tools") and "distance" in prompt:
                            tool_name = body["tools"][0]["function"]["name"]
                            tool_call = {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps({
                                        "start_loc": "New York",
                                        "end_loc": "Washington DC",
                                        "shortest_route": True,
                                    }),
                                },
                            }
                            message = {"role": "assistant", "content": "", "tool_calls": [tool_call]}
                        else:
                            message = {"role": "assistant", "content": "42"}
                        self._send_json({
                            "choices": [{"message": message}],
                            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                            "timings": {"predicted_per_second": 33.3},
                        })

                def main():
                    host = "127.0.0.1"
                    port = 0
                    argv = sys.argv[1:]
                    for i, arg in enumerate(argv):
                        if arg == "--host":
                            host = argv[i + 1]
                        elif arg == "--port":
                            port = int(argv[i + 1])
                    server = ThreadingHTTPServer((host, port), Handler)
                    try:
                        server.serve_forever()
                    finally:
                        server.server_close()

                if __name__ == "__main__":
                    main()
                """))
            server_script.chmod(0o755)

            with patch.dict(os.environ, {"CRUCIBLE_LLAMA_SERVER": str(server_script)}):
                run_id = run_suite(model, tests_dir=tests_dir, db_path=db_path, hardware="mock-box")

            conn = db.connect(db_path)
            try:
                summary = {row["category"]: row for row in db.category_summary(conn, run_id)}
                self.assertEqual(summary["math"]["n_passed"], 1)
                self.assertEqual(summary["toolcall_mock"]["n_passed"], 1)
                self.assertIsNotNone(db.get_run(conn, run_id)["finished_at"])
            finally:
                conn.close()

    def test_aggregate_charts_ignore_unfinished_runs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "results.db"
            conn = db.connect(db_path)
            try:
                finished = db.create_run(
                    conn,
                    model_file="model-finished.gguf",
                    model_name="demo",
                    quant="Q4_K_M",
                    lineage="base",
                    hardware="test-box",
                    llama_cpp_commit="abc123",
                    ctx=4096,
                    ngl=99,
                    repeat=1,
                    started_at="2026-01-01T00:00:00+00:00",
                    load_time_s=1.0,
                    model_size_bytes=123,
                )
                db.insert_result(
                    conn,
                    run_id=finished,
                    test_id="math-001",
                    category="math",
                    rep=0,
                    response="42",
                    passed=1,
                    label=None,
                    detail="ok",
                    latency_ms=10,
                    tok_per_sec=1.0,
                    prompt_tokens=1,
                    completion_tokens=1,
                )
                db.finish_run(conn, finished, "2026-01-01T00:01:00+00:00")

                unfinished = db.create_run(
                    conn,
                    model_file="model-unfinished.gguf",
                    model_name="demo",
                    quant="Q4_K_M",
                    lineage="base",
                    hardware="test-box",
                    llama_cpp_commit="abc123",
                    ctx=4096,
                    ngl=99,
                    repeat=1,
                    started_at="2026-01-01T00:00:00+00:00",
                    load_time_s=1.0,
                    model_size_bytes=456,
                )
                db.insert_result(
                    conn,
                    run_id=unfinished,
                    test_id="math-001",
                    category="math",
                    rep=0,
                    response="0",
                    passed=0,
                    label=None,
                    detail="bad",
                    latency_ms=10,
                    tok_per_sec=1.0,
                    prompt_tokens=1,
                    completion_tokens=1,
                )

                groups = merged_stats(conn)
                self.assertEqual(groups[("demo", "Q4_K_M", "base")].categories["math"].rate, 1.0)
                self.assertEqual(groups[("demo", "Q4_K_M", "base")].model_size_bytes, 123)
            finally:
                conn.close()

    def test_run_suite_supports_full_message_lists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            (tests_dir / "rag_grounded.yaml").write_text(
                "- id: rag-001\n"
                "  messages:\n"
                "  - role: system\n"
                "    content: Answer only from the provided context.\n"
                "  - role: user\n"
                "    content: |\n"
                "      Context: The capital of France is Paris.\n"
                "      Question: What is the capital of France?\n"
                "  grader: exact\n"
                "  expected: Paris\n"
            )
            model = root / "model-Q4_K_M.gguf"
            model.write_text("dummy gguf")
            db_path = root / "results.db"

            @contextmanager
            def fake_server(*_args, **_kwargs):
                yield SimpleNamespace(base_url="http://127.0.0.1:1", load_time_s=0.0)

            def fake_chat(base_url, messages, *, tools=None, temperature=0.0, seed=0, max_tokens=512, timeout_s=300.0):
                self.assertEqual(base_url, "http://127.0.0.1:1")
                self.assertEqual([m["role"] for m in messages], ["system", "user"])
                self.assertIn("provided context", messages[0]["content"])
                self.assertIn("capital of France", messages[1]["content"])
                return ChatResult(
                    text="Paris",
                    tokens_per_second=1.0,
                    prompt_tokens=1,
                    completion_tokens=1,
                    raw={},
                    tool_calls=[],
                )

            with (
                patch("crucible.runner.llama_server", fake_server),
                patch("crucible.runner.chat", fake_chat),
                patch("crucible.runner.db.finish_run", lambda *args, **kwargs: None),
            ):
                run_id = run_suite(model, tests_dir=tests_dir, db_path=db_path, hardware="test-box")

            conn = db.connect(db_path)
            try:
                summary = {row["category"]: row for row in db.category_summary(conn, run_id)}
                self.assertEqual(summary["rag_grounded"]["n_passed"], 1)
            finally:
                conn.close()

    def test_retrieval_returns_relevant_context(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            docs = Path(td)
            (docs / "france.md").write_text("The capital of France is Paris.\n")
            (docs / "history.md").write_text("The Battle of Hastings took place in 1066.\n")

            context = retrieve_context("What is the capital of France?", docs, top_k=1)
            self.assertIn("france.md#0", context)
            self.assertIn("Paris", context)

    def test_run_suite_supports_retrieval_backed_grounded_qa(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tests_dir = root / "tests"
            docs_dir = root / "docs"
            tests_dir.mkdir()
            docs_dir.mkdir()
            (docs_dir / "france.md").write_text("The capital of France is Paris.\n")
            (tests_dir / "rag_grounded.yaml").write_text(
                "- id: rag-001\n"
                "  prompt: What is the capital of France?\n"
                "  grader: exact\n"
                "  expected: Paris\n"
                "  retrieval: true\n"
            )
            model = root / "model-Q4_K_M.gguf"
            model.write_text("dummy gguf")
            db_path = root / "results.db"

            @contextmanager
            def fake_server(*_args, **_kwargs):
                yield SimpleNamespace(base_url="http://127.0.0.1:1", load_time_s=0.0)

            def fake_chat(base_url, messages, *, tools=None, temperature=0.0, seed=0, max_tokens=512, timeout_s=300.0):
                self.assertEqual(base_url, "http://127.0.0.1:1")
                self.assertEqual(messages[0]["role"], "system")
                self.assertIn("retrieved context", messages[1]["content"].lower())
                self.assertIn("france.md#0", messages[1]["content"])
                self.assertIn("capital of France is Paris", messages[1]["content"])
                return ChatResult(
                    text="Paris",
                    tokens_per_second=1.0,
                    prompt_tokens=1,
                    completion_tokens=1,
                    raw={},
                    tool_calls=[],
                )

            with (
                patch("crucible.runner.llama_server", fake_server),
                patch("crucible.runner.chat", fake_chat),
                patch("crucible.runner.db.finish_run", lambda *args, **kwargs: None),
            ):
                run_id = run_suite(model, tests_dir=tests_dir, db_path=db_path, hardware="test-box", docs_dir=docs_dir)

            conn = db.connect(db_path)
            try:
                summary = {row["category"]: row for row in db.category_summary(conn, run_id)}
                self.assertEqual(summary["rag_grounded"]["n_passed"], 1)
            finally:
                conn.close()

    def test_run_suite_supports_agent_style_conversations(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            (tests_dir / "agent_dialogue.yaml").write_text(
                "- id: agent-001\n"
                "  grader: exact\n"
                "  expected: Zaakir\n"
                "  conversation:\n"
                "  - role: system\n"
                "    content: You are a concise assistant.\n"
                "  - role: user\n"
                "    content: My name is Zaakir. Remember it.\n"
                "  - role: assistant\n"
                "    content: Understood.\n"
                "  - role: user\n"
                "    content: What is my name?\n"
            )
            model = root / "model-Q4_K_M.gguf"
            model.write_text("dummy gguf")
            db_path = root / "results.db"

            @contextmanager
            def fake_server(*_args, **_kwargs):
                yield SimpleNamespace(base_url="http://127.0.0.1:1", load_time_s=0.0)

            def fake_chat(base_url, messages, *, tools=None, temperature=0.0, seed=0, max_tokens=512, timeout_s=300.0):
                self.assertEqual(base_url, "http://127.0.0.1:1")
                self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
                self.assertEqual(messages[1]["content"], "My name is Zaakir. Remember it.")
                self.assertEqual(messages[-1]["content"], "What is my name?")
                return ChatResult(
                    text="Zaakir",
                    tokens_per_second=1.0,
                    prompt_tokens=1,
                    completion_tokens=1,
                    raw={},
                    tool_calls=[],
                )

            with (
                patch("crucible.runner.llama_server", fake_server),
                patch("crucible.runner.chat", fake_chat),
                patch("crucible.runner.db.finish_run", lambda *args, **kwargs: None),
            ):
                run_id = run_suite(model, tests_dir=tests_dir, db_path=db_path, hardware="test-box")

            conn = db.connect(db_path)
            try:
                summary = {row["category"]: row for row in db.category_summary(conn, run_id)}
                self.assertEqual(summary["agent_dialogue"]["n_passed"], 1)
            finally:
                conn.close()

    def test_run_suite_records_provenance_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tests_dir = root / "tests"
            docs_dir = root / "docs"
            tests_dir.mkdir()
            docs_dir.mkdir()
            (tests_dir / "rag_grounded.yaml").write_text(
                "- id: rag-001\n"
                "  prompt: What is the capital of France?\n"
                "  grader: exact\n"
                "  expected: Paris\n"
                "  retrieval: true\n"
            )
            (docs_dir / "france.md").write_text("The capital of France is Paris.\n")
            model = root / "model-Q4_K_M.gguf"
            model.write_text("dummy gguf")
            db_path = root / "results.db"

            @contextmanager
            def fake_server(*_args, **_kwargs):
                yield SimpleNamespace(base_url="http://127.0.0.1:1", load_time_s=0.0)

            def fake_chat(*_args, **_kwargs):
                return ChatResult(
                    text="Paris",
                    tokens_per_second=1.0,
                    prompt_tokens=1,
                    completion_tokens=1,
                    raw={},
                    tool_calls=[],
                )

            with (
                patch("crucible.runner.llama_server", fake_server),
                patch("crucible.runner.chat", fake_chat),
            ):
                run_id = run_suite(
                    model,
                    tests_dir=tests_dir,
                    db_path=db_path,
                    hardware="test-box",
                    only={"rag_grounded"},
                    docs_dir=docs_dir,
                )

            conn = db.connect(db_path)
            try:
                run = db.get_run(conn, run_id)
                self.assertEqual(len(run["model_sha256"]), 64)
                self.assertEqual(len(run["tests_sha256"]), 64)
                self.assertEqual(len(run["docs_sha256"]), 64)
                self.assertEqual(run["only_filter"], "rag_grounded")
                self.assertEqual(run["crucible_version"], "0.0.1")
            finally:
                conn.close()

    def test_report_renders_markdown_and_json_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "results.db"
            conn = db.connect(db_path)
            try:
                run_id = db.create_run(
                    conn,
                    model_file="model-Q4_K_M.gguf",
                    model_name="model",
                    quant="Q4_K_M",
                    lineage="base",
                    hardware="test-box",
                    llama_cpp_commit="abc123",
                    ctx=4096,
                    ngl=99,
                    repeat=1,
                    started_at="2026-01-01T00:00:00+00:00",
                    finished_at="2026-01-01T00:01:00+00:00",
                    load_time_s=1.0,
                    model_size_bytes=123,
                    model_sha256="a" * 64,
                    tests_sha256="b" * 64,
                    docs_sha256=None,
                    only_filter="math",
                    crucible_version="0.0.1",
                )
                db.insert_result(
                    conn,
                    run_id=run_id,
                    test_id="math-001",
                    category="math",
                    rep=0,
                    response="42",
                    passed=1,
                    label=None,
                    detail="ok",
                    latency_ms=10,
                    tok_per_sec=12.5,
                    prompt_tokens=1,
                    completion_tokens=1,
                )
                db.insert_result(
                    conn,
                    run_id=run_id,
                    test_id="math-002",
                    category="math",
                    rep=0,
                    response="0",
                    passed=0,
                    label=None,
                    detail="expected 1, got 0",
                    latency_ms=10,
                    tok_per_sec=10.0,
                    prompt_tokens=1,
                    completion_tokens=1,
                )

                report = build_run_report(conn, run_id)
                markdown = render_markdown(report)
                payload = json.loads(render_json(report))
                self.assertIn("# Crucible Run Report #1", markdown)
                self.assertIn("model sha256: `aaaaaaaaaaaa`", markdown)
                self.assertIn("| `math` | 1/2 (50%) | 11.2 |", markdown)
                self.assertIn("`math/math-002`", markdown)
                self.assertEqual(payload["summary"]["total_passed"], 1)
                self.assertEqual(payload["failures"][0]["test_id"], "math-002")
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
