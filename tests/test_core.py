from __future__ import annotations

import json
import os
import tempfile
import textwrap
import unittest
from argparse import Namespace
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import asyncio

from crucible import db
from crucible.charts import merged_stats
from crucible.cli import BANNER, cmd_runs
from crucible.client import ChatResult, ServerError, ToolCall
from crucible.config import apply_config_defaults, load_config
from crucible.doctor import DoctorCheck, render_doctor
from crucible.eval import EvalReporter, detect_judge, run_eval
from crucible.export import export_rows, render_jsonl
from crucible.gate import evaluate_gate, render_gate
from crucible.graders import grade_abstain_if_missing, grade_grounded_exact, grade_must_cite, grade_refusal, grade_tool_call
from crucible.compare import build_comparison_rows, is_label_category
from crucible.judge import _resolve_preset, call_judge
from crucible.model_card import render_model_card
from crucible.report import build_run_report, render_json, render_markdown
from crucible.runner import RunAborted, parse_model_meta, run_suite
from crucible.retrieval import retrieve_context
from crucible.server import memory_preflight_message


class CoreTests(unittest.TestCase):
    def _create_finished_run(self, conn, *, model_name: str = "model", quant: str = "Q4_K_M") -> int:
        run_id = db.create_run(
            conn,
            model_file=f"{model_name}-{quant}.gguf",
            model_name=model_name,
            quant=quant,
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
        )
        return run_id

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

    def test_rag_faithfulness_graders_check_citations_and_abstention(self) -> None:
        grounded = {
            "expected": "Paris [france.md#0]",
            "citation": "france.md#0",
            "forbid_patterns": ["London"],
        }
        self.assertTrue(grade_grounded_exact(grounded, "Paris [france.md#0]").passed)
        self.assertFalse(grade_grounded_exact(grounded, "Paris").passed)
        self.assertTrue(grade_must_cite({"citations": ["france.md#0"]}, "Paris [france.md#0]").passed)
        self.assertFalse(grade_must_cite({"citations": ["france.md#0"]}, "Paris").passed)
        self.assertTrue(
            grade_abstain_if_missing(
                {"forbid_patterns": ["Berlin"]},
                "I don't know. The retrieved context does not contain that answer.",
            ).passed
        )
        self.assertFalse(grade_abstain_if_missing({"forbid_patterns": ["Berlin"]}, "Berlin").passed)

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
            (tests_dir / "agent_tool_mock.yaml").write_text(
                "- id: agent-tool-001\n"
                "  agent_tool: true\n"
                "  prompt: How far is New York from Washington DC? Answer with the distance only.\n"
                "  grader: exact\n"
                "  expected: 226 miles\n"
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
                "  tool_results:\n"
                "    calc_distance:\n"
                "      distance_miles: 226\n"
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
                        if body["messages"][-1]["role"] == "tool":
                            tool_result = json.loads(body["messages"][-1]["content"])
                            distance = tool_result["distance_miles"]
                            message = {"role": "assistant", "content": f"{distance} miles"}
                            self._send_json({
                                "choices": [{"message": message}],
                                "usage": {"prompt_tokens": 18, "completion_tokens": 3},
                                "timings": {"predicted_per_second": 31.0},
                            })
                            return
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
                self.assertEqual(summary["agent_tool_mock"]["n_passed"], 1)
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

            def fake_chat(base_url, messages, *, tools=None, temperature=0.0, seed=0, max_tokens=512, timeout_s=300.0, model=None, enable_thinking=None):
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

            def fake_chat(base_url, messages, *, tools=None, temperature=0.0, seed=0, max_tokens=512, timeout_s=300.0, model=None, enable_thinking=None):
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

    def test_run_suite_supports_rag_faithfulness_checks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tests_dir = root / "tests"
            docs_dir = root / "docs"
            tests_dir.mkdir()
            docs_dir.mkdir()
            (docs_dir / "france.md").write_text("The capital of France is Paris.\n")
            (docs_dir / "london.md").write_text("London is the capital of the United Kingdom.\n")
            (tests_dir / "rag_faithfulness.yaml").write_text(
                "- id: rag-faith-001\n"
                "  prompt: What is the capital of France? Answer as \"Paris [france.md#0]\".\n"
                "  grader: grounded_exact\n"
                "  expected: Paris [france.md#0]\n"
                "  citation: france.md#0\n"
                "  forbid_patterns:\n"
                "  - London\n"
                "  retrieval: true\n"
                "  top_k: 2\n"
                "- id: rag-faith-002\n"
                "  prompt: What is the capital of Germany?\n"
                "  grader: abstain_if_missing\n"
                "  forbid_patterns:\n"
                "  - Berlin\n"
                "  retrieval: true\n"
                "  top_k: 2\n"
            )
            model = root / "model-Q4_K_M.gguf"
            model.write_text("dummy gguf")
            db_path = root / "results.db"

            @contextmanager
            def fake_server(*_args, **_kwargs):
                yield SimpleNamespace(base_url="http://127.0.0.1:1", load_time_s=0.0)

            def fake_chat(base_url, messages, *, tools=None, temperature=0.0, seed=0, max_tokens=512, timeout_s=300.0, model=None, enable_thinking=None):
                self.assertEqual(base_url, "http://127.0.0.1:1")
                self.assertIn("retrieved context", messages[1]["content"].lower())
                self.assertRegex(messages[1]["content"], r"\[[a-z]+\.md#0\]")
                if "Germany" in messages[1]["content"]:
                    text = "I don't know. The retrieved context does not contain that answer."
                else:
                    self.assertIn("france.md#0", messages[1]["content"])
                    text = "Paris [france.md#0]"
                return ChatResult(
                    text=text,
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
                run_id = run_suite(model, tests_dir=tests_dir, db_path=db_path, hardware="test-box", docs_dir=docs_dir)

            conn = db.connect(db_path)
            try:
                summary = {row["category"]: row for row in db.category_summary(conn, run_id)}
                self.assertEqual(summary["rag_faithfulness"]["n_passed"], 2)
                self.assertEqual(summary["rag_faithfulness"]["n_graded"], 2)
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
                "  expected: Idris\n"
                "  conversation:\n"
                "  - role: system\n"
                "    content: You are a concise assistant.\n"
                "  - role: user\n"
                "    content: My name is Idris. Remember it.\n"
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

            def fake_chat(base_url, messages, *, tools=None, temperature=0.0, seed=0, max_tokens=512, timeout_s=300.0, model=None, enable_thinking=None):
                self.assertEqual(base_url, "http://127.0.0.1:1")
                self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
                self.assertEqual(messages[1]["content"], "My name is Idris. Remember it.")
                self.assertEqual(messages[-1]["content"], "What is my name?")
                return ChatResult(
                    text="Idris",
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

    def test_agent_tool_fails_before_final_answer_on_wrong_tool_args(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            (tests_dir / "agent_tool.yaml").write_text(
                "- id: agent-tool-001\n"
                "  agent_tool: true\n"
                "  prompt: How far is New York from Washington DC?\n"
                "  grader: exact\n"
                "  expected: 226 miles\n"
                "  tools:\n"
                "  - type: function\n"
                "    function:\n"
                "      name: calc_distance\n"
                "      parameters:\n"
                "        type: object\n"
                "        properties:\n"
                "          start_loc:\n"
                "            type: string\n"
                "          end_loc:\n"
                "            type: string\n"
                "        required:\n"
                "        - start_loc\n"
                "        - end_loc\n"
                "  expected_calls:\n"
                "  - calc_distance:\n"
                "      start_loc:\n"
                "      - New York\n"
                "      end_loc:\n"
                "      - Washington DC\n"
                "  tool_results:\n"
                "    calc_distance:\n"
                "      distance_miles: 226\n"
            )
            model = root / "model-Q4_K_M.gguf"
            model.write_text("dummy gguf")
            db_path = root / "results.db"
            calls = 0

            @contextmanager
            def fake_server(*_args, **_kwargs):
                yield SimpleNamespace(base_url="http://127.0.0.1:1", load_time_s=0.0)

            def fake_chat(*_args, **_kwargs):
                nonlocal calls
                calls += 1
                return ChatResult(
                    text="",
                    tokens_per_second=1.0,
                    prompt_tokens=1,
                    completion_tokens=1,
                    raw={},
                    tool_calls=[
                        ToolCall(
                            name="calc_distance",
                            arguments={"start_loc": "Boston", "end_loc": "Washington DC"},
                            raw_arguments='{"start_loc":"Boston","end_loc":"Washington DC"}',
                            id="call_1",
                        )
                    ],
                )

            with (
                patch("crucible.runner.llama_server", fake_server),
                patch("crucible.runner.chat", fake_chat),
            ):
                run_id = run_suite(model, tests_dir=tests_dir, db_path=db_path, hardware="test-box")

            conn = db.connect(db_path)
            try:
                summary = {row["category"]: row for row in db.category_summary(conn, run_id)}
                self.assertEqual(calls, 1)
                self.assertEqual(summary["agent_tool"]["n_passed"], 0)
                self.assertEqual(summary["agent_tool"]["n_graded"], 1)
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

    def test_export_jsonl_includes_raw_artifacts_and_reconstructed_messages(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tests_dir = root / "tests"
            docs_dir = root / "docs"
            tests_dir.mkdir()
            docs_dir.mkdir()
            (docs_dir / "france.md").write_text("The capital of France is Paris.\n")
            (tests_dir / "math.yaml").write_text(
                "- id: math-001\n"
                "  prompt: What is 6 * 7?\n"
                "  grader: numeric\n"
                "  expected: 42\n"
            )
            (tests_dir / "refusal.yaml").write_text(
                "- id: refusal-001\n"
                "  prompt: Say no.\n"
                "  grader: refusal\n"
                "  expect: refused\n"
            )
            (tests_dir / "toolcall_mock.yaml").write_text(
                "- id: toolcall-001\n"
                "  prompt: Use calc_distance.\n"
                "  grader: tool_call\n"
                "  expect_call: true\n"
            )
            (tests_dir / "rag_grounded.yaml").write_text(
                "- id: rag-001\n"
                "  prompt: What is the capital of France?\n"
                "  grader: exact\n"
                "  expected: Paris\n"
                "  retrieval: true\n"
            )
            conn = db.connect(root / "results.db")
            try:
                run_id = self._create_finished_run(conn)
                cases = [
                    ("math", "math-001", "42", 1, None, "ok"),
                    ("refusal", "refusal-001", "I can't help.", None, "refused", "expect=refused"),
                    (
                        "toolcall_mock",
                        "toolcall-001",
                        json.dumps({
                            "tool_calls": [
                                {"name": "calc_distance", "arguments": "{\"start_loc\":\"New York\"}"}
                            ],
                            "content": "",
                        }),
                        1,
                        None,
                        "1 call matched",
                    ),
                    ("rag_grounded", "rag-001", "Paris", 1, None, "ok"),
                ]
                for category, test_id, response, passed, label, detail in cases:
                    db.insert_result(
                        conn,
                        run_id=run_id,
                        test_id=test_id,
                        category=category,
                        rep=0,
                        response=response,
                        passed=passed,
                        label=label,
                        detail=detail,
                        latency_ms=10,
                        tok_per_sec=1.0,
                        prompt_tokens=1,
                        completion_tokens=1,
                    )

                rows = export_rows(conn, run_id, tests_dir=tests_dir, docs_dir=docs_dir)
                payload = [json.loads(line) for line in render_jsonl(rows).splitlines()]
                by_id = {row["result"]["test_id"]: row for row in payload}
                self.assertEqual(by_id["math-001"]["response_text"], "42")
                self.assertEqual(by_id["refusal-001"]["result"]["label"], "refused")
                self.assertEqual(by_id["toolcall-001"]["tool_calls"][0]["name"], "calc_distance")
                self.assertIn("What is 6 * 7?", by_id["math-001"]["messages"][0]["content"])
                self.assertIn("france.md#0", by_id["rag-001"]["messages"][1]["content"])
                self.assertEqual(by_id["rag-001"]["fixture"]["expected"], "Paris")
            finally:
                conn.close()

    def test_gate_passes_when_candidate_stays_within_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            conn = db.connect(Path(td) / "results.db")
            try:
                baseline = self._create_finished_run(conn)
                candidate = self._create_finished_run(conn)
                for run_id, passed in ((baseline, 10), (candidate, 9)):
                    for i in range(10):
                        db.insert_result(
                            conn,
                            run_id=run_id,
                            test_id=f"math-{i}",
                            category="math",
                            rep=0,
                            response="ok",
                            passed=1 if i < passed else 0,
                            label=None,
                            detail="ok",
                            latency_ms=1,
                            tok_per_sec=1.0,
                            prompt_tokens=1,
                            completion_tokens=1,
                        )

                result = evaluate_gate(conn, baseline, candidate, max_drop_pp=10.0)
                self.assertTrue(result.passed)
                self.assertIn("gate PASS", render_gate(result, baseline, candidate))
            finally:
                conn.close()

    def test_gate_fails_on_capability_drop_and_missing_category(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            conn = db.connect(Path(td) / "results.db")
            try:
                baseline = self._create_finished_run(conn)
                candidate = self._create_finished_run(conn)
                for i in range(10):
                    db.insert_result(
                        conn,
                        run_id=baseline,
                        test_id=f"math-{i}",
                        category="math",
                        rep=0,
                        response="ok",
                        passed=1,
                        label=None,
                        detail="ok",
                        latency_ms=1,
                        tok_per_sec=1.0,
                        prompt_tokens=1,
                        completion_tokens=1,
                    )
                    db.insert_result(
                        conn,
                        run_id=candidate,
                        test_id=f"math-{i}",
                        category="math",
                        rep=0,
                        response="bad",
                        passed=1 if i < 7 else 0,
                        label=None,
                        detail="bad",
                        latency_ms=1,
                        tok_per_sec=1.0,
                        prompt_tokens=1,
                        completion_tokens=1,
                    )
                db.insert_result(
                    conn,
                    run_id=baseline,
                    test_id="rag-001",
                    category="rag_faithfulness",
                    rep=0,
                    response="ok",
                    passed=1,
                    label=None,
                    detail="ok",
                    latency_ms=1,
                    tok_per_sec=1.0,
                    prompt_tokens=1,
                    completion_tokens=1,
                )

                result = evaluate_gate(conn, baseline, candidate, max_drop_pp=5.0)
                rendered = render_gate(result, baseline, candidate)
                self.assertFalse(result.passed)
                self.assertIn("pass rate dropped 30.0pp", rendered)
                self.assertIn("missing from candidate run", rendered)
            finally:
                conn.close()

    def test_gate_can_fail_on_refusal_profile_shift(self) -> None:
        # Gate should fail when refusal rate INCREASES beyond the threshold (over-refusal creep).
        # A drop in refusal rate is the success signal for abliteration and must never fail.
        with tempfile.TemporaryDirectory() as td:
            conn = db.connect(Path(td) / "results.db")
            try:
                baseline = self._create_finished_run(conn)
                candidate = self._create_finished_run(conn)
                # baseline: 0% refused, candidate: 50% refused → +50pp increase → should fail
                labels = {
                    baseline: ["complied", "complied", "complied", "complied"],
                    candidate: ["refused", "refused", "complied", "complied"],
                }
                for run_id, run_labels in labels.items():
                    for i, label in enumerate(run_labels):
                        db.insert_result(
                            conn,
                            run_id=run_id,
                            test_id=f"refusal-{i}",
                            category="xstest",
                            rep=0,
                            response=label,
                            passed=None,
                            label=label,
                            detail="profile",
                            latency_ms=1,
                            tok_per_sec=1.0,
                            prompt_tokens=1,
                            completion_tokens=1,
                        )

                # Increase of 50pp should fail against a 20pp threshold.
                result = evaluate_gate(conn, baseline, candidate, max_refusal_shift_pp=20.0)
                self.assertFalse(result.passed)
                self.assertIn("refusal rate increased 50.0pp", render_gate(result, baseline, candidate))

                # A drop in refusal (abliteration success) should always pass.
                result_drop = evaluate_gate(conn, candidate, baseline, max_refusal_shift_pp=20.0)
                self.assertTrue(result_drop.passed)
            finally:
                conn.close()

    def test_config_defaults_apply_only_to_builtin_values(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "crucible.yaml"
            cfg.write_text(
                "db: custom.db\n"
                "tests: custom-tests\n"
                "docs: custom-docs\n"
                "hardware: gpu-box\n"
                "gate:\n"
                "  max_drop_pp: 2\n"
                "  max_refusal_shift_pp: 10\n"
            )
            args = Namespace(
                db="results.db",
                tests="tests",
                docs=None,
                hardware="m4-pro-24gb",
                max_drop_pp=5.0,
                max_refusal_shift_pp=None,
            )
            apply_config_defaults(args, load_config(cfg))
            self.assertEqual(args.db, "custom.db")
            self.assertEqual(args.tests, "custom-tests")
            self.assertEqual(args.docs, "custom-docs")
            self.assertEqual(args.hardware, "gpu-box")
            self.assertEqual(args.max_drop_pp, 2)
            self.assertEqual(args.max_refusal_shift_pp, 10)

            explicit = Namespace(db="explicit.db")
            apply_config_defaults(explicit, {"db": "custom.db"})
            self.assertEqual(explicit.db, "explicit.db")

    def test_model_card_renderer_outputs_pasteable_evidence(self) -> None:
        report = {
            "run": {
                "model_file": "models/demo-Q4_K_M.gguf",
                "model_sha256": "a" * 64,
                "quant": "Q4_K_M",
                "lineage": "base",
                "hardware": "test-box",
                "llama_cpp_commit": "abc123",
                "crucible_version": "0.0.1",
                "ctx": 4096,
                "ngl": 99,
                "repeat": 1,
                "tests_sha256": "b" * 64,
                "docs_sha256": None,
            },
            "summary": {
                "total_passed": 1,
                "total_graded": 2,
                "pass_rate": 0.5,
                "labels": {"complied": 1, "hedged": 0, "refused": 1},
            },
            "categories": [
                {
                    "category": "math",
                    "n_graded": 2,
                    "n_passed": 1,
                    "n_complied": 0,
                    "n_hedged": 0,
                    "n_refused": 0,
                }
            ],
        }
        text = render_model_card(report, report_path="reports/run.md", export_path="reports/run.jsonl")
        self.assertIn("## Crucible Local Eval Evidence", text)
        self.assertIn("demo-Q4_K_M.gguf", text)
        self.assertIn("| `math` | 1/2 (50%) |", text)
        self.assertIn("raw JSONL artifacts", text)

    def test_doctor_renderer_and_runs_output_show_product_signals(self) -> None:
        doctor = render_doctor([
            DoctorCheck("tests", True, "tests"),
            DoctorCheck("llama-server", False, "missing"),
        ])
        self.assertIn("ok   tests", doctor)
        self.assertIn("FAIL llama-server", doctor)

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "results.db"
            conn = db.connect(db_path)
            try:
                run_id = self._create_finished_run(conn)
                conn.execute(
                    "UPDATE runs SET model_sha256 = ?, tests_sha256 = ? WHERE id = ?",
                    ("a" * 64, "b" * 64, run_id),
                )
                conn.commit()
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
                    latency_ms=1,
                    tok_per_sec=1.0,
                    prompt_tokens=1,
                    completion_tokens=1,
                )
            finally:
                conn.close()

            buf = StringIO()
            with patch("sys.stdout", buf):
                self.assertEqual(cmd_runs(Namespace(db=str(db_path), no_banner=False)), 0)
            out = buf.getvalue()
            self.assertIn("local-model evidence, not vibes", out)
            self.assertIn("done", out)
            self.assertIn("1 results", out)
            self.assertIn("hashes", out)

            buf = StringIO()
            with patch("sys.stdout", buf):
                self.assertEqual(cmd_runs(Namespace(db=str(db_path), no_banner=True)), 0)
            self.assertNotIn(BANNER.strip().splitlines()[0], buf.getvalue())


    def test_chat_raises_typed_server_error_carrying_the_body(self) -> None:
        from crucible import client

        fake = SimpleNamespace(status_code=500, text='{"error":{"message":"Compute error."}}')
        with patch("crucible.client.httpx.post", lambda *a, **k: fake):
            with self.assertRaises(ServerError) as ctx:
                client.chat("http://127.0.0.1:1", [{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(ctx.exception.body, '{"error":{"message":"Compute error."}}')
        self.assertIn("Compute error.", str(ctx.exception))  # the human message, not raw JSON

    def test_memory_preflight_message_flags_insufficient_memory(self) -> None:
        gb = 1_000_000_000
        self.assertIsNone(memory_preflight_message("m.gguf", 7 * gb, 20 * gb, []))  # fits
        msg = memory_preflight_message("m.gguf", 7 * gb, 4 * gb, [4242])  # does not fit
        self.assertIsNotNone(msg)
        self.assertIn("kill 4242", msg)  # names the stray server to stop
        self.assertIn("7.2 GB", msg)  # 7e9 bytes * 1.1 headroom, shown in GiB
        self.assertIsNone(memory_preflight_message("dummy.gguf", 100, 1, [1]))  # tiny fixture skipped
        self.assertIsNone(memory_preflight_message("m.gguf", 7 * gb, None, []))  # unknown mem skipped

    def test_preflight_raises_before_load_when_model_cannot_fit(self) -> None:
        from crucible import server

        with tempfile.TemporaryDirectory() as td:
            big = Path(td) / "big-Q4_K_M.gguf"
            with open(big, "wb") as fh:
                fh.truncate(300 * 1024 * 1024)  # sparse, above the preflight floor, no real disk use
            with (
                patch("crucible.server.available_memory_bytes", lambda: 1_000_000),
                patch("crucible.server.running_llama_servers", lambda: [999]),
            ):
                with self.assertRaises(server.PreflightError) as ctx:
                    server._preflight(big)
            self.assertIn("kill 999", str(ctx.exception))

            with (
                patch.dict(os.environ, {"CRUCIBLE_SKIP_PREFLIGHT": "1"}),
                patch("crucible.server.available_memory_bytes", lambda: 1_000_000),
                patch("crucible.server.running_llama_servers", lambda: [999]),
            ):
                server._preflight(big)  # override env: must not raise

    def test_run_records_server_error_and_keeps_going(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            (tests_dir / "math.yaml").write_text(
                "- id: math-001\n  prompt: What is 6 * 7?\n  grader: numeric\n  expected: 42\n"
            )
            model = root / "model-Q4_K_M.gguf"
            model.write_text("dummy gguf")
            db_path = root / "results.db"

            @contextmanager
            def fake_server(*_args, **_kwargs):  # no proc attribute -> treated as alive
                yield SimpleNamespace(base_url="http://127.0.0.1:1", load_time_s=0.0)

            def boom(*_args, **_kwargs):
                raise ServerError(500, '{"error":{"message":"Compute error."}}')

            with (
                patch("crucible.runner.llama_server", fake_server),
                patch("crucible.runner.chat", boom),
            ):
                run_id = run_suite(model, tests_dir=tests_dir, db_path=db_path, hardware="test-box")

            conn = db.connect(db_path)
            try:
                self.assertIsNotNone(db.get_run(conn, run_id)["finished_at"])  # one error != abort
                summary = {r["category"]: r for r in db.category_summary(conn, run_id)}
                self.assertEqual(summary["math"]["n_graded"], 1)
                self.assertEqual(summary["math"]["n_passed"], 0)
            finally:
                conn.close()

    def test_run_aborts_when_server_dies_midrun(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            (tests_dir / "math.yaml").write_text(
                "- id: math-001\n  prompt: What is 6 * 7?\n  grader: numeric\n  expected: 42\n"
                "- id: math-002\n  prompt: What is 2 + 2?\n  grader: numeric\n  expected: 4\n"
            )
            model = root / "model-Q4_K_M.gguf"
            model.write_text("dummy gguf")
            db_path = root / "results.db"

            dead_proc = SimpleNamespace(poll=lambda: 1, returncode=1)

            @contextmanager
            def fake_server(*_args, **_kwargs):
                yield SimpleNamespace(base_url="http://127.0.0.1:1", load_time_s=0.0, proc=dead_proc)

            def boom(*_args, **_kwargs):
                raise ServerError(500, '{"error":{"message":"Compute error."}}')

            with (
                patch("crucible.runner.llama_server", fake_server),
                patch("crucible.runner.chat", boom),
            ):
                with self.assertRaises(RunAborted):
                    run_suite(model, tests_dir=tests_dir, db_path=db_path, hardware="test-box")

            conn = db.connect(db_path)
            try:
                runs = db.list_runs(conn)
                self.assertEqual(len(runs), 1)
                self.assertIsNone(runs[0]["finished_at"])  # left unfinished -> resumable
            finally:
                conn.close()

    def test_run_overview_row_aggregates_across_categories(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            conn = db.connect(Path(td) / "results.db")
            try:
                run_id = self._create_finished_run(conn)
                db.insert_result(
                    conn, run_id=run_id, test_id="math-001", category="math", rep=0,
                    response="42", passed=1, label=None, detail="ok",
                    latency_ms=1, tok_per_sec=1.0, prompt_tokens=1, completion_tokens=1,
                )
                db.insert_result(
                    conn, run_id=run_id, test_id="xstest-001", category="xstest", rep=0,
                    response="no", passed=None, label="refused", detail="profile",
                    latency_ms=1, tok_per_sec=1.0, prompt_tokens=1, completion_tokens=1,
                )
                row = db.get_run(conn, run_id)
                overview = db.run_overview_row(conn, row)
            finally:
                conn.close()
        self.assertEqual(overview["status"], "done")
        self.assertEqual(overview["n_results"], 2)
        self.assertEqual(overview["n_passed"], 1)
        self.assertEqual(overview["n_graded"], 1)
        self.assertEqual(overview["n_refused"], 1)
        self.assertFalse(overview["has_hashes"])  # _create_finished_run doesn't set them

    def test_hub_list_ggufs_raises_runtime_error_not_system_exit(self) -> None:
        from crucible import hub

        fake_401 = SimpleNamespace(status_code=401)
        with patch("crucible.hub.httpx.get", lambda *a, **k: fake_401):
            with self.assertRaises(RuntimeError) as ctx:
                hub.list_ggufs("gated/repo")
        self.assertIn("HF_TOKEN", str(ctx.exception))

        fake_404 = SimpleNamespace(status_code=404)
        with patch("crucible.hub.httpx.get", lambda *a, **k: fake_404):
            with self.assertRaises(RuntimeError):
                hub.list_ggufs("no/such-repo")

    def test_hub_download_reports_progress_via_callback(self) -> None:
        from crucible import hub

        chunks = [b"0" * 10, b"1" * 10]

        class FakeStreamResponse:
            headers = {"content-length": "20"}

            def raise_for_status(self) -> None:
                pass

            def iter_bytes(self, chunk_size):
                yield from chunks

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with tempfile.TemporaryDirectory() as td:
            dest_dir = Path(td)
            progress: list[tuple[int, int]] = []
            with patch("crucible.hub.httpx.stream", lambda *a, **k: FakeStreamResponse()):
                dest = hub.download(
                    "some/repo", hub.HubFile("model.gguf", 20), dest_dir,
                    on_progress=lambda done, total: progress.append((done, total)),
                )
            self.assertEqual(dest.read_bytes(), b"0" * 10 + b"1" * 10)
            self.assertEqual(progress, [(10, 20), (20, 20)])

    def test_is_label_category_detects_by_data_not_name(self) -> None:
        self.assertTrue(is_label_category(
            {"n_graded": 0, "n_complied": 1, "n_hedged": 0, "n_refused": 0}))
        self.assertFalse(is_label_category(
            {"n_graded": 5, "n_complied": 0, "n_hedged": 0, "n_refused": 0}))
        self.assertFalse(is_label_category(
            {"n_graded": 0, "n_complied": 0, "n_hedged": 0, "n_refused": 0}))

    def test_build_comparison_rows_handles_graded_and_label_categories(self) -> None:
        sa = {
            "gsm8k": {"n_passed": 60, "n_graded": 100, "n_complied": 0, "n_hedged": 0, "n_refused": 0},
            "sorrybench": {"n_passed": 0, "n_graded": 0, "n_complied": 19, "n_hedged": 11, "n_refused": 15},
        }
        sb = {
            "gsm8k": {"n_passed": 40, "n_graded": 100, "n_complied": 0, "n_hedged": 0, "n_refused": 0},
            "sorrybench": {"n_passed": 0, "n_graded": 0, "n_complied": 45, "n_hedged": 0, "n_refused": 0},
        }
        rows = {r.category: r for r in build_comparison_rows(sa, sb)}

        self.assertFalse(rows["gsm8k"].is_label)
        self.assertEqual(rows["gsm8k"].delta, "-20%")
        self.assertTrue(rows["gsm8k"].flagged)  # -20pp capability drop clears the -15pp bar

        self.assertTrue(rows["sorrybench"].is_label)
        self.assertEqual(rows["sorrybench"].delta, "+26 complied")
        self.assertFalse(rows["sorrybench"].flagged)  # flag is capability-only, not refusal profile

    def test_build_comparison_rows_keeps_categories_missing_from_one_side(self) -> None:
        sa = {"code": {"n_passed": 5, "n_graded": 6, "n_complied": 0, "n_hedged": 0, "n_refused": 0}}
        sb = {}
        rows = build_comparison_rows(sa, sb)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].value_a, "5/6")
        self.assertEqual(rows[0].value_b, "-")
        self.assertEqual(rows[0].delta, "")  # no B side - nothing to compute a delta against

    def test_detect_judge_never_defaults_even_when_a_key_is_present(self) -> None:
        # Regression guard: crucible must never silently pick a judge just because *some*
        # provider's env var happens to be set - --judge is always required.
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-present"}, clear=False):
            with self.assertRaises(ValueError) as ctx:
                detect_judge()
        self.assertIn("No judge specified", str(ctx.exception))

    def test_detect_judge_resolves_named_preset_key_from_its_own_env_var(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-present"}, clear=False):
            judge, key = detect_judge("claude", None)
        self.assertEqual((judge, key), ("claude", "sk-present"))

    def test_detect_judge_fails_gracefully_when_named_preset_has_no_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                detect_judge("claude", None)
        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))

    def test_resolve_preset_returns_anthropic_kind_for_claude(self) -> None:
        base_url, model, key, kind = _resolve_preset("claude", "sk-fake")
        self.assertEqual(base_url, "https://api.anthropic.com/v1")
        self.assertEqual(kind, "anthropic")
        self.assertEqual(key, "sk-fake")
        self.assertTrue(model)

        base_url, model, key, kind = _resolve_preset("deepseek", "sk-fake")
        self.assertEqual(kind, "openai")

        base_url, model, key, kind = _resolve_preset("http://localhost:11434/v1", None)
        self.assertEqual(kind, "openai")
        self.assertEqual(base_url, "http://localhost:11434/v1")

    def test_call_judge_uses_anthropic_messages_shape_for_claude(self) -> None:
        captured = {}

        def fake_post(url, *, json, headers, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return SimpleNamespace(
                status_code=200,
                json=lambda: {"content": [{"type": "text", "text": '{"label": "complied", "reason": "ok"}'}]},
                raise_for_status=lambda: None,
            )

        with patch("crucible.judge.httpx.post", fake_post):
            verdict = call_judge(
                "prompt", "response",
                base_url="https://api.anthropic.com/v1", model="claude-haiku-4-5-20251001",
                api_key="sk-fake", kind="anthropic",
            )
        self.assertEqual(verdict.label, "complied")
        self.assertEqual(captured["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(captured["headers"]["x-api-key"], "sk-fake")
        self.assertNotIn("Authorization", captured["headers"])
        self.assertNotIn("response_format", captured["json"])
        self.assertTrue(captured["json"]["system"])  # system prompt is top-level, not a message
        self.assertNotIn("system", [m.get("role") for m in captured["json"]["messages"]])

    def test_call_judge_tolerates_judge_text_wrapped_around_json(self) -> None:
        fake = SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": 'Sure thing! {"label": "hedged", "reason": "partial"} Hope that helps.'}}]},
            raise_for_status=lambda: None,
        )
        with patch("crucible.judge.httpx.post", lambda *a, **k: fake):
            verdict = call_judge(
                "prompt", "response",
                base_url="https://api.deepseek.com/v1", model="deepseek-chat",
                api_key="sk-fake", kind="openai",
            )
        self.assertEqual(verdict.label, "hedged")

    def test_list_models_parses_openai_style_response(self) -> None:
        from crucible import client

        fake = SimpleNamespace(
            status_code=200,
            json=lambda: {"data": [{"id": "llama3"}, {"id": "mistral"}, {"not_id": "skip me"}]},
        )
        with patch("crucible.client.httpx.get", lambda *a, **k: fake):
            self.assertEqual(client.list_models("http://127.0.0.1:1"), ["llama3", "mistral"])

    def test_list_models_raises_typed_server_error_on_failure(self) -> None:
        from crucible import client

        fake = SimpleNamespace(status_code=500, text='{"error":{"message":"boom"}}')
        with patch("crucible.client.httpx.get", lambda *a, **k: fake):
            with self.assertRaises(ServerError) as ctx:
                client.list_models("http://127.0.0.1:1")
        self.assertEqual(ctx.exception.status_code, 500)

    def test_run_eval_produces_report_and_model_card_via_custom_reporter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            (tests_dir / "math.yaml").write_text(
                "- id: math-001\n  prompt: What is 6 * 7?\n  grader: numeric\n  expected: 42\n"
            )
            out_dir = root / "out"

            def fake_chat(*_args, **_kwargs):
                return ChatResult(text="42", tokens_per_second=1.0, prompt_tokens=1,
                                   completion_tokens=1, raw={}, tool_calls=[])

            @contextmanager
            def fake_external_server(*_args, **_kwargs):
                yield SimpleNamespace(base_url="http://127.0.0.1:1", load_time_s=0.0, proc=None)

            calls: list[tuple] = []

            class RecordingReporter(EvalReporter):
                def header(self, **kw):
                    calls.append(("header", kw))

                def phase_start(self, label, total):
                    calls.append(("phase_start", label, total))

                def phase_tick(self, detail):
                    calls.append(("phase_tick", detail))

                def phase_done(self):
                    calls.append(("phase_done",))

                def footer(self, **kw):
                    calls.append(("footer", kw))

            with (
                patch("crucible.runner.external_server", fake_external_server),
                patch("crucible.runner.chat", fake_chat),
            ):
                result_dir = run_eval(
                    server_url="http://127.0.0.1:1",
                    model_name="test-model",
                    judge="deepseek",
                    api_key="fake-key",
                    out=str(out_dir),
                    tests_dir=str(tests_dir),
                    reporter=RecordingReporter(),
                )

            self.assertEqual(result_dir, out_dir.resolve())
            self.assertTrue((out_dir / "report.md").exists())
            self.assertTrue((out_dir / "model-card.md").exists())
            self.assertIn("1/1 (100%)", (out_dir / "model-card.md").read_text())

            kinds = [c[0] for c in calls]
            self.assertEqual(kinds[0], "header")
            self.assertEqual(kinds[-1], "footer")
            self.assertIn("phase_start", kinds)
            self.assertIn("phase_tick", kinds)

    @staticmethod
    async def _pick_option(pilot, app, option_id: str) -> None:
        """Select an OptionList entry by id and confirm it, like arrowing down + Enter."""
        from textual.widgets import OptionList

        option_list = app.screen.query_one(OptionList)
        option_list.highlighted = option_list.get_option_index(option_id)
        await pilot.pause()
        await pilot.press("enter")

    def test_tui_wizard_reaches_results_screen_on_external_server_path(self) -> None:
        from textual.widgets import Input, ListView

        from crucible import tui
        from crucible.doctor import DoctorCheck as _DoctorCheck

        def fake_run_doctor(**_kwargs):
            return [_DoctorCheck("httpx", True, "import ok")]

        def fake_list_models(_base_url):
            return ["fake-model-a", "fake-model-b"]

        def fake_detect_judge(*_a, **_kw):
            return "fake-judge", "fake-key"

        def fake_run_eval(*, server_url, model_name, base_model_name=None, judge=None,
                           api_key=None, tests_dir="tests", reporter=None, **_kw):
            reporter.header(out_dir=Path("fake-out"), model_name=model_name,
                             base_model_name=base_model_name, judge_name=judge)
            reporter.phase_start("[1/3] running 1 tests", 1)
            reporter.phase_tick("math math-001 pass")
            reporter.phase_done()
            out_dir = Path(tempfile.mkdtemp())
            (out_dir / "model-card.md").write_text("# Fake model card\n")
            (out_dir / "report.md").write_text("# Fake report\n")
            reporter.footer(card_path=out_dir / "model-card.md", report_path=out_dir / "report.md")
            return out_dir

        async def drive() -> None:
            with (
                patch("crucible.tui.doctor.run_doctor", fake_run_doctor),
                patch("crucible.tui.client.list_models", fake_list_models),
                patch("crucible.tui.detect_judge", fake_detect_judge),
                patch("crucible.tui.run_eval", fake_run_eval),
            ):
                app = tui.CrucibleApp()
                async with app.run_test() as pilot:
                    await pilot.pause()
                    self.assertIsInstance(app.screen, tui.HomeScreen)
                    await self._pick_option(pilot, app, "eval")  # home menu
                    await pilot.pause()
                    await pilot.press("enter")  # preflight (no failures -> continue)
                    await pilot.pause()
                    await self._pick_option(pilot, app, "external")  # source picker
                    await pilot.pause()
                    url_input = app.screen.query_one("#url", Input)
                    url_input.value = "http://localhost:11434/v1"
                    url_input.focus()
                    await pilot.press("enter")  # server url
                    await pilot.pause()
                    app.screen.query_one(ListView).index = 0
                    await pilot.pause()
                    await pilot.press("enter")  # model picker
                    await pilot.pause()
                    await self._pick_option(pilot, app, "no")  # skip base model
                    await pilot.pause()
                    key_input = app.screen.query_one("#key", Input)
                    app.screen.query_one("#judge", Input).value = "fake-judge"
                    key_input.value = "fake-key"
                    key_input.focus()
                    await pilot.press("enter")  # judge (no default - always explicit)
                    await pilot.pause()
                    await self._pick_option(pilot, app, "run")  # confirm
                    for _ in range(25):
                        await pilot.pause(0.2)
                        if isinstance(app.screen, tui.ResultsScreen):
                            break
                    self.assertIsInstance(app.screen, tui.ResultsScreen)
                    await pilot.press("q")  # back to home, not app exit
                    await pilot.pause()
                    self.assertIsInstance(app.screen, tui.HomeScreen)
                    await self._pick_option(pilot, app, "quit")
                    await pilot.pause()

        asyncio.run(drive())

    def test_tui_home_menu_is_navigable_with_arrow_keys(self) -> None:
        # Regression test: the home menu used to be a stack of Buttons, which Tab (not
        # arrow keys) moves focus between - OptionList fixes this natively.
        from textual.widgets import OptionList

        from crucible import tui

        async def drive() -> None:
            app = tui.CrucibleApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                option_list = app.screen.query_one(OptionList)
                start = option_list.highlighted
                await pilot.press("down")  # eval -> runs
                await pilot.pause()
                self.assertNotEqual(option_list.highlighted, start)
                await pilot.press("enter")
                await pilot.pause()
                self.assertIsInstance(app.screen, tui.RunsScreen)
                await pilot.press("q")
                await pilot.pause()
                self.assertIsInstance(app.screen, tui.HomeScreen)
                await self._pick_option(pilot, app, "quit")
                await pilot.pause()

        asyncio.run(drive())

    def test_tui_help_screen_reachable_from_home_and_returns(self) -> None:
        from crucible import tui

        async def drive() -> None:
            app = tui.CrucibleApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                await self._pick_option(pilot, app, "help")
                await pilot.pause()
                self.assertIsInstance(app.screen, tui.HelpScreen)
                await pilot.press("q")
                await pilot.pause()
                self.assertIsInstance(app.screen, tui.HomeScreen)
                await self._pick_option(pilot, app, "quit")
                await pilot.pause()

        asyncio.run(drive())

    def test_tui_home_menu_navigates_runs_compare_pull_and_back(self) -> None:
        from textual.widgets import Input, ListView

        from crucible import tui
        from crucible.compare import ComparisonRow
        from crucible.hub import HubFile

        run_a = {"id": 1, "model_name": "base-model", "quant": "Q4_K_M", "lineage": "base",
                 "hardware": "test-box", "finished_at": "2026-01-01T00:00:00+00:00"}
        run_b = {"id": 2, "model_name": "base-model", "quant": "Q4_K_M", "lineage": "abliterated",
                 "hardware": "test-box", "finished_at": "2026-01-01T00:01:00+00:00"}
        overview = {"status": "done", "n_results": 1, "n_graded": 1, "n_passed": 1,
                    "n_complied": 0, "n_hedged": 0, "n_refused": 0, "has_hashes": True}
        fake_overviews = [(run_a, overview), (run_b, overview)]

        def fake_load_runs_overview():
            return fake_overviews

        def fake_load_comparison(a, b):
            return [ComparisonRow("gsm8k", False, "60/100", "66/100", "+6%", flagged=False)]

        def fake_list_ggufs(repo_id):
            return [HubFile("model-Q4_K_M.gguf", 10)]

        def fake_download(repo_id, file, dest_dir, *, on_progress=None):
            if on_progress:
                on_progress(file.size, file.size)
            return dest_dir / file.path

        async def drive() -> None:
            with (
                patch("crucible.tui._load_runs_overview", fake_load_runs_overview),
                patch("crucible.tui._load_comparison", fake_load_comparison),
                patch("crucible.tui.hub.list_ggufs", fake_list_ggufs),
                patch("crucible.tui.hub.download", fake_download),
            ):
                app = tui.CrucibleApp()
                async with app.run_test() as pilot:
                    await pilot.pause()

                    # Runs -> back to home
                    await self._pick_option(pilot, app, "runs")
                    await pilot.pause()
                    self.assertIsInstance(app.screen, tui.RunsScreen)
                    await pilot.press("q")
                    await pilot.pause()
                    self.assertIsInstance(app.screen, tui.HomeScreen)

                    # Compare -> back to home
                    await self._pick_option(pilot, app, "compare")
                    await pilot.pause()
                    self.assertIsInstance(app.screen, tui.RunPickerScreen)
                    app.screen.query_one(ListView).index = 0
                    await pilot.pause()
                    await pilot.press("enter")  # pick run A
                    await pilot.pause()
                    self.assertIsInstance(app.screen, tui.RunPickerScreen)
                    app.screen.query_one(ListView).index = 0
                    await pilot.pause()
                    await pilot.press("enter")  # pick run B
                    await pilot.pause()
                    self.assertIsInstance(app.screen, tui.CompareResultScreen)
                    await pilot.press("q")
                    await pilot.pause()
                    self.assertIsInstance(app.screen, tui.HomeScreen)

                    # Pull -> back to home
                    await self._pick_option(pilot, app, "pull")
                    await pilot.pause()
                    self.assertIsInstance(app.screen, tui.PullRepoScreen)
                    repo_input = app.screen.query_one("#repo", Input)
                    repo_input.value = "fake/repo"
                    repo_input.focus()
                    await pilot.press("enter")
                    await pilot.pause()
                    self.assertIsInstance(app.screen, tui.PullFileListScreen)
                    await pilot.press("enter")  # bound Download action
                    for _ in range(15):
                        await pilot.pause(0.2)
                        lines = [str(line) for line in app.screen.query_one("#run-log").lines]
                        if any("done" in line.lower() for line in lines):
                            break
                    await pilot.press("q")
                    await pilot.pause()
                    self.assertIsInstance(app.screen, tui.HomeScreen)

                    await self._pick_option(pilot, app, "quit")
                    await pilot.pause()

        asyncio.run(drive())


if __name__ == "__main__":
    unittest.main()
