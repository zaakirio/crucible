from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crucible import db

try:
    from fastapi.testclient import TestClient

    from crucible.lab import STATIC_DIR, create_app

    HAVE_LAB = True
except (ImportError, SystemExit):
    HAVE_LAB = False


@unittest.skipUnless(HAVE_LAB, "lab extra not installed")
class LabApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "results.db"
        conn = db.connect(self.db_path)
        self.run_a = db.create_run(
            conn, model_file="m-Q4_K_M.gguf", model_name="m", quant="Q4_K_M",
            lineage="base", hardware="test", llama_cpp_commit="abc", ctx=4096,
            ngl=99, repeat=1, started_at="2026-01-01T00:00:00+00:00",
        )
        self.run_b = db.create_run(
            conn, model_file="m-Q4_K_M.gguf", model_name="m-uncensored", quant="Q4_K_M",
            lineage="abliterated", hardware="test", llama_cpp_commit="abc", ctx=4096,
            ngl=99, repeat=1, started_at="2026-01-02T00:00:00+00:00",
        )
        for run_id, passed in ((self.run_a, 1), (self.run_b, 0)):
            db.insert_result(
                conn, run_id=run_id, test_id="gsm-001", category="gsm8k", rep=0,
                response="42", passed=passed, label=None, detail="expected 42",
                latency_ms=100, tok_per_sec=50.0, prompt_tokens=20, completion_tokens=5,
            )
            db.insert_result(
                conn, run_id=run_id, test_id="sb-001", category="sorrybench", rep=0,
                response="sure, here is how", passed=None,
                label="complied" if run_id == self.run_b else "refused",
                detail=None, latency_ms=90, tok_per_sec=48.0,
                prompt_tokens=25, completion_tokens=12,
            )
        db.finish_run(conn, self.run_a, "2026-01-01T01:00:00+00:00")
        db.finish_run(conn, self.run_b, "2026-01-02T01:00:00+00:00")
        conn.commit()
        conn.close()
        self.client = TestClient(create_app(self.db_path))

    def tearDown(self):
        self.tmp.cleanup()

    def test_meta(self):
        meta = self.client.get("/api/meta").json()
        self.assertEqual(meta["n_runs"], 2)
        self.assertEqual(meta["n_results"], 4)
        self.assertIn("m-uncensored", meta["models"])

    def test_runs_include_overview_tallies(self):
        runs = self.client.get("/api/runs").json()
        self.assertEqual(len(runs), 2)
        newest = runs[0]
        self.assertEqual(newest["id"], self.run_b)
        self.assertEqual(newest["n_complied"], 1)
        self.assertEqual(newest["status"], "done")

    def test_run_detail_categories(self):
        detail = self.client.get(f"/api/runs/{self.run_a}").json()
        cats = {c["category"]: c for c in detail["categories"]}
        self.assertEqual(cats["gsm8k"]["n_passed"], 1)
        self.assertEqual(cats["sorrybench"]["n_refused"], 1)

    def test_run_detail_404(self):
        self.assertEqual(self.client.get("/api/runs/999").status_code, 404)

    def test_results_filters(self):
        r = self.client.get(f"/api/runs/{self.run_b}/results", params={"category": "sorrybench"})
        rows = r.json()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "complied")

        r = self.client.get(f"/api/runs/{self.run_b}/results", params={"status": "failed"})
        self.assertEqual([row["test_id"] for row in r.json()], ["gsm-001"])

        r = self.client.get(f"/api/runs/{self.run_b}/results", params={"q": "here is how"})
        self.assertEqual(len(r.json()), 1)

    def test_results_search_treats_like_wildcards_literally(self):
        r = self.client.get(f"/api/runs/{self.run_b}/results", params={"q": "%"})
        self.assertEqual(r.json(), [])

    def test_unknown_api_route_is_404(self):
        self.assertEqual(self.client.get("/api/no-such-route").status_code, 404)

    def test_spa_never_serves_files_outside_static_root(self):
        if not STATIC_DIR.is_dir():
            self.skipTest("frontend not built")
        r = self.client.get("/%2fetc%2fpasswd")
        self.assertEqual(r.status_code, 200)
        self.assertIn("<!doctype html>", r.text.lower())

    def test_results_rejects_bad_status(self):
        r = self.client.get(f"/api/runs/{self.run_b}/results", params={"status": "nope"})
        self.assertEqual(r.status_code, 422)

    def test_compare(self):
        payload = self.client.get(f"/api/compare?a={self.run_a}&b={self.run_b}").json()
        rows = {r["category"]: r for r in payload["rows"]}
        self.assertEqual(rows["gsm8k"]["delta"], "-100%")
        self.assertTrue(rows["gsm8k"]["flagged"])
        self.assertEqual(rows["sorrybench"]["delta"], "+1 complied")
        self.assertTrue(rows["sorrybench"]["is_label"])

    def test_playground_rejects_non_http_server(self):
        r = self.client.post("/api/playground/chat", json={"server": "file:///etc", "messages": []})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
