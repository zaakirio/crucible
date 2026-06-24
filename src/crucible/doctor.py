"""Environment checks for local Crucible runs."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str


def run_doctor(*, db_path: str, tests_dir: str, docs_dir: str | None = None, model: str | None = None) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    for module in ("httpx", "yaml", "matplotlib"):
        try:
            importlib.import_module(module)
            checks.append(DoctorCheck(module, True, "import ok"))
        except Exception as e:
            checks.append(DoctorCheck(module, False, f"import failed: {e}"))

    try:
        from .server import _find_llama_server

        checks.append(DoctorCheck("llama-server", True, str(_find_llama_server())))
    except Exception as e:
        checks.append(DoctorCheck("llama-server", False, str(e)))

    try:
        from .ppl import _find_llama_perplexity

        checks.append(DoctorCheck("llama-perplexity", True, str(_find_llama_perplexity())))
    except Exception as e:
        checks.append(DoctorCheck("llama-perplexity", False, str(e)))

    tests = Path(tests_dir)
    checks.append(DoctorCheck("tests", tests.exists() and tests.is_dir(), str(tests)))
    if docs_dir:
        docs = Path(docs_dir)
        checks.append(DoctorCheck("docs", docs.exists() and docs.is_dir(), str(docs)))
    db_parent = Path(db_path).parent
    checks.append(DoctorCheck("database", db_parent.exists() and db_parent.is_dir(), str(Path(db_path))))
    if model:
        model_path = Path(model).expanduser()
        checks.append(DoctorCheck("model", model_path.exists() and model_path.is_file(), str(model_path)))
    return checks


def render_doctor(checks: list[DoctorCheck]) -> str:
    lines = ["crucible doctor"]
    for check in checks:
        mark = "ok" if check.ok else "FAIL"
        lines.append(f"  {mark:4} {check.name:17} {check.detail}")
    return "\n".join(lines) + "\n"
