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
    warn: bool = False  # ok but worth flagging (e.g. a stray server, low free memory)


def run_doctor(*, db_path: str, tests_dir: str, docs_dir: str | None = None, model: str | None = None) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    for module in ("httpx", "yaml", "matplotlib"):
        try:
            importlib.import_module(module)
            checks.append(DoctorCheck(module, True, "import ok"))
        except Exception as e:
            checks.append(DoctorCheck(module, False, f"import failed: {e}"))

    try:
        from .server import _find_llama_server, llama_server_version

        binp = _find_llama_server()
        ver = llama_server_version(binp)
        detail = str(binp) + (f"  (build {ver})" if ver else "")
        checks.append(DoctorCheck("llama-server", True, detail))
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

    # Memory headroom: the single most common local failure is a second model that won't fit.
    # With --model this is a real go/no-go preflight; without it, just report free memory.
    try:
        from .server import _gib, available_memory_bytes, running_llama_servers, total_memory_bytes

        total = total_memory_bytes()
        avail = available_memory_bytes()
        if total and avail is not None:
            detail = f"~{_gib(avail):.1f} GB free of {_gib(total):.0f} GB"
            model_path = Path(model).expanduser() if model else None
            if model_path and model_path.is_file():
                need = model_path.stat().st_size * 1.1
                detail += f"; {model_path.name} needs ~{_gib(need):.1f} GB"
                checks.append(DoctorCheck("memory", avail >= need, detail))
            else:
                checks.append(DoctorCheck("memory", True, detail, warn=_gib(avail) < 4))
        strays = running_llama_servers()
        if strays:
            pids = " ".join(str(p) for p in strays)
            checks.append(DoctorCheck(
                "llama-server (stray)", True,
                f"PID {', '.join(str(p) for p in strays)} already running - Crucible spawns its "
                f"own, so stop strays before a run:  kill {pids}",
                warn=True,
            ))
    except Exception:
        pass  # memory introspection is best-effort and must never break doctor

    return checks


def render_doctor(checks: list[DoctorCheck]) -> str:
    lines = ["crucible doctor"]
    for check in checks:
        mark = "warn" if (check.ok and check.warn) else ("ok" if check.ok else "FAIL")
        lines.append(f"  {mark:4} {check.name:17} {check.detail}")
    return "\n".join(lines) + "\n"
