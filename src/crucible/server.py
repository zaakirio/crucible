"""llama-server lifecycle: spawn it, wait until healthy, kill it cleanly.

We drive the *server* (not llama-cpp-python) so we evaluate a model exactly as it's
served in production: same chat template (--jinja), same sampler defaults, same
tool-call parsing the users of your published GGUFs get.
"""

from __future__ import annotations

import contextlib
import os
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


class PreflightError(RuntimeError):
    """Raised before spawning when the machine cannot serve this model (e.g. low memory)."""


# Below this size a .gguf is a test fixture / dummy, not a real servable model: skip preflight.
_MIN_SERVABLE_BYTES = 256 * 1024 * 1024


def _find_llama_server() -> Path:
    """Locate the llama-server binary.

    Order: $CRUCIBLE_LLAMA_SERVER, then walk up from cwd for llama.cpp/build/bin, then $PATH.
    """
    env = os.environ.get("CRUCIBLE_LLAMA_SERVER")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(f"$CRUCIBLE_LLAMA_SERVER points at {p}, which does not exist")

    # Walk up from cwd looking for llama.cpp/build/bin/llama-server (handles running
    # from the repo root or from crucible/ inside it).
    rel = Path("llama.cpp") / "build" / "bin" / "llama-server"
    for base in [Path.cwd(), *Path.cwd().parents]:
        candidate = base / rel
        if candidate.exists():
            return candidate

    from shutil import which

    found = which("llama-server")
    if found:
        return Path(found)

    raise FileNotFoundError(
        "Could not find llama-server. Build it (./scripts/build-llama.sh) or set "
        "$CRUCIBLE_LLAMA_SERVER to its path."
    )


def llama_cpp_commit() -> str | None:
    """Short git commit of the llama.cpp checkout the binary was built from, or None.

    Recorded per run: sampler defaults and tool-call parsers change between versions, so a
    score shift might be the engine, not the model.
    """
    try:
        binary = _find_llama_server()
    except FileNotFoundError:
        return None
    # binary is <root>/llama.cpp/build/bin/llama-server -> llama.cpp dir is parents[2]
    if len(binary.parents) < 3:
        return None
    llama_dir = binary.parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(llama_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (subprocess.SubprocessError, OSError):
        return None


def llama_server_version(binary: Path | None = None) -> str | None:
    """Build string from `llama-server --version`, e.g. '9759 (099b579ac)', or None."""
    try:
        binary = binary or _find_llama_server()
        out = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    text = (out.stderr or "") + (out.stdout or "")
    m = re.search(r"version:\s*(\S+)\s*\(([0-9a-f]+)\)", text)
    if m:
        return f"{m.group(1)} ({m.group(2)})"
    m = re.search(r"version:\s*(\S+)", text)
    return m.group(1) if m else None


def _gib(n: float) -> float:
    """Bytes to GiB - matches the '24 GB' macOS reports for RAM (powers of 1024)."""
    return n / (1024 ** 3)


def total_memory_bytes() -> int | None:
    """Total physical RAM, or None if it can't be read."""
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def available_memory_bytes() -> int | None:
    """Best-effort reclaimable physical memory on macOS (free + inactive + speculative pages).

    Used only as a preflight guard; an over- or under-estimate just shifts where the warning
    line sits, never correctness of a run. Returns None on non-macOS / parse failure (skip check).
    """
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    text = out.stdout
    page_m = re.search(r"page size of (\d+) bytes", text)
    page = int(page_m.group(1)) if page_m else 4096

    def pages(label: str) -> int:
        m = re.search(rf"{re.escape(label)}:\s+(\d+)", text)
        return int(m.group(1)) if m else 0

    reclaimable = pages("Pages free") + pages("Pages inactive") + pages("Pages speculative")
    return reclaimable * page if reclaimable else None


def running_llama_servers() -> list[int]:
    """PIDs of llama-server processes already running (Crucible can't share one, so these
    are a memory hazard before a run). Empty list on any lookup failure."""
    try:
        out = subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(p) for p in out.stdout.split() if p.strip().isdigit()]


def memory_preflight_message(
    model_name: str, size_bytes: int, available_bytes: int | None, others: list[int],
    *, headroom: float = 1.1,
) -> str | None:
    """A single actionable line if this model almost certainly won't fit, else None.

    Pure (all inputs injected) so it is unit-testable without touching real memory.
    """
    if available_bytes is None or size_bytes < _MIN_SERVABLE_BYTES:
        return None
    need = int(size_bytes * headroom)
    if available_bytes >= need:
        return None
    parts = [
        f"Not enough free memory to serve {model_name} "
        f"(~{_gib(need):.1f} GB needed, ~{_gib(available_bytes):.1f} GB free).",
    ]
    if others:
        pids = " ".join(str(p) for p in others)
        parts.append(
            f"Another llama-server is already running (PID {', '.join(str(p) for p in others)}); "
            f"Crucible spawns its own, so stop the stray one first:  kill {pids}"
        )
    parts.append(
        "Then retry, or lower --ngl / pick a smaller quant. "
        "Override this check with CRUCIBLE_SKIP_PREFLIGHT=1."
    )
    return "  ".join(parts)


def _preflight(model_path: Path) -> None:
    if os.environ.get("CRUCIBLE_SKIP_PREFLIGHT"):
        return
    msg = memory_preflight_message(
        model_path.name,
        model_path.stat().st_size,
        available_memory_bytes(),
        running_llama_servers(),
    )
    if msg:
        raise PreflightError(msg)


def list_gguf_files(root: Path) -> list[Path]:
    """All .gguf files under root, recursively, sorted by path."""
    return sorted(Path(root).rglob("*.gguf"))


def _free_port() -> int:
    """Grab an OS-assigned free port. Small TOCTOU race; fine for local single-runner use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class ServerHandle:
    base_url: str
    model_path: Path | None
    load_time_s: float
    proc: subprocess.Popen | None


@contextlib.contextmanager
def llama_server(
    model_path: str | Path,
    *,
    ngl: int = 99,
    ctx: int = 4096,
    n_parallel: int = 1,
    jinja: bool = True,
    flash_attn: str = "auto",
    health_timeout_s: float = 180.0,
    log_to: Path | None = None,
):
    """Context manager that yields a healthy ServerHandle and tears the server down on exit."""
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    _preflight(model_path)  # fail fast on a known-OOM situation, before the slow model load

    binary = _find_llama_server()
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    cmd = [
        str(binary),
        "-m", str(model_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-ngl", str(ngl),
        "--ctx-size", str(ctx),
        "--flash-attn", flash_attn,
        "-np", str(n_parallel),
    ]
    if jinja:
        cmd.append("--jinja")

    log_file = open(log_to, "w") if log_to else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    try:
        load_time_s = _wait_healthy(base_url, proc, timeout_s=health_timeout_s)
        yield ServerHandle(
            base_url=base_url,
            model_path=model_path,
            load_time_s=load_time_s,
            proc=proc,
        )
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        if proc.poll() is None:
            proc.kill()
        if log_to:
            log_file.close()


@contextlib.contextmanager
def external_server(base_url: str, *, health_timeout_s: float = 10.0):
    """Yield a ServerHandle for an already-running OpenAI-compatible server at base_url.

    No spawn, no kill, no memory preflight. Just verify the server is reachable and hand
    back a handle. tok/s will be measured client-side (timing_source='client').
    """
    url = base_url.rstrip("/")
    health_url = f"{url}/health"
    started = time.monotonic()
    reachable = False
    last_err = ""
    while time.monotonic() - started < health_timeout_s:
        try:
            r = httpx.get(health_url, timeout=2.0)
            if r.status_code < 500:
                reachable = True
                break
        except httpx.HTTPError as e:
            last_err = str(e)
        time.sleep(0.5)

    if not reachable:
        # Many servers (Ollama, vLLM) don't expose /health - fall back to a models list ping.
        try:
            r = httpx.get(f"{url}/v1/models", timeout=5.0)
            reachable = r.status_code < 500
        except httpx.HTTPError as e:
            last_err = str(e)

    if not reachable:
        raise RuntimeError(
            f"Could not reach server at {url} (tried /health and /v1/models). "
            f"Last error: {last_err}. Is the server running?"
        )

    yield ServerHandle(
        base_url=url,
        model_path=None,
        load_time_s=0.0,
        proc=None,
    )


def _wait_healthy(base_url: str, proc: subprocess.Popen, *, timeout_s: float) -> float:
    """Poll /health until the server reports ok. Returns seconds-to-ready."""
    started = time.monotonic()
    deadline = started + timeout_s
    url = f"{base_url}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited early (code {proc.returncode}) before becoming healthy. "
                "Pass log_to=... to capture its output."
            )
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return time.monotonic() - started
        except (httpx.HTTPError, ValueError):
            pass  # loading model / not up yet
        time.sleep(0.5)
    raise TimeoutError(f"llama-server did not become healthy within {timeout_s}s")
