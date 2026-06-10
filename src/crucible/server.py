"""llama-server lifecycle: spawn it, wait until healthy, kill it cleanly.

We drive the *server* (not llama-cpp-python) so we evaluate a model exactly as it's
served in production: same chat template (--jinja), same sampler defaults, same
tool-call parsing the users of your published GGUFs get.
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


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


def _free_port() -> int:
    """Grab an OS-assigned free port. Small TOCTOU race; fine for local single-runner use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class ServerHandle:
    base_url: str
    model_path: Path
    load_time_s: float
    proc: subprocess.Popen


@contextlib.contextmanager
def llama_server(
    model_path: str | Path,
    *,
    ngl: int = 99,
    ctx: int = 4096,
    jinja: bool = True,
    flash_attn: str = "auto",
    health_timeout_s: float = 180.0,
    log_to: Path | None = None,
):
    """Context manager that yields a healthy ServerHandle and tears the server down on exit."""
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

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
