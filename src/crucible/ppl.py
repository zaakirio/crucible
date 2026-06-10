"""WikiText-2 perplexity via llama.cpp's llama-perplexity binary.

This is the literature-standard intrinsic metric (arXiv 2601.14277 reports it alongside
GSM8K/IFEval per quant). It complements task pass-rates: perplexity moves smoothly with
quantization while task scores move in jumps, so it catches degradation between task cliffs.

The corpus is the standard wikitext-2-raw test set, fetched once from ggml-org's CI bucket
(same file llama.cpp's own scripts use) and cached under ~/.cache/crucible/.

PPL is computed over a fixed number of 512-token chunks (default 32 ≈ 16K tokens) so a
sweep stays minutes, not hours. Values are only comparable at the SAME chunk count — it is
stored alongside the value for that reason.
"""

from __future__ import annotations

import re
import subprocess
import zipfile
from pathlib import Path

import httpx

from .server import _find_llama_server

WIKITEXT_ZIP_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
CACHE_DIR = Path.home() / ".cache" / "crucible"

_FINAL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([\d.]+)")


def wikitext_test_file() -> Path:
    """Download + extract wiki.test.raw once; return the cached path."""
    dest = CACHE_DIR / "wiki.test.raw"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / "wikitext-2-raw-v1.zip"
    with httpx.stream("GET", WIKITEXT_ZIP_URL, timeout=60, follow_redirects=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        member = next(n for n in z.namelist() if n.endswith("wiki.test.raw"))
        dest.write_bytes(z.read(member))
    zip_path.unlink(missing_ok=True)
    return dest


def _find_llama_perplexity() -> Path:
    """llama-perplexity ships next to llama-server in build/bin."""
    sibling = _find_llama_server().parent / "llama-perplexity"
    if not sibling.exists():
        raise FileNotFoundError(
            f"{sibling} not found — rebuild llama.cpp (it's part of the default build)"
        )
    return sibling


def measure_ppl(model_path: Path, *, chunks: int = 32, ngl: int = 99,
                timeout_s: float = 1800) -> float:
    """Run llama-perplexity and return the final PPL estimate."""
    corpus = wikitext_test_file()
    binary = _find_llama_perplexity()
    proc = subprocess.run(
        [str(binary), "-m", str(model_path), "-f", str(corpus),
         "--chunks", str(chunks), "-ngl", str(ngl)],
        capture_output=True, text=True, timeout=timeout_s,
    )
    out = proc.stdout + proc.stderr
    m = _FINAL_RE.search(out)
    if not m:
        tail = "\n".join(out.strip().splitlines()[-8:])
        raise RuntimeError(f"llama-perplexity gave no final estimate (exit {proc.returncode}):\n{tail}")
    return float(m.group(1))
