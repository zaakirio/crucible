"""Pull GGUF files from a Hugging Face model repo — no `huggingface_hub` dependency.

Two endpoints, both public:
  https://huggingface.co/api/models/{repo}/tree/main      -> file listing (path, size)
  https://huggingface.co/{repo}/resolve/main/{file}       -> the file itself

$HF_TOKEN is honored for gated/private repos. Existing files with a matching size are
skipped, so re-running a pull is a cheap no-op.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass
class HubFile:
    path: str
    size: int


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def list_ggufs(repo_id: str) -> list[HubFile]:
    """All .gguf files in a repo (mmproj projectors included — they're .gguf too)."""
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    r = httpx.get(url, headers=_auth_headers(), timeout=30, follow_redirects=True)
    if r.status_code == 401:
        raise SystemExit(f"{repo_id} is gated/private — set $HF_TOKEN (huggingface.co/settings/tokens)")
    if r.status_code == 404:
        raise SystemExit(f"No such repo: {repo_id}")
    r.raise_for_status()
    return sorted(
        (HubFile(f["path"], f.get("size") or 0) for f in r.json()
         if f["path"].lower().endswith(".gguf")),
        key=lambda f: f.path,
    )


def download(repo_id: str, file: HubFile, dest_dir: Path) -> Path:
    """Stream one file to dest_dir with a progress line. Skips if already complete."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(file.path).name
    if dest.exists() and file.size and dest.stat().st_size == file.size:
        print(f"  have  {dest.name} ({file.size / 1e9:.2f} GB) — skipping")
        return dest

    url = f"https://huggingface.co/{repo_id}/resolve/main/{file.path}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = 0
    with httpx.stream("GET", url, headers=_auth_headers(), timeout=60,
                      follow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or file.size or 0)
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  pull  {dest.name}  {done / 1e9:5.2f}/{total / 1e9:.2f} GB ({pct:3.0f}%)",
                          end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    tmp.rename(dest)
    return dest
