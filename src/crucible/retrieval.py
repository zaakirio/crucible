"""Tiny local retrieval for grounded QA / RAG-style evals.

This is intentionally simple: a lightweight term-overlap retriever over local docs so the
harness can turn a question into retrieved context without depending on a full vector DB.
It is good enough for small, inspectable corpora and for validating the end-to-end eval path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+")
TEXT_EXTS = {".md", ".markdown", ".txt", ".rst"}


@dataclass(frozen=True)
class Chunk:
    source: str
    index: int
    text: str
    score: int = 0


def _tokens(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1}


def _chunk_text(text: str, *, max_chars: int = 900) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for para in paras:
        if size and size + len(para) + 2 > max_chars:
            chunks.append("\n\n".join(buf))
            buf = []
            size = 0
        buf.append(para)
        size += len(para) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks or [text.strip()]


def load_chunks(docs_dir: str | Path) -> list[Chunk]:
    root = Path(docs_dir)
    if not root.exists():
        raise FileNotFoundError(f"docs dir not found: {root}")

    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTS:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        for i, chunk in enumerate(_chunk_text(text)):
            chunks.append(Chunk(source=str(path.relative_to(root)), index=i, text=chunk))
    return chunks


def score_chunks(query: str, chunks: list[Chunk]) -> list[Chunk]:
    q_tokens = _tokens(query)
    scored: list[Chunk] = []
    for chunk in chunks:
        c_tokens = _tokens(chunk.text)
        overlap = len(q_tokens & c_tokens)
        bonus = 0
        if query.lower() in chunk.text.lower():
            bonus += 4
        if any(tok in chunk.text.lower() for tok in q_tokens):
            bonus += 1
        scored.append(Chunk(chunk.source, chunk.index, chunk.text, overlap + bonus))
    return sorted(scored, key=lambda c: (-c.score, c.source, c.index))


def retrieve_context(
    query: str,
    docs_dir: str | Path,
    *,
    top_k: int = 3,
    max_chars: int = 2500,
) -> str:
    """Return a compact context block for a query, drawn from local docs."""
    chunks = score_chunks(query, load_chunks(docs_dir))[:top_k]
    if not chunks:
        raise ValueError(f"no retrievable text files found under {docs_dir}")

    blocks: list[str] = []
    used = 0
    for chunk in chunks:
        body = chunk.text.strip()
        block = f"[{chunk.source}#{chunk.index}]\n{body}"
        if used + len(block) + 2 > max_chars:
            break
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks)
