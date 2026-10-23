"""Append query inputs, retrieval results, and scores to a JSONL audit log."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_query_log(
    log_file: Path,
    query: str,
    settings: dict[str, Any],
    chunks: list[dict[str, Any]],
    answer: str | None = None,
    error: str | None = None,
) -> None:
    """Append one self-contained query record without overwriting prior runs."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "settings": settings,
        "chunks": [
            {
                "id": chunk.get("id"),
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {}),
                "score": chunk.get("score"),
                "abstention_score": chunk.get("abstention_score"),
                "scores": chunk.get("scores", {}),
            }
            for chunk in chunks
        ],
        "answer": answer,
        "error": error,
    }
    with log_file.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")