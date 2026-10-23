"""Tests for the Stage 7 query service and append-only log."""

import json

from src.ask import MiniCrosscheckService


class FakeRetriever:
    """Return fixed source chunks and capture requested retrieval options."""

    def __init__(self) -> None:
        self.options: dict = {}

    def retrieve(self, query: str, **options: object) -> list[dict]:
        self.options = options
        return [{
            "id": "manual::C::1",
            "text": "Maximum process temperature: 120 deg C",
            "metadata": {"doc_name": "manual.pdf", "doc_number": "MAN-1", "revision": "C", "page": 1},
            "score": 0.9,
            "abstention_score": 0.9,
            "scores": {"vector_similarity": 0.9},
        }]


class FakeBackend:
    """Return a stable answer without external model calls."""

    def complete(self, prompt: str) -> str:
        return "120 deg C [manual.pdf, revision C, page 1]"


def test_ask_retrieves_answers_and_appends_scores(tmp_path) -> None:
    """One service call should return evidence and append it to query_log.jsonl."""
    retriever = FakeRetriever()
    query_log = tmp_path / "outputs" / "query_log.jsonl"
    service = MiniCrosscheckService(
        retriever,
        FakeBackend(),
        {"mode": "hybrid", "top_k": 5, "reranker_enabled": False, "rrf_k": 60},
        0.25,
        query_log,
    )

    result = service.ask("What is the limit?", revision="C", latest_only=True)
    record = json.loads(query_log.read_text(encoding="utf-8"))

    assert result["answer"].startswith("120 deg C")
    assert retriever.options["revision"] == "C"
    assert record["chunks"][0]["scores"] == {"vector_similarity": 0.9}
    assert record["settings"]["latest_only"] is True