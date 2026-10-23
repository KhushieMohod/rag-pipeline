"""Tests for Stage 6 source-level metrics and faithfulness checks."""

import math

from src.evaluate import expected_sources, faithfulness_check, source_retrieval_metrics


def test_expected_sources_supports_single_and_multiple_documents() -> None:
    """Question source labels should normalize for source-level evaluation."""
    assert expected_sources({"source": "manual.pdf"}) == {"manual.pdf"}
    assert expected_sources({"sources": ["manual.pdf", "ecn.pdf"]}) == {"manual.pdf", "ecn.pdf"}
    assert expected_sources({"source": None}) == set()


def test_source_metrics_deduplicate_chunks_from_same_file() -> None:
    """Repeated chunks from one PDF must not inflate source-file ranking metrics."""
    results = [
        {"metadata": {"doc_name": "wrong.pdf"}},
        {"metadata": {"doc_name": "wrong.pdf"}},
        {"metadata": {"doc_name": "manual.pdf"}},
    ]

    metrics = source_retrieval_metrics({"manual.pdf"}, results, k=3)

    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 0.5
    assert math.isclose(metrics["ndcg"], 1.0 / math.log2(3))


def test_faithfulness_requires_retrieved_citation_and_claimed_value() -> None:
    """Cited metadata and numeric values must both match retrieved evidence."""
    chunks = [{
        "text": "Maximum process temperature: 120 deg C",
        "metadata": {"doc_name": "manual.pdf", "revision": "C", "page": 1},
    }]

    assert faithfulness_check("120 deg C [manual.pdf, revision C, page 1]", chunks)
    assert not faithfulness_check("130 deg C [manual.pdf, revision C, page 1]", chunks)
    assert not faithfulness_check("120 deg C [other.pdf, revision C, page 1]", chunks)