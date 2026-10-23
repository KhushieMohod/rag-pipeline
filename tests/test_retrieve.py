"""Tests for Stage 4 metadata filtering and reciprocal-rank fusion."""

import pytest

from src.retrieve import (
    build_chroma_where,
    filter_chunks,
    latest_revisions,
    reciprocal_rank_fusion,
    vector_distance_score,
)


@pytest.fixture
def revision_chunks() -> list[dict]:
    """Chunks from two documents with different revision histories."""
    return [
        {"id": "manual-b", "text": "90 deg C", "metadata": {"doc_number": "MAN-1", "revision": "B"}},
        {"id": "manual-c", "text": "120 deg C", "metadata": {"doc_number": "MAN-1", "revision": "C"}},
        {"id": "procedure-a", "text": "90 deg C", "metadata": {"doc_number": "PROC-1", "revision": "A"}},
        {"id": "procedure-c", "text": "120 deg C", "metadata": {"doc_number": "PROC-1", "revision": "C"}},
    ]


def test_latest_revision_is_selected_per_document(revision_chunks: list[dict]) -> None:
    """Latest-only should retain each document number's highest revision."""
    assert latest_revisions(revision_chunks) == {"MAN-1": "C", "PROC-1": "C"}
    assert [chunk["id"] for chunk in filter_chunks(revision_chunks, latest_only=True)] == [
        "manual-c", "procedure-c"
    ]


def test_exact_revision_filter(revision_chunks: list[dict]) -> None:
    """An explicit revision filter should match exactly, case-insensitively."""
    assert [chunk["id"] for chunk in filter_chunks(revision_chunks, revision="b")] == ["manual-b"]


def test_chroma_filter_combines_revision_and_latest(revision_chunks: list[dict]) -> None:
    """Chroma predicates should express both requested metadata constraints."""
    where = build_chroma_where(revision_chunks, revision="c", latest_only=True)

    assert "$and" in where
    assert {"revision": {"$in": ["C"]}} in where["$and"]
    assert {"$or"} == {next(iter(where["$and"][1]))}


def test_reciprocal_rank_fusion_combines_ranks() -> None:
    """A chunk ranked highly by both retrieval methods should gain fused score."""
    fused = reciprocal_rank_fusion(
        [
            [{"id": "shared", "text": "A", "score": 0.9}, {"id": "vector-only", "text": "B", "score": 0.7}],
            [{"id": "shared", "text": "A", "score": 5.0}, {"id": "bm25-only", "text": "C", "score": 3.0}],
        ],
        rrf_k=60,
    )

    assert fused[0]["id"] == "shared"
    assert fused[0]["score"] == pytest.approx(2 / 61)
    assert fused[0]["scores"] == {"0": 0.9, "1": 5.0}


def test_reciprocal_rank_fusion_rejects_invalid_constant() -> None:
    """The RRF constant must remain positive."""
    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fusion([], rrf_k=0)


def test_vector_distance_maps_normalized_l2_to_cosine_score() -> None:
    """Squared L2 distances for unit vectors should map to bounded cosine scores."""
    assert vector_distance_score(0.0) == 1.0
    assert vector_distance_score(1.0) == 0.5
    assert vector_distance_score(2.0) == 0.0