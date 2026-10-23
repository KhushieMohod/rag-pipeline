"""Tests for Stage 3 chunk validation and BM25 token preparation."""

import pytest

from src.index import tokenize, validate_chunks


def test_tokenize_normalizes_words_and_numeric_values() -> None:
    """Tokenization should preserve useful engineering terms and values."""
    assert tokenize("Yield strength: 262 MPa at 100 deg C") == [
        "yield", "strength", "262", "mpa", "at", "100", "deg", "c"
    ]


def test_validate_chunks_preserves_records_in_order() -> None:
    """The index input should retain stable IDs, text, and metadata order."""
    chunks = [
        {"id": "manual::1", "text": "Maximum temperature is 120 deg C.", "metadata": {"revision": "C", "page": 1}},
        {"id": "manual::2", "text": "Grease interval is 3000 hours.", "metadata": {"revision": "C", "page": 2}},
    ]

    assert validate_chunks(chunks) == chunks


def test_validate_chunks_rejects_duplicate_ids() -> None:
    """Repeated IDs would make the vector and lexical stores disagree."""
    chunk = {"id": "duplicate", "text": "Some evidence.", "metadata": {"page": 1}}

    with pytest.raises(ValueError, match="Duplicate chunk id"):
        validate_chunks([chunk, chunk])


def test_validate_chunks_rejects_empty_corpus() -> None:
    """An empty input should not silently create empty indexes."""
    with pytest.raises(ValueError, match="No chunks found"):
        validate_chunks([])