"""Tests for Stage 2 recursive prose and table chunking."""

import pytest

from src.chunk import chunk_page_record, split_text


def test_split_text_respects_size_and_overlaps_chunks() -> None:
    """Prose chunks should fit the limit and retain overlap between neighbors."""
    chunks = split_text("Alpha beta gamma delta epsilon zeta eta theta.", chunk_size=18, overlap=5)

    assert len(chunks) > 1
    assert all(len(chunk) <= 18 for chunk in chunks)
    assert any(set(left.split()) & set(right.split()) for left, right in zip(chunks, chunks[1:]))


def test_chunk_page_record_preserves_metadata_and_groups_table_rows() -> None:
    """Text and table chunks should carry source metadata and keep table rows intact."""
    record = {
        "metadata": {
            "doc_name": "materials.pdf",
            "doc_number": "MAT-1",
            "revision": "B",
            "date": "2024-01-01",
            "page": 3,
        },
        "text": "A short paragraph about the component.",
        "tables": [{
            "table_index": 1,
            "text": "Temperature: 100 | Yield: 262\nTemperature: 120 | Yield: 245",
        }],
    }

    chunks = chunk_page_record(record, chunk_size=40, overlap=5)
    text_chunk = next(chunk for chunk in chunks if chunk["metadata"]["chunk_type"] == "text")
    table_chunks = [chunk for chunk in chunks if chunk["metadata"]["chunk_type"] == "table"]

    assert text_chunk["metadata"]["doc_number"] == "MAT-1"
    assert text_chunk["metadata"]["page"] == 3
    assert text_chunk["metadata"]["revision"] == "B"
    assert len(table_chunks) == 2
    assert "Temperature: 100" in table_chunks[0]["text"]
    assert "Temperature: 120" in table_chunks[1]["text"]


def test_split_text_rejects_invalid_overlap() -> None:
    """Invalid chunk settings should fail instead of silently changing behavior."""
    with pytest.raises(ValueError, match="overlap"):
        split_text("Some prose", chunk_size=10, overlap=10)