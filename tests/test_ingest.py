"""Tests for Stage 1 PDF metadata and table formatting."""

from src.ingest import extract_metadata, table_to_text


def test_extract_metadata_prefers_header_values() -> None:
    """Header metadata should take precedence over filename fallbacks."""
    metadata = extract_metadata(
        "Document No: MAN-HX200-001 Revision: C Date: 2025-06-02",
        "HX200_Pump_Manual_RevB.pdf",
        2,
    )

    assert metadata == {
        "doc_name": "HX200_Pump_Manual_RevB.pdf",
        "doc_number": "MAN-HX200-001",
        "revision": "C",
        "date": "2025-06-02",
        "page": 2,
    }


def test_extract_metadata_falls_back_to_filename_revision() -> None:
    """A revision token in the filename should be used when the header is absent."""
    metadata = extract_metadata("No document metadata", "MP17_Seal_Replacement_RevA.pdf", 1)

    assert metadata["doc_number"] == "unknown"
    assert metadata["revision"] == "A"
    assert metadata["date"] == "unknown"


def test_table_to_text_repeats_column_labels_for_each_row() -> None:
    """Every cell value should retain its table column label and unit context."""
    result = table_to_text(
        [
            ["Temperature (deg C)", "Yield strength (MPa)"],
            ["100", "262"],
            ["120", "245"],
        ]
    )

    assert result == (
        "Temperature (deg C): 100 | Yield strength (MPa): 262\n"
        "Temperature (deg C): 120 | Yield strength (MPa): 245"
    )