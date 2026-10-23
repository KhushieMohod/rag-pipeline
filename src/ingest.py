"""Parse PDF pages and tables into a page-oriented JSONL file."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, TypedDict


class PageMetadata(TypedDict):
    """Metadata associated with a single source PDF page."""

    doc_name: str
    doc_number: str
    revision: str
    date: str
    page: int


_HEADER_RE = re.compile(
    r"Document\s+No:\s*(?P<doc_number>\S+)\s+"
    r"Revision:\s*(?P<revision>\S+)\s+"
    r"Date:\s*(?P<date>\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)
_FILENAME_REVISION_RE = re.compile(r"(?:^|[_\s-])Rev([A-Za-z0-9]+)(?:$|[_\s.])", re.IGNORECASE)


def extract_metadata(header_text: str, filename: str, page_number: int) -> PageMetadata:
    """Extract page metadata from document text, falling back to the filename revision."""
    header_match = _HEADER_RE.search(header_text)
    filename_match = _FILENAME_REVISION_RE.search(Path(filename).stem)
    return {
        "doc_name": Path(filename).name,
        "doc_number": header_match.group("doc_number") if header_match else "unknown",
        "revision": (
            header_match.group("revision")
            if header_match
            else filename_match.group(1) if filename_match else "unknown"
        ),
        "date": header_match.group("date") if header_match else "unknown",
        "page": page_number,
    }


def _clean_cell(value: str | None) -> str:
    """Normalize a table cell to one line of readable text."""
    return " ".join((value or "").split())


def table_to_text(table: Iterable[Iterable[str | None]]) -> str:
    """Render table rows as labeled values so each value retains its column context."""
    rows = [[_clean_cell(cell) for cell in row] for row in table]
    rows = [row for row in rows if any(row)]
    if not rows:
        return ""

    headers = [value or f"Column {index + 1}" for index, value in enumerate(rows[0])]
    rendered_rows: list[str] = []
    for row in rows[1:]:
        if all(index < len(row) and row[index].casefold() == headers[index].casefold()
               for index in range(min(len(headers), len(row)))):
            continue
        labeled_values = [
            f"{headers[index] if index < len(headers) else f'Column {index + 1}'}: {value}"
            for index, value in enumerate(row)
            if value
        ]
        if labeled_values:
            rendered_rows.append(" | ".join(labeled_values))
    return "\n".join(rendered_rows)


def _point_is_in_table(character: dict[str, Any], boxes: list[tuple[float, float, float, float]]) -> bool:
    """Return whether a character's center falls inside any extracted table."""
    center_x = (character["x0"] + character["x1"]) / 2
    center_y = (character["top"] + character["bottom"]) / 2
    return any(x0 <= center_x <= x1 and top <= center_y <= bottom for x0, top, x1, bottom in boxes)


def _extract_page(page: Any, metadata: PageMetadata) -> dict[str, Any]:
    """Extract prose and tables from one pdfplumber page."""
    found_tables = page.find_tables()
    table_boxes = [table.bbox for table in found_tables]
    tables = [
        {"table_index": index, "text": table_to_text(table.extract())}
        for index, table in enumerate(found_tables, start=1)
    ]
    prose_page = page.filter(
        lambda item: item.get("object_type") != "char"
        or not _point_is_in_table(item, table_boxes)
    )
    return {
        "metadata": metadata,
        "text": prose_page.extract_text() or "",
        "tables": [table for table in tables if table["text"]],
    }


def parse_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    """Parse a PDF page by page using pdfplumber."""
    import pdfplumber

    records: list[dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            metadata = extract_metadata(page_text, pdf_path.name, page_number)
            records.append(_extract_page(page, metadata))
    return records


def parse_directory(input_dir: Path, output_file: Path, file_glob: str = "*.pdf") -> int:
    """Parse matching PDFs and write one JSON object per page to a JSONL file."""
    pdf_paths = sorted(input_dir.glob(file_glob))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    record_count = 0
    with output_file.open("w", encoding="utf-8") as output:
        for pdf_path in pdf_paths:
            for record in parse_pdf(pdf_path):
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                record_count += 1
    return record_count


def main() -> None:
    """Run PDF ingestion from the command line."""
    parser = argparse.ArgumentParser(description="Parse engineering PDFs into page-level JSONL records.")
    parser.add_argument("--input", type=Path, default=Path("data"), help="Directory containing PDFs")
    parser.add_argument("--output", type=Path, default=Path("outputs/parsed.jsonl"), help="Output JSONL path")
    parser.add_argument("--glob", default="*.pdf", help="PDF filename pattern")
    args = parser.parse_args()
    record_count = parse_directory(args.input, args.output, args.glob)
    print(f"Parsed {record_count} pages into {args.output}")


if __name__ == "__main__":
    main()