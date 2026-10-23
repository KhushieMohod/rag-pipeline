"""Split parsed PDF pages into metadata-preserving text and table chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypedDict


class Chunk(TypedDict):
    """A searchable content chunk with its source metadata."""

    id: str
    text: str
    metadata: dict[str, str | int]


_METADATA_FIELDS = ("doc_name", "doc_number", "revision", "date", "page")
_SEPARATORS = ("\n\n", "\n", ". ", " ")


def _split_recursively(text: str, chunk_size: int, separators: tuple[str, ...]) -> list[str]:
    """Prefer paragraph, line, sentence, and word boundaries before hard splitting."""
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        return [text[start:start + chunk_size] for start in range(0, len(text), chunk_size)]

    separator = separators[0]
    if separator not in text:
        return _split_recursively(text, chunk_size, separators[1:])

    sections = text.split(separator)
    split_sections: list[str] = []
    for index, section in enumerate(sections):
        if index < len(sections) - 1:
            section += separator
        split_sections.extend(_split_recursively(section, chunk_size, separators[1:]))
    return split_sections


def _overlap_suffix(text: str, overlap: int) -> str:
    """Return a bounded overlap suffix without starting in the middle of a word."""
    if overlap == 0:
        return ""
    suffix = text[-overlap:]
    boundary = next((index for index, character in enumerate(suffix) if character.isspace()), None)
    return suffix[boundary + 1:].lstrip() if boundary is not None else suffix


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursively split prose into bounded chunks with character overlap."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")

    pieces = [piece.strip() for piece in _split_recursively(text.strip(), chunk_size, _SEPARATORS)]
    pieces = [piece for piece in pieces if piece]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + len(piece) + 1 <= chunk_size:
            current = f"{current} {piece}"
        else:
            chunks.append(current)
            suffix = _overlap_suffix(current, overlap)
            available = max(0, chunk_size - len(piece) - 1)
            suffix = suffix[-available:] if available else ""
            current = f"{suffix} {piece}".strip() if suffix else piece
    if current:
        chunks.append(current)
    return chunks


def _group_table_rows(table_text: str, chunk_size: int) -> list[str]:
    """Group complete labeled table rows, splitting only an individually oversized row."""
    groups: list[str] = []
    current_rows: list[str] = []
    current_length = 0
    for row in (line.strip() for line in table_text.splitlines()):
        if not row:
            continue
        if len(row) > chunk_size:
            if current_rows:
                groups.append("\n".join(current_rows))
                current_rows = []
                current_length = 0
            groups.extend(split_text(row, chunk_size, 0))
            continue
        added_length = len(row) + (1 if current_rows else 0)
        if current_rows and current_length + added_length > chunk_size:
            groups.append("\n".join(current_rows))
            current_rows = []
            current_length = 0
            added_length = len(row)
        current_rows.append(row)
        current_length += added_length
    if current_rows:
        groups.append("\n".join(current_rows))
    return groups


def chunk_page_record(page_record: dict[str, Any], chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    """Chunk a single parsed page while carrying its document and page metadata."""
    metadata = page_record["metadata"]
    base_metadata = {field: metadata[field] for field in _METADATA_FIELDS if field in metadata}
    source_id = (
        f"{base_metadata.get('doc_name', 'unknown')}"
        f"::r{base_metadata.get('revision', 'unknown')}"
        f"::p{base_metadata.get('page', 0)}"
    )
    chunks: list[Chunk] = []

    for index, text in enumerate(split_text(page_record.get("text", ""), chunk_size, overlap), start=1):
        chunks.append({
            "id": f"{source_id}::text::{index}",
            "text": text,
            "metadata": {**base_metadata, "chunk_type": "text"},
        })

    for table in page_record.get("tables", []):
        table_index = table.get("table_index", 1)
        for group_index, text in enumerate(_group_table_rows(table.get("text", ""), chunk_size), start=1):
            chunks.append({
                "id": f"{source_id}::table::{table_index}::{group_index}",
                "text": text,
                "metadata": {**base_metadata, "chunk_type": "table", "table_index": table_index},
            })
    return chunks


def chunk_jsonl(input_file: Path, output_file: Path, chunk_size: int, overlap: int) -> dict[str, int | float]:
    """Chunk page records from JSONL and return printable corpus statistics."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[Chunk] = []
    with input_file.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                chunks.extend(chunk_page_record(json.loads(line), chunk_size, overlap))

    with output_file.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    text_count = sum(chunk["metadata"]["chunk_type"] == "text" for chunk in chunks)
    table_count = len(chunks) - text_count
    return {
        "chunks": len(chunks),
        "text_chunks": text_count,
        "table_chunks": table_count,
        "average_characters": round(sum(len(chunk["text"]) for chunk in chunks) / len(chunks), 1)
        if chunks else 0.0,
        "largest_chunk_characters": max((len(chunk["text"]) for chunk in chunks), default=0),
    }


def main() -> None:
    """Run chunking from the command line using config.yaml defaults."""
    parser = argparse.ArgumentParser(description="Chunk parsed pages into searchable JSONL records.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    input_file = args.input or Path(config["paths"]["parsed_file"])
    output_file = args.output or Path(config["paths"]["chunks_file"])
    chunk_size = args.chunk_size if args.chunk_size is not None else config["chunking"]["chunk_size"]
    overlap = args.overlap if args.overlap is not None else config["chunking"]["overlap"]
    stats = chunk_jsonl(input_file, output_file, chunk_size, overlap)
    print(f"Wrote {stats['chunks']} chunks to {output_file}")
    print(
        f"Text: {stats['text_chunks']} | Tables: {stats['table_chunks']} | "
        f"Average chars: {stats['average_characters']} | "
        f"Largest chunk: {stats['largest_chunk_characters']} chars"
    )


if __name__ == "__main__":
    main()