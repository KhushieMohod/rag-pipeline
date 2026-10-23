"""Build local BM25 and Chroma indexes from the same chunk JSONL corpus."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any, cast

from src.chunk import Chunk


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Normalize text into lowercase word and number tokens for BM25."""
    return _TOKEN_RE.findall(text.casefold())


def validate_chunks(chunks: list[dict[str, Any]]) -> list[Chunk]:
    """Validate chunk records and reject duplicate IDs before indexing."""
    validated: list[Chunk] = []
    seen_ids: set[str] = set()
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk {index} must be a JSON object")
        chunk_id = chunk.get("id")
        text = chunk.get("text")
        metadata = chunk.get("metadata")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"Chunk {index} must have a non-empty string id")
        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk id: {chunk_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk {chunk_id} must have non-empty text")
        if not isinstance(metadata, dict):
            raise ValueError(f"Chunk {chunk_id} must have metadata")
        if not all(isinstance(value, (str, int, float, bool)) for value in metadata.values()):
            raise ValueError(f"Chunk {chunk_id} metadata values must be scalar")
        validated.append({"id": chunk_id, "text": text, "metadata": metadata})
        seen_ids.add(chunk_id)
    if not validated:
        raise ValueError("No chunks found; run Stage 2 before indexing")
    return cast(list[Chunk], validated)


def load_chunks(chunks_file: Path) -> list[Chunk]:
    """Load and validate all chunks from a JSONL file."""
    records: list[dict[str, Any]] = []
    with chunks_file.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number}: {error.msg}") from error
    return validate_chunks(records)


def build_bm25_index(chunks: list[Chunk], output_file: Path) -> None:
    """Build and serialize BM25 alongside the exact source chunks and token lists."""
    from rank_bm25 import BM25Okapi

    tokenized_documents = [tokenize(chunk["text"]) for chunk in chunks]
    artifact = {
        "chunk_ids": [chunk["id"] for chunk in chunks],
        "chunks": chunks,
        "tokenized_documents": tokenized_documents,
        "bm25": BM25Okapi(tokenized_documents),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as output:
        pickle.dump(artifact, output, protocol=pickle.HIGHEST_PROTOCOL)


def build_vector_index(
    chunks: list[Chunk],
    index_dir: Path,
    collection_name: str,
    model_name: str,
    batch_size: int,
) -> None:
    """Embed chunks and rebuild the persistent Chroma collection deterministically."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    import chromadb
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_or_create_collection(name=collection_name)
    existing_ids = collection.get(include=["metadatas"])["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        embeddings = model.encode(
            [chunk["text"] for chunk in batch],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        collection.add(
            ids=[chunk["id"] for chunk in batch],
            embeddings=embeddings,
            documents=[chunk["text"] for chunk in batch],
            metadatas=[chunk["metadata"] for chunk in batch],
        )


def build_indexes(
    chunks_file: Path,
    bm25_file: Path,
    index_dir: Path,
    collection_name: str,
    model_name: str,
    batch_size: int,
) -> int:
    """Load one canonical chunk list and build both local retrieval indexes."""
    chunks = load_chunks(chunks_file)
    build_bm25_index(chunks, bm25_file)
    build_vector_index(chunks, index_dir, collection_name, model_name, batch_size)
    return len(chunks)


def main() -> None:
    """Build BM25 and Chroma indexes using config.yaml settings."""
    parser = argparse.ArgumentParser(description="Build BM25 and Chroma indexes from chunk JSONL.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--chunks", type=Path)
    parser.add_argument("--bm25-file", type=Path)
    parser.add_argument("--index-dir", type=Path)
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    count = build_indexes(
        chunks_file=args.chunks or Path(paths["chunks_file"]),
        bm25_file=args.bm25_file or Path(paths["bm25_file"]),
        index_dir=args.index_dir or Path(paths["index_dir"]),
        collection_name=config["vector_store"]["collection_name"],
        model_name=config["embedding"]["model_name"],
        batch_size=config["indexing"]["embedding_batch_size"],
    )
    print(f"Indexed {count} chunks in BM25 and Chroma")


if __name__ == "__main__":
    main()