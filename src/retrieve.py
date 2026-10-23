"""Retrieve evidence with vector, BM25, hybrid, and optional reranker modes."""

from __future__ import annotations

import argparse
import math
import pickle
import re
from pathlib import Path
from typing import Any

from src.chunk import Chunk
from src.index import tokenize, validate_chunks


_REVISION_PART_RE = re.compile(r"[A-Za-z]+|\d+")


def _revision_key(revision: str) -> tuple[tuple[int, int], ...]:
    """Build a natural-order key for alphabetic and numeric revision labels."""
    parts: list[tuple[int, int]] = []
    for part in _REVISION_PART_RE.findall(revision):
        if part.isdigit():
            parts.append((1, int(part)))
        else:
            alpha_value = 0
            for character in part.casefold():
                alpha_value = alpha_value * 26 + ord(character) - ord("a") + 1
            parts.append((0, alpha_value))
    return tuple(parts)


def latest_revisions(chunks: list[Chunk]) -> dict[str, str]:
    """Return the highest observed revision for each document number."""
    latest: dict[str, str] = {}
    for chunk in chunks:
        metadata = chunk["metadata"]
        doc_number = str(metadata.get("doc_number", "unknown"))
        revision = str(metadata.get("revision", "unknown"))
        current = latest.get(doc_number)
        if current is None or _revision_key(revision) > _revision_key(current):
            latest[doc_number] = revision
    return latest


def filter_chunks(
    chunks: list[Chunk], revision: str | None = None, latest_only: bool = False
) -> list[Chunk]:
    """Filter chunks by an exact revision and/or each document's highest revision."""
    latest = latest_revisions(chunks) if latest_only else {}
    filtered: list[Chunk] = []
    for chunk in chunks:
        metadata = chunk["metadata"]
        chunk_revision = str(metadata.get("revision", "unknown"))
        doc_number = str(metadata.get("doc_number", "unknown"))
        if revision is not None and chunk_revision.casefold() != revision.casefold():
            continue
        if latest_only and latest.get(doc_number) != chunk_revision:
            continue
        filtered.append(chunk)
    return filtered


def build_chroma_where(
    chunks: list[Chunk], revision: str | None = None, latest_only: bool = False
) -> dict[str, Any] | None:
    """Create a Chroma metadata predicate equivalent to `filter_chunks`."""
    conditions: list[dict[str, Any]] = []
    if revision is not None:
        matching_revisions = sorted({
            str(chunk["metadata"].get("revision", "unknown"))
            for chunk in chunks
            if str(chunk["metadata"].get("revision", "unknown")).casefold() == revision.casefold()
        })
        if not matching_revisions:
            return {"doc_number": {"$eq": "__no_matching_document__"}}
        conditions.append({"revision": {"$in": matching_revisions}})
    if latest_only:
        latest = latest_revisions(chunks)
        latest_conditions = [
            {"$and": [
                {"doc_number": {"$eq": doc_number}},
                {"revision": {"$eq": latest_revision}},
            ]}
            for doc_number, latest_revision in latest.items()
        ]
        if not latest_conditions:
            return {"doc_number": {"$eq": "__no_matching_document__"}}
        conditions.append({"$or": latest_conditions})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]], rrf_k: int = 60, limit: int | None = None
) -> list[dict[str, Any]]:
    """Fuse ranked result lists using reciprocal rank and preserve source scores."""
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    fused: dict[str, dict[str, Any]] = {}
    for list_index, ranked_list in enumerate(ranked_lists):
        source = str(ranked_list[0].get("source", list_index)) if ranked_list else str(list_index)
        for rank, result in enumerate(ranked_list, start=1):
            chunk_id = result["id"]
            entry = fused.setdefault(chunk_id, {**result, "score": 0.0, "scores": {}})
            entry["score"] += 1.0 / (rrf_k + rank)
            entry["scores"][source] = result["score"]
    ranked = sorted(fused.values(), key=lambda result: result["score"], reverse=True)
    maximum_score = len(ranked_lists) / (rrf_k + 1)
    for result in ranked:
        result["abstention_score"] = min(result["score"] / maximum_score, 1.0)
    return ranked[:limit] if limit is not None else ranked


def vector_distance_score(distance: float) -> float:
    """Convert squared L2 distance between normalized vectors to cosine similarity."""
    return max(0.0, min(1.0, 1.0 - float(distance) / 2.0))


class Retriever:
    """Load local indexes and retrieve metadata-filtered evidence chunks."""

    def __init__(
        self,
        bm25_file: Path,
        index_dir: Path,
        collection_name: str,
        embedding_model: str,
        reranker_model: str,
    ) -> None:
        """Open the BM25 artifact and Chroma collection; load the encoder on demand."""
        with bm25_file.open("rb") as source:
            artifact = pickle.load(source)
        self.chunks = validate_chunks(artifact["chunks"])
        self.bm25 = artifact["bm25"]
        if artifact.get("chunk_ids") != [chunk["id"] for chunk in self.chunks]:
            raise ValueError("BM25 artifact chunk IDs do not match its chunk records")

        import chromadb

        client = chromadb.PersistentClient(path=str(index_dir))
        self.collection = client.get_collection(name=collection_name)
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self._embedding_model: Any | None = None
        self._reranker: Any | None = None
        self._chunk_by_id = {chunk["id"]: chunk for chunk in self.chunks}

    def _get_embedding_model(self) -> Any:
        """Load the configured sentence-transformer encoder only when needed."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def _search_bm25(
        self, query: str, top_k: int, revision: str | None, latest_only: bool
    ) -> list[dict[str, Any]]:
        """Score only chunks allowed by the requested metadata filters."""
        allowed = {chunk["id"] for chunk in filter_chunks(self.chunks, revision, latest_only)}
        scores = self.bm25.get_scores(tokenize(query))
        candidates = [
            (chunk, float(scores[index]))
            for index, chunk in enumerate(self.chunks)
            if chunk["id"] in allowed
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)
        return [
            {
                **chunk,
                "score": score,
                "abstention_score": max(score, 0.0) / (1.0 + max(score, 0.0)),
                "scores": {"bm25": score},
                "source": "bm25",
            }
            for chunk, score in candidates[:top_k]
        ]

    def _search_vector(
        self, query: str, top_k: int, revision: str | None, latest_only: bool
    ) -> list[dict[str, Any]]:
        """Query Chroma with metadata predicates and convert distance to similarity."""
        where = build_chroma_where(self.chunks, revision, latest_only)
        count = self.collection.count()
        if count == 0:
            return []
        query_embedding = self._get_embedding_model().encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()
        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        results: list[dict[str, Any]] = []
        for chunk_id, text, metadata, distance in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        ):
            score = vector_distance_score(distance)
            results.append({
                "id": chunk_id,
                "text": text,
                "metadata": metadata,
                "score": score,
                "abstention_score": score,
                "scores": {"vector_similarity": score},
                "source": "vector",
            })
        return results

    def _rerank(self, query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score retrieved query/chunk pairs with the configured cross-encoder."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_model_name)
        pairs = [(query, result["text"]) for result in results]
        scores = self._reranker.predict(pairs)
        for result, score in zip(results, scores):
            score = float(score)
            result["scores"]["reranker"] = score
            result["score"] = score
            result["abstention_score"] = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, score))))
        return sorted(results, key=lambda result: result["score"], reverse=True)

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
        revision: str | None = None,
        latest_only: bool = False,
        rerank: bool = False,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """Retrieve the top evidence chunks in vector, BM25, or hybrid mode."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if mode not in {"vector", "bm25", "hybrid"}:
            raise ValueError("mode must be one of: vector, bm25, hybrid")

        if mode == "bm25":
            results = self._search_bm25(query, top_k, revision, latest_only)
        elif mode == "vector":
            results = self._search_vector(query, top_k, revision, latest_only)
        else:
            vector_results = self._search_vector(query, top_k, revision, latest_only)
            bm25_results = self._search_bm25(query, top_k, revision, latest_only)
            results = reciprocal_rank_fusion(
                [vector_results, bm25_results], rrf_k=rrf_k, limit=top_k
            )

        if rerank and results:
            results = self._rerank(query, results)
        return results


def main() -> None:
    """Run retrieval from the command line using config.yaml defaults."""
    parser = argparse.ArgumentParser(description="Retrieve evidence from Mini-Crosscheck indexes.")
    parser.add_argument("query", help="Question or search query")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--mode", choices=("vector", "bm25", "hybrid"))
    parser.add_argument("--revision")
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    retrieval_config = config["retrieval"]
    paths = config["paths"]
    retriever = Retriever(
        bm25_file=Path(paths["bm25_file"]),
        index_dir=Path(paths["index_dir"]),
        collection_name=config["vector_store"]["collection_name"],
        embedding_model=config["embedding"]["model_name"],
        reranker_model=retrieval_config["reranker_model"],
    )
    results = retriever.retrieve(
        args.query,
        mode=args.mode or retrieval_config["mode"],
        top_k=retrieval_config["top_k"],
        revision=args.revision,
        latest_only=args.latest_only,
        rerank=args.rerank or retrieval_config["reranker_enabled"],
        rrf_k=retrieval_config["rrf_k"],
    )
    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(
            f"{rank}. {metadata.get('doc_name', 'unknown')} "
            f"Rev {metadata.get('revision', 'unknown')} p.{metadata.get('page', '?')} "
            f"score={result['score']:.4f}"
        )
        print(result["text"])


if __name__ == "__main__":
    main()