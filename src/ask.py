"""Command-line question answering and shared pipeline service."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.generate import create_backend, generate_answer
from src.querylog import append_query_log
from src.retrieve import Retriever


class MiniCrosscheckService:
    """Coordinate retrieval, score-based abstention, generation, and query logging."""

    def __init__(
        self,
        retriever: Retriever,
        backend: Any,
        retrieval_settings: dict[str, Any],
        abstention_threshold: float,
        query_log: Path,
    ) -> None:
        """Store the shared pipeline components and run-time settings."""
        self.retriever = retriever
        self.backend = backend
        self.retrieval_settings = retrieval_settings
        self.abstention_threshold = abstention_threshold
        self.query_log = query_log

    def ask(
        self,
        question: str,
        mode: str | None = None,
        top_k: int | None = None,
        revision: str | None = None,
        latest_only: bool = False,
        rerank: bool | None = None,
    ) -> dict[str, Any]:
        """Retrieve evidence, generate or abstain, and log the complete query."""
        resolved_mode = mode or self.retrieval_settings["mode"]
        resolved_top_k = (
            int(self.retrieval_settings["top_k"]) if top_k is None else top_k
        )
        resolved_rerank = (
            bool(self.retrieval_settings["reranker_enabled"])
            if rerank is None else rerank
        )
        settings = {
            "mode": resolved_mode,
            "top_k": resolved_top_k,
            "revision": revision,
            "latest_only": latest_only,
            "rerank": resolved_rerank,
        }
        chunks: list[dict[str, Any]] = []
        answer: str | None = None
        try:
            chunks = self.retriever.retrieve(
                question,
                mode=resolved_mode,
                top_k=resolved_top_k,
                revision=revision,
                latest_only=latest_only,
                rerank=resolved_rerank,
                rrf_k=int(self.retrieval_settings["rrf_k"]),
            )
            result = generate_answer(
                question, chunks, self.backend, self.abstention_threshold
            )
            answer = result["answer"]
            return {**result, "chunks": chunks}
        except Exception as error:
            append_query_log(
                self.query_log,
                question,
                settings,
                chunks,
                answer=answer,
                error=type(error).__name__,
            )
            raise
        finally:
            if answer is not None:
                append_query_log(self.query_log, question, settings, chunks, answer=answer)


def create_service(config_file: Path = Path("config.yaml")) -> MiniCrosscheckService:
    """Load configured local indexes and the selected generation backend."""
    import yaml

    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    paths = config["paths"]
    retrieval_settings = config["retrieval"]
    retriever = Retriever(
        bm25_file=Path(paths["bm25_file"]),
        index_dir=Path(paths["index_dir"]),
        collection_name=config["vector_store"]["collection_name"],
        embedding_model=config["embedding"]["model_name"],
        reranker_model=retrieval_settings["reranker_model"],
    )
    return MiniCrosscheckService(
        retriever=retriever,
        backend=create_backend(config["generation"]),
        retrieval_settings=retrieval_settings,
        abstention_threshold=float(config["generation"]["abstention_threshold"]),
        query_log=Path(paths["query_log"]),
    )


def main() -> None:
    """Answer one question from the command line."""
    parser = argparse.ArgumentParser(description="Ask Mini-Crosscheck about engineering documents.")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--mode", choices=("vector", "bm25", "hybrid"))
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--revision")
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    service = create_service(args.config)
    result = service.ask(
        args.question,
        mode=args.mode,
        top_k=args.top_k,
        revision=args.revision,
        latest_only=args.latest_only,
        rerank=True if args.rerank else None,
    )
    print(f"Answer: {result['answer']}")
    for index, chunk in enumerate(result["chunks"], start=1):
        metadata = chunk["metadata"]
        print(
            f"\n{index}. {metadata.get('doc_name', 'unknown')} "
            f"[revision {metadata.get('revision', 'unknown')}, page {metadata.get('page', '?')}] "
            f"score={chunk.get('score', 0.0):.4f}"
        )
        print(chunk["text"])
    if result["revision_variants"]:
        print("\nRevision warning: multiple revisions of the same document were retrieved.")


if __name__ == "__main__":
    main()