"""Evaluate source retrieval, abstention, and numeric citation faithfulness."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from src.generate import INSUFFICIENT_EVIDENCE, create_backend, generate_answer
from src.querylog import append_query_log
from src.retrieve import Retriever


_CITATION_RE = re.compile(
    r"\[([^,\]]+),\s*revision\s+([^,\]]+),\s*page\s+(\d+)\]",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?")


def load_questions(
    sample_file: Path, proposed_file: Path | None = None, include_proposed: bool = True
) -> list[dict[str, Any]]:
    """Load supplied questions and optionally append the reviewable proposals."""
    questions = json.loads(sample_file.read_text(encoding="utf-8"))
    if proposed_file is not None and include_proposed:
        questions.extend(json.loads(proposed_file.read_text(encoding="utf-8")))
    return questions


def expected_sources(question: dict[str, Any]) -> set[str]:
    """Normalize single-source and multi-source question labels."""
    sources = question.get("sources")
    if sources is None:
        source = question.get("source")
        sources = [source] if source else []
    return {str(source) for source in sources if source}


def _ranked_source_files(results: list[dict[str, Any]]) -> list[str]:
    """Deduplicate retrieved source files while retaining first-rank order."""
    ranked: list[str] = []
    for result in results:
        source = result.get("metadata", {}).get("doc_name")
        if source and source not in ranked:
            ranked.append(str(source))
    return ranked


def source_retrieval_metrics(
    relevant_sources: set[str], results: list[dict[str, Any]], k: int
) -> dict[str, float]:
    """Compute source-file-level Recall@k, MRR, and binary nDCG."""
    if not relevant_sources:
        return {"recall_at_k": 0.0, "mrr": 0.0, "ndcg": 0.0}
    ranked_sources = _ranked_source_files(results)[:k]
    relevant_ranks = [
        rank for rank, source in enumerate(ranked_sources, start=1) if source in relevant_sources
    ]
    recall = len(relevant_ranks) / len(relevant_sources)
    mrr = 1.0 / min(relevant_ranks) if relevant_ranks else 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_hits = min(len(relevant_sources), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {"recall_at_k": recall, "mrr": mrr, "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0}


def faithfulness_check(answer: str, retrieved_chunks: list[dict[str, Any]]) -> bool:
    """Check citation provenance and that numeric answer values occur in cited chunks."""
    if answer.strip() == INSUFFICIENT_EVIDENCE:
        return True
    citations = list(_CITATION_RE.finditer(answer))
    if not citations:
        return False

    cited_chunks: list[dict[str, Any]] = []
    for citation in citations:
        doc_name, revision, page = (part.strip() for part in citation.groups())
        match = next((
            chunk for chunk in retrieved_chunks
            if str(chunk.get("metadata", {}).get("doc_name", "")).casefold() == doc_name.casefold()
            and str(chunk.get("metadata", {}).get("revision", "")).casefold() == revision.casefold()
            and str(chunk.get("metadata", {}).get("page", "")).casefold() == page.casefold()
        ), None)
        if match is None:
            return False
        cited_chunks.append(match)

    answer_without_citations = _CITATION_RE.sub("", answer)
    numeric_values = set(_NUMBER_RE.findall(answer_without_citations))
    for value in numeric_values:
        value_pattern = re.compile(rf"(?<![\d.]){re.escape(value)}(?![\d.])")
        if not any(value_pattern.search(str(chunk.get("text", ""))) for chunk in cited_chunks):
            return False
    return True


def _average(values: list[float]) -> float:
    """Return a stable zero for empty metric collections."""
    return sum(values) / len(values) if values else 0.0


def evaluate_questions(
    questions: list[dict[str, Any]],
    retriever: Retriever,
    backend: Any,
    retrieval_config: dict[str, Any],
    abstention_threshold: float,
    query_log: Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate retrieval modes and generated answers for the same question set."""
    modes = [
        ("vector", "vector", False),
        ("bm25", "bm25", False),
        ("hybrid", "hybrid", False),
        ("hybrid+rerank", "hybrid", True),
    ]
    rows: list[dict[str, Any]] = []
    top_k = int(retrieval_config["top_k"])
    for label, mode, rerank in modes:
        retrieval_scores: list[dict[str, float]] = []
        unanswerable_flags: list[bool] = []
        abstained_flags: list[bool] = []
        faithfulness_flags: list[bool] = []
        for question in questions:
            results = retriever.retrieve(
                question["q"],
                mode=mode,
                top_k=top_k,
                rerank=rerank,
                rrf_k=int(retrieval_config["rrf_k"]),
            )
            relevant = expected_sources(question)
            if relevant:
                retrieval_scores.append(source_retrieval_metrics(relevant, results, top_k))

            answer_result = generate_answer(
                question["q"], results, backend, abstention_threshold
            )
            if query_log is not None:
                append_query_log(
                    query_log,
                    question["q"],
                    {"mode": label, "top_k": top_k, "rerank": rerank},
                    results,
                    answer=answer_result["answer"],
                )
            is_unanswerable = question.get("answer", "").strip() == INSUFFICIENT_EVIDENCE
            if is_unanswerable:
                unanswerable_flags.append(answer_result["answer"] == INSUFFICIENT_EVIDENCE)
            elif not answer_result["abstained"]:
                faithfulness_flags.append(faithfulness_check(answer_result["answer"], results))

        rows.append({
            "mode": label,
            "questions": len(questions),
            "answerable_questions": len(retrieval_scores),
            "recall_at_k": _average([metric["recall_at_k"] for metric in retrieval_scores]),
            "mrr": _average([metric["mrr"] for metric in retrieval_scores]),
            "ndcg": _average([metric["ndcg"] for metric in retrieval_scores]),
            "unanswerable_questions": len(unanswerable_flags),
            "correct_abstention_rate": _average([float(value) for value in unanswerable_flags]),
            "faithfulness_rate": _average([float(value) for value in faithfulness_flags]),
            "faithfulness_checked": len(faithfulness_flags),
        })
    return rows


def save_results(rows: list[dict[str, Any]], output_file: Path) -> None:
    """Write evaluation comparisons to CSV, creating the parent directory if needed."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with output_file.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def print_results(rows: list[dict[str, Any]]) -> None:
    """Print a compact side-by-side comparison table."""
    headers = ("Mode", "Recall@k", "MRR", "nDCG", "Correct abstention", "Faithfulness")
    print(f"{headers[0]:<16} {headers[1]:>10} {headers[2]:>8} {headers[3]:>8} {headers[4]:>20} {headers[5]:>14}")
    for row in rows:
        print(
            f"{row['mode']:<16} {row['recall_at_k']:>10.3f} {row['mrr']:>8.3f} "
            f"{row['ndcg']:>8.3f} {row['correct_abstention_rate']:>20.3f} "
            f"{row['faithfulness_rate']:>14.3f}"
        )


def main() -> None:
    """Run the configured evaluation and save its comparison CSV."""
    parser = argparse.ArgumentParser(description="Evaluate Mini-Crosscheck retrieval and answers.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--limit", type=int, help="Evaluate only the first N questions")
    parser.add_argument("--exclude-proposed", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    questions = load_questions(
        Path(paths["questions_file"]),
        Path(paths["proposed_questions_file"]),
        include_proposed=(
            config["evaluation"]["include_proposed_questions"] and not args.exclude_proposed
        ),
    )
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        questions = questions[:args.limit]

    retriever = Retriever(
        bm25_file=Path(paths["bm25_file"]),
        index_dir=Path(paths["index_dir"]),
        collection_name=config["vector_store"]["collection_name"],
        embedding_model=config["embedding"]["model_name"],
        reranker_model=config["retrieval"]["reranker_model"],
    )
    backend = create_backend(config["generation"])
    rows = evaluate_questions(
        questions,
        retriever,
        backend,
        config["retrieval"],
        float(config["generation"]["abstention_threshold"]),
        query_log=Path(paths["query_log"]),
    )
    print_results(rows)
    output_file = Path(paths["eval_results"])
    save_results(rows, output_file)
    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    main()