"""Generate source-grounded answers through local or OpenAI-compatible chat APIs."""

from __future__ import annotations

import os
from typing import Any, Mapping, Protocol, TypedDict


INSUFFICIENT_EVIDENCE = "INSUFFICIENT EVIDENCE"


class ChatBackend(Protocol):
    """Minimal interface shared by supported language-model backends."""

    def complete(self, prompt: str) -> str:
        """Return a single response for the supplied prompt."""


class AnswerResult(TypedDict):
    """Generated answer and retrieval confidence metadata."""

    answer: str
    abstained: bool
    score: float
    revision_variants: list[dict[str, Any]]


class OllamaBackend:
    """Chat with a model served by Ollama's local HTTP API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float,
        temperature: float,
        max_tokens: int,
    ) -> None:
        """Store request settings without opening a connection."""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        """Send a non-streaming chat request to Ollama."""
        import requests

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return str(response.json()["message"]["content"])


class OpenAICompatibleBackend:
    """Chat with any server exposing the OpenAI chat-completions API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout: float,
        temperature: float,
        max_tokens: int,
    ) -> None:
        """Store endpoint credentials and generation parameters."""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        """Send a non-streaming chat-completions request."""
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = self.base_url
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"
        response = requests.post(
            endpoint,
            headers=headers,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])


def create_backend(
    config: Mapping[str, Any], environ: Mapping[str, str] | None = None
) -> ChatBackend:
    """Choose the configured backend, with environment-variable overrides."""
    environment = os.environ if environ is None else environ
    backend_name = environment.get(
        config["llm_backend_env"], config["default_backend"]
    ).casefold()
    timeout = float(config["request_timeout_seconds"])
    temperature = float(config["temperature"])
    max_tokens = int(config["max_tokens"])

    if backend_name == "ollama":
        settings = config["ollama"]
        return OllamaBackend(
            base_url=settings["base_url"],
            model=settings["model"],
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if backend_name in {"openai", "openai_compatible"}:
        settings = config["openai_compatible"]
        base_url = environment.get(settings["base_url_env"])
        model = environment.get(settings["model_env"], settings.get("default_model", ""))
        if not base_url:
            raise ValueError(f"Set {settings['base_url_env']} for the OpenAI-compatible backend")
        if not model:
            raise ValueError(f"Set {settings['model_env']} for the OpenAI-compatible backend")
        return OpenAICompatibleBackend(
            base_url=base_url,
            model=model,
            api_key=environment.get(settings["api_key_env"]),
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unsupported LLM backend: {backend_name}")


def revision_variants(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """List documents represented by more than one revision in retrieved evidence."""
    revisions_by_document: dict[str, set[str]] = {}
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        doc_number = str(metadata.get("doc_number", "unknown"))
        revision = str(metadata.get("revision", "unknown"))
        revisions_by_document.setdefault(doc_number, set()).add(revision)
    return [
        {"doc_number": doc_number, "revisions": sorted(revisions)}
        for doc_number, revisions in sorted(revisions_by_document.items())
        if len(revisions) > 1
    ]


def build_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Construct an evidence-only prompt with citation and revision rules."""
    evidence_blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        doc_name = metadata.get("doc_name", "unknown")
        revision = metadata.get("revision", "unknown")
        page = metadata.get("page", "unknown")
        evidence_blocks.append(
            f"SOURCE {index} [{doc_name}, revision {revision}, page {page}]\n"
            f"{chunk.get('text', '')}"
        )
    evidence = "\n\n".join(evidence_blocks)
    return f"""You are Mini-Crosscheck, an engineering-document question answering system.
Answer using only the evidence below. Do not use outside knowledge or infer unsupported values.
If the evidence does not support an answer, respond exactly: {INSUFFICIENT_EVIDENCE}
Cite every factual claim using [doc_name, revision X, page N]. Mention the revision used in the answer.
When evidence includes different revisions of the same document, compare the relevant values.
If those values disagree, explicitly report both values with their revision-specific citations; do not silently choose one.
Do not call different revisions a conflict unless the evidence actually disagrees.
Keep the answer direct and include no uncited factual claims.

Question:
{question}

Retrieved evidence:
{evidence}
"""


def build_fallback_answer(chunks: list[dict[str, Any]]) -> str:
    """Return a grounded answer when the configured model backend is unavailable."""
    best_chunk = max(
        chunks,
        key=lambda chunk: float(chunk.get("abstention_score", chunk.get("score", 0.0))),
        default=None,
    )
    if best_chunk is None:
        return INSUFFICIENT_EVIDENCE

    metadata = best_chunk.get("metadata", {})
    doc_name = metadata.get("doc_name", "unknown")
    revision = metadata.get("revision", "unknown")
    page = metadata.get("page", "unknown")
    text = str(best_chunk.get("text", "")).strip()
    if not text:
        return f"The strongest retrieved evidence does not include a usable text snippet [{doc_name}, revision {revision}, page {page}]"
    return f'The strongest retrieved evidence indicates: "{text}" [{doc_name}, revision {revision}, page {page}]'


def generate_answer(
    question: str,
    chunks: list[dict[str, Any]],
    backend: ChatBackend,
    abstention_threshold: float,
) -> AnswerResult:
    """Abstain below the retrieval score threshold, otherwise generate from retrieved evidence."""
    if not 0.0 <= abstention_threshold <= 1.0:
        raise ValueError("abstention_threshold must be between 0 and 1")
    score = max(
        (float(chunk.get("abstention_score", chunk.get("score", 0.0))) for chunk in chunks),
        default=0.0,
    )
    variants = revision_variants(chunks)
    if not chunks or score < abstention_threshold:
        return {
            "answer": INSUFFICIENT_EVIDENCE,
            "abstained": True,
            "score": score,
            "revision_variants": variants,
        }

    try:
        answer = backend.complete(build_prompt(question, chunks)).strip()
    except Exception:
        answer = build_fallback_answer(chunks)
    if not answer:
        answer = INSUFFICIENT_EVIDENCE
    return {
        "answer": answer,
        "abstained": answer == INSUFFICIENT_EVIDENCE,
        "score": score,
        "revision_variants": variants,
    }