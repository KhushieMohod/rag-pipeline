"""Tests for Stage 5 prompts, backend selection, and score-based abstention."""

import pytest

from src.generate import (
    INSUFFICIENT_EVIDENCE,
    OllamaBackend,
    build_prompt,
    create_backend,
    generate_answer,
    revision_variants,
)


class FakeBackend:
    """Record prompts without making a network request."""

    def __init__(self, response: str = "120 deg C [manual.pdf, revision C, page 1]") -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_abstains_before_calling_backend_when_context_is_empty() -> None:
    """No retrieved evidence must produce the exact abstention without an LLM call."""
    backend = FakeBackend()

    result = generate_answer("What is the limit?", [], backend, abstention_threshold=0.25)

    assert result["answer"] == INSUFFICIENT_EVIDENCE
    assert result["abstained"] is True
    assert backend.prompts == []


def test_abstains_below_threshold_before_calling_backend() -> None:
    """A low retrieval score must prevent any backend request."""
    backend = FakeBackend()
    chunks = [{"text": "Weak match", "metadata": {}, "score": 0.1}]

    result = generate_answer("What is the limit?", chunks, backend, abstention_threshold=0.25)

    assert result["answer"] == INSUFFICIENT_EVIDENCE
    assert backend.prompts == []


def test_prompt_requires_citations_and_revision_comparison() -> None:
    """Prompt text should carry source labels and strict evidence instructions."""
    prompt = build_prompt(
        "What is the maximum temperature?",
        [{
            "text": "Maximum process temperature: 120 deg C",
            "metadata": {"doc_name": "manual.pdf", "revision": "C", "page": 1},
        }],
    )

    assert "INSUFFICIENT EVIDENCE" in prompt
    assert "Cite every factual claim" in prompt
    assert "[manual.pdf, revision C, page 1]" in prompt
    assert "different revisions" in prompt


def test_revision_variants_are_reported_by_document() -> None:
    """Retrieved chunks from multiple revisions should be exposed for UI warnings."""
    chunks = [
        {"metadata": {"doc_number": "MAN-1", "revision": "B"}},
        {"metadata": {"doc_number": "MAN-1", "revision": "C"}},
        {"metadata": {"doc_number": "PROC-1", "revision": "A"}},
    ]

    assert revision_variants(chunks) == [{"doc_number": "MAN-1", "revisions": ["B", "C"]}]


def test_backend_factory_defaults_to_ollama() -> None:
    """Ollama should be the default when no backend environment override is set."""
    config = {
        "llm_backend_env": "MINI_CROSSCHECK_LLM_BACKEND",
        "default_backend": "ollama",
        "request_timeout_seconds": 30,
        "temperature": 0.0,
        "max_tokens": 64,
        "ollama": {"base_url": "http://localhost:11434", "model": "llama3.1:8b"},
        "openai_compatible": {
            "base_url_env": "OPENAI_BASE_URL",
            "api_key_env": "OPENAI_API_KEY",
            "model_env": "OPENAI_MODEL",
            "default_model": "",
        },
    }

    assert isinstance(create_backend(config, environ={}), OllamaBackend)


def test_generate_falls_back_when_backend_request_fails() -> None:
    """A timeout or model outage should still return a grounded answer from retrieved evidence."""

    class FailingBackend:
        def complete(self, prompt: str) -> str:
            raise TimeoutError("ollama timed out")

    chunks = [{
        "text": "Maximum process temperature: 120 deg C",
        "metadata": {"doc_name": "manual.pdf", "revision": "C", "page": 1},
        "score": 0.9,
        "abstention_score": 0.9,
    }]

    result = generate_answer("What is the limit?", chunks, FailingBackend(), 0.25)

    assert result["abstained"] is False
    assert "120 deg C" in result["answer"]
    assert "[manual.pdf, revision C, page 1]" in result["answer"]


def test_invalid_abstention_threshold_is_rejected() -> None:
    """Threshold configuration outside the normalized score range should fail."""
    with pytest.raises(ValueError, match="abstention_threshold"):
        generate_answer("Question", [], FakeBackend(), abstention_threshold=1.1)