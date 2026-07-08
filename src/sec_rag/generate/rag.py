"""Citation-grounded query pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from sec_rag.ingest.tickers import YEARS
from sec_rag.retrieve import RetrievalResult, Retriever

WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-']*")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")


@dataclass(frozen=True)
class Citation:
    """A source citation included with an answer."""

    id: str
    label: str
    ticker: str | None
    year: int | None
    section: str | None
    chunk_index: int | None
    distance: float
    excerpt: str


@dataclass(frozen=True)
class RagResponse:
    """Grounded response returned by CLI/API/eval."""

    question: str
    answer: str
    citations: list[Citation]
    retrieved_count: int
    refused: bool = False
    refusal_reason: str | None = None


class AnswerGenerator(Protocol):
    """Generate an answer from retrieved context."""

    def generate(self, question: str, contexts: list[RetrievalResult]) -> RagResponse:
        """Generate an answer with citations."""
        ...


class RefusalPolicy:
    """Conservative rules for questions outside the corpus contract."""

    def should_refuse(self, question: str, *, year: int | None = None) -> str | None:
        text = question.lower()
        requested_years = {int(match) for match in YEAR_PATTERN.findall(question)}
        if year is not None:
            requested_years.add(year)
        unsupported_years = sorted(y for y in requested_years if y not in YEARS)
        if unsupported_years:
            return (
                "the requested filing year is outside the configured SEC 10-K corpus "
                f"({min(YEARS)}-{max(YEARS)})"
            )
        if any(marker in text for marker in ("social security", "ssn", "personal identifier")):
            return "the question asks for private personal information outside public filings"
        if any(marker in text for marker in ("private slack", "private email", "internal chat")):
            return "the question asks about private communications outside public filings"
        if any(
            marker in text
            for marker in (
                "stock price",
                "guaranteed",
                "outperform",
                "next quarter",
                "predict",
                "forecast",
            )
        ):
            return "the question asks for a future prediction or investment outcome"
        return None

    def refusal_response(self, question: str, reason: str) -> RagResponse:
        return RagResponse(
            question=question,
            answer=(
                "I can't answer that from the SEC 10-K corpus: "
                f"{reason}. Ask a question grounded in the retrieved public filings."
            ),
            citations=[],
            retrieved_count=0,
            refused=True,
            refusal_reason=reason,
        )


class ExtractiveAnswerGenerator:
    """Deterministic citation-first answer baseline.

    The baseline intentionally avoids unsupported synthesis. It extracts the
    highest-overlap sentence from each retrieved chunk and cites every claim,
    which gives the eval harness a conservative baseline before LLM generation
    experiments are introduced.
    """

    def generate(self, question: str, contexts: list[RetrievalResult]) -> RagResponse:
        if not contexts:
            return RagResponse(
                question=question,
                answer=(
                    "I do not have enough retrieved SEC filing context to answer this question."
                ),
                citations=[],
                retrieved_count=0,
            )

        query_terms = _content_terms(question)
        selected: list[tuple[RetrievalResult, str]] = []
        seen_sentences: set[str] = set()
        for result in contexts:
            sentence = _best_sentence(result.text, query_terms)
            normalized = sentence.lower()
            if normalized in seen_sentences:
                continue
            seen_sentences.add(normalized)
            selected.append((result, sentence))
            if len(selected) == 3:
                break

        citations = [_citation_from_result(result) for result, _ in selected]
        cited_sentences = [
            f"{sentence} [{number}]" for number, (_, sentence) in enumerate(selected, start=1)
        ]
        answer = "Based on the retrieved SEC filing passages: " + " ".join(cited_sentences)
        return RagResponse(
            question=question,
            answer=answer,
            citations=citations,
            retrieved_count=len(contexts),
        )


class QueryEngine:
    """RAG query orchestrator."""

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        generator: AnswerGenerator | None = None,
        provider: str | None = None,
        model: str | None = None,
        refusal_policy: RefusalPolicy | None = None,
    ) -> None:
        self.retriever = retriever or Retriever(provider=provider, model=model)
        self.generator = generator or ExtractiveAnswerGenerator()
        self.refusal_policy = refusal_policy or RefusalPolicy()

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        ticker: str | None = None,
        year: int | None = None,
        section: str | None = None,
    ) -> RagResponse:
        refusal_reason = self.refusal_policy.should_refuse(question, year=year)
        if refusal_reason is not None:
            return self.refusal_policy.refusal_response(question, refusal_reason)
        contexts = self.retriever.search(
            question,
            top_k=top_k,
            ticker=ticker,
            year=year,
            section=section,
        )
        return self.generator.generate(question, contexts)


def format_rag_response(response: RagResponse) -> str:
    """Render a response for the CLI."""
    lines = [response.answer]
    if response.citations:
        lines.append("")
        lines.append("Citations:")
        for number, citation in enumerate(response.citations, start=1):
            lines.append(
                f"[{number}] {citation.label} (id={citation.id}, distance={citation.distance:.4f})"
            )
    return "\n".join(lines)


def _citation_from_result(result: RetrievalResult) -> Citation:
    return Citation(
        id=result.id,
        label=result.citation,
        ticker=_optional_str(result.metadata.get("ticker")),
        year=_optional_int(result.metadata.get("year")),
        section=_optional_str(result.metadata.get("section")),
        chunk_index=_optional_int(result.metadata.get("chunk_index")),
        distance=result.distance,
        excerpt=_excerpt(result.text),
    )


def _best_sentence(text: str, query_terms: set[str]) -> str:
    sentences = [s.strip() for s in SENTENCE_PATTERN.split(text) if s.strip()]
    if not sentences:
        return _excerpt(text, max_chars=260)
    best = max(sentences[:20], key=lambda sentence: _overlap_score(sentence, query_terms))
    return _excerpt(best, max_chars=320)


def _overlap_score(sentence: str, query_terms: set[str]) -> tuple[int, int]:
    terms = _content_terms(sentence)
    overlap = len(terms & query_terms)
    return overlap, len(terms)


def _content_terms(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "were",
        "with",
    }
    return {
        token
        for token in WORD_PATTERN.findall(text.lower())
        if len(token) > 2 and token not in stopwords
    }


def _excerpt(text: str, *, max_chars: int = 420) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None
