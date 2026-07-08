"""FastAPI service exposing health and query endpoints."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI
from pydantic import BaseModel, Field

from sec_rag.generate import QueryEngine, RagResponse


class QueryRequest(BaseModel):
    """Request body for /query."""

    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    ticker: str | None = None
    year: int | None = None
    section: str | None = None
    provider: str | None = None
    model: str | None = None


class CitationResponse(BaseModel):
    """One citation returned to API clients."""

    id: str
    label: str
    ticker: str | None
    year: int | None
    section: str | None
    chunk_index: int | None
    distance: float
    excerpt: str


class QueryResponse(BaseModel):
    """API response for /query."""

    question: str
    answer: str
    citations: list[CitationResponse]
    retrieved_count: int
    refused: bool
    refusal_reason: str | None = None


EngineFactory = Callable[[QueryRequest], QueryEngine]


def create_app(engine_factory: EngineFactory | None = None) -> FastAPI:
    """Create the FastAPI app.

    engine_factory is injectable so tests can exercise request/response
    contracts without opening a persistent Chroma index.
    """
    app = FastAPI(title="sec-rag-eval", version="0.1.0")

    def build_engine(request: QueryRequest) -> QueryEngine:
        if engine_factory is not None:
            return engine_factory(request)
        return QueryEngine(provider=request.provider, model=request.model)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        engine = build_engine(request)
        response = engine.answer(
            request.question,
            top_k=request.top_k,
            ticker=request.ticker,
            year=request.year,
            section=request.section,
        )
        return _to_query_response(response)

    return app


def _to_query_response(response: RagResponse) -> QueryResponse:
    return QueryResponse(
        question=response.question,
        answer=response.answer,
        citations=[
            CitationResponse(
                id=citation.id,
                label=citation.label,
                ticker=citation.ticker,
                year=citation.year,
                section=citation.section,
                chunk_index=citation.chunk_index,
                distance=citation.distance,
                excerpt=citation.excerpt,
            )
            for citation in response.citations
        ],
        retrieved_count=response.retrieved_count,
        refused=response.refused,
        refusal_reason=response.refusal_reason,
    )


app = create_app()
