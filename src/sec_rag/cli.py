"""CLI entrypoint. Subcommands grow over the project.

sec-rag ingest download   # Day 1
sec-rag ingest parse      # Day 2
sec-rag ingest chunk      # Day 2 (next)
sec-rag ingest embed      # Day 3
sec-rag query "..."       # Day 4
sec-rag eval run          # Day 7
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="sec-rag-eval CLI.",
)

ingest_app = typer.Typer(no_args_is_help=True, help="Ingestion pipeline commands.")
app.add_typer(ingest_app, name="ingest")


@ingest_app.command("download")
def ingest_download() -> None:
    """Download 10-K filings from SEC EDGAR for all configured tickers."""
    from sec_rag.ingest.download import download_all

    download_all()


@ingest_app.command("parse")
def ingest_parse() -> None:
    """Extract Item 1A and Item 7 sections from raw filings."""
    from sec_rag.ingest.parse import parse_all

    parse_all()


@ingest_app.command("chunk")
def ingest_chunk(
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", help="Maximum tokens per chunk."),
    ] = 512,
    overlap: Annotated[
        int,
        typer.Option("--overlap", help="Token overlap between adjacent chunks."),
    ] = 50,
) -> None:
    """Split parsed sections into embedding-ready chunks."""
    from sec_rag.config import settings
    from sec_rag.ingest.chunk import CHUNKS_JSONL, chunk_all

    chunks = chunk_all(chunk_size_tokens=chunk_size, overlap_tokens=overlap)
    typer.echo(f"Wrote {len(chunks)} chunks to {settings.chunks_dir / CHUNKS_JSONL}")


@ingest_app.command("embed")
def ingest_embed(
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Embedding provider: openai or hash."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Embedding model name."),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", help="Chunks per embedding batch."),
    ] = None,
    reset: Annotated[
        bool,
        typer.Option("--reset", help="Delete and rebuild the Chroma collection."),
    ] = False,
) -> None:
    """Embed chunks and load into Chroma."""
    from sec_rag.ingest.embed import embed_all

    report = embed_all(
        provider=provider,
        model=model,
        batch_size=batch_size,
        reset_collection=reset,
    )
    typer.echo(f"Indexed {report.indexed_chunks} chunks into {report.collection_name}")


@app.command("query")
def query(
    question: str,
    top_k: Annotated[
        int,
        typer.Option("--top-k", help="Number of chunks to retrieve."),
    ] = 5,
    ticker: Annotated[
        str | None,
        typer.Option("--ticker", help="Restrict retrieval to one ticker."),
    ] = None,
    year: Annotated[
        int | None,
        typer.Option("--year", help="Restrict retrieval to one filing year."),
    ] = None,
    section: Annotated[
        str | None,
        typer.Option("--section", help="Restrict retrieval to an SEC item, e.g. 1A or 7."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Embedding provider for the query."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Embedding model for the query."),
    ] = None,
) -> None:
    """Run a single query against the RAG."""
    from sec_rag.generate import QueryEngine, format_rag_response

    engine = QueryEngine(provider=provider, model=model)
    response = engine.answer(question, top_k=top_k, ticker=ticker, year=year, section=section)
    typer.echo(format_rag_response(response))


eval_app = typer.Typer(no_args_is_help=True, help="Evaluation commands.")
app.add_typer(eval_app, name="eval")


@eval_app.command("validate")
def eval_validate(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", help="Golden dataset JSONL path."),
    ] = Path("data/eval/golden.jsonl"),
    check_evidence: Annotated[
        bool,
        typer.Option("--check-evidence", help="Check evidence strings against chunks.jsonl."),
    ] = False,
) -> None:
    """Validate the golden dataset schema and optional source grounding."""
    from sec_rag.eval.dataset import (
        load_golden_dataset,
        summarize_by_category,
        validate_evidence_against_chunks,
        validate_golden_dataset,
    )

    examples = load_golden_dataset(dataset)
    issues = validate_golden_dataset(examples)
    if check_evidence:
        issues.extend(validate_evidence_against_chunks(examples))
    if issues:
        for issue in issues:
            typer.echo(f"ERROR: {issue}")
        raise typer.Exit(code=1)
    summary = summarize_by_category(examples)
    typer.echo(f"Validated {len(examples)} examples: {summary}")


@eval_app.command("run")
def eval_run(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", help="Golden dataset JSONL path."),
    ] = Path("data/eval/golden.jsonl"),
    output: Annotated[
        Path,
        typer.Option("--output", help="Where to write the eval report JSON."),
    ] = Path("experiments/baseline_eval/results.json"),
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Embedding provider for retrieval."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Embedding model for retrieval."),
    ] = None,
    top_k: Annotated[
        int,
        typer.Option("--top-k", help="Number of chunks to retrieve per query."),
    ] = 5,
    fail_under: Annotated[
        float,
        typer.Option("--fail-under", help="Fail if overall pass rate is below this value."),
    ] = 0.0,
) -> None:
    """Run the deterministic eval suite over the golden dataset."""
    from sec_rag.eval.dataset import load_golden_dataset
    from sec_rag.eval.scoring import run_eval, write_eval_report
    from sec_rag.generate import QueryEngine

    examples = load_golden_dataset(dataset)
    engine = QueryEngine(provider=provider, model=model)
    report = run_eval(examples, engine=engine, top_k=top_k)
    write_eval_report(report, output)
    typer.echo(
        f"Eval pass_rate={report.pass_rate:.3f} examples={report.total_examples} output={output}"
    )
    if report.pass_rate < fail_under:
        raise typer.Exit(code=1)


@eval_app.command("summarize")
def eval_summarize(
    results: Annotated[
        Path,
        typer.Option("--results", help="JSON eval or experiment artifact."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional Markdown output path."),
    ] = None,
    title: Annotated[
        str | None,
        typer.Option("--title", help="Optional report title."),
    ] = None,
) -> None:
    """Render an eval or experiment JSON artifact as Markdown."""
    from sec_rag.eval.reporting import (
        load_json,
        render_json_report_markdown,
        write_markdown_report,
    )

    markdown = render_json_report_markdown(load_json(results), title=title)
    if output is None:
        typer.echo(markdown)
        return
    write_markdown_report(markdown, output)
    typer.echo(f"Wrote Markdown report to {output}")


experiment_app = typer.Typer(no_args_is_help=True, help="Experiment commands.")
app.add_typer(experiment_app, name="experiment")


@experiment_app.command("retrieval")
def experiment_retrieval(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", help="Golden dataset JSONL path."),
    ] = Path("data/eval/golden.jsonl"),
    output: Annotated[
        Path,
        typer.Option("--output", help="Where to write the experiment JSON."),
    ] = Path("experiments/retrieval_sweep/results.json"),
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Embedding provider for retrieval."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Embedding model for retrieval."),
    ] = None,
    top_k_values: Annotated[
        str,
        typer.Option("--top-k-values", help="Comma-separated top-k values."),
    ] = "1,3,5,8",
) -> None:
    """Run a retrieval top-k sweep over the golden dataset."""
    from sec_rag.eval.dataset import load_golden_dataset
    from sec_rag.eval.experiments import parse_top_k_values, run_retrieval_sweep
    from sec_rag.generate import QueryEngine

    examples = load_golden_dataset(dataset)
    engine = QueryEngine(provider=provider, model=model)
    result = run_retrieval_sweep(
        examples,
        engine=engine,
        top_k_values=parse_top_k_values(top_k_values),
        output_path=output,
        metadata={
            "provider": provider or "configured-default",
            "model": model or "configured-default",
            "dataset": str(dataset),
        },
    )
    typer.echo(f"Wrote retrieval sweep with {len(result['runs'])} runs to {output}")


if __name__ == "__main__":
    app()
