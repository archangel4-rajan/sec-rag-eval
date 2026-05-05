"""CLI entrypoint. Subcommands grow over the project.

sec-rag ingest download   # Day 1
sec-rag ingest parse      # Day 2
sec-rag ingest chunk      # Day 2 (next)
sec-rag ingest embed      # Day 3
sec-rag query "..."       # Day 4
sec-rag eval run          # Day 7
"""

from __future__ import annotations

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
def ingest_chunk() -> None:
    """Split parsed sections into embedding-ready chunks. (Day 2 next.)"""
    typer.echo("Not implemented yet — coming next.")
    raise typer.Exit(code=1)


@ingest_app.command("embed")
def ingest_embed() -> None:
    """Embed chunks and load into Chroma. (Day 3.)"""
    typer.echo("Not implemented yet — coming Day 3.")
    raise typer.Exit(code=1)


@app.command("query")
def query(question: str) -> None:
    """Run a single query against the RAG. (Day 4.)"""
    typer.echo("Not implemented yet — coming Day 4.")
    raise typer.Exit(code=1)


eval_app = typer.Typer(no_args_is_help=True, help="Evaluation commands.")
app.add_typer(eval_app, name="eval")


@eval_app.command("run")
def eval_run() -> None:
    """Run the eval suite over the golden dataset. (Day 7.)"""
    typer.echo("Not implemented yet — coming Day 7.")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
