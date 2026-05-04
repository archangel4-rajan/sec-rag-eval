"""Centralized configuration. Reads from environment / .env file.

All other modules import `settings` from here. Never read os.environ directly elsewhere.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # SEC EDGAR
    sec_user_agent: str = Field(
        default="sec-rag-eval test@example.com",
        description="SEC requires a User-Agent with name and email. Override in .env.",
    )

    # LLM providers
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    together_api_key: str = ""
    groq_api_key: str = ""

    # Langfuse (used Day 8+)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # Paths
    data_dir: Path = Path("./data")
    chroma_path: Path = Path("./data/chroma")

    @property
    def filings_dir(self) -> Path:
        return self.data_dir / "filings"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"


settings = Settings()
