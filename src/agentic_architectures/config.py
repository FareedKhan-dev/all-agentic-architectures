"""Environment-driven configuration shared by every architecture.

A single `Settings` object reads `.env` (or process env) once and is reused
everywhere. Notebooks never read `os.environ` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import find_dotenv
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _locate_env_file() -> str:
    """Find `.env` by walking up from this file, then CWD. Returns "" if not found."""
    # 1. Walk up from this source file (works for `python -c "import lib"`).
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if candidate.is_file():
            return str(candidate)
    # 2. Walk up from CWD (works for arbitrary scripts/notebooks).
    found = find_dotenv(usecwd=True)
    return found

Provider = Literal[
    "nebius",
    "openai",
    "anthropic",
    "groq",
    "ollama",
    "together",
    "fireworks",
    "mistralai",
    "google",
    "huggingface",
]

VectorBackend = Literal["faiss", "chroma", "qdrant"]
GraphBackend = Literal["networkx", "neo4j"]


class Settings(BaseSettings):
    """All env vars in one place. Read from `.env` at the repo root."""

    # ---- LLM selection ----
    llm_provider: Provider = "nebius"
    llm_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    llm_temperature: float = 0.0

    # ---- Provider API keys ----
    nebius_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    together_api_key: SecretStr | None = None
    fireworks_api_key: SecretStr | None = None
    mistral_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    huggingfacehub_api_token: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"

    # ---- Tools ----
    tavily_api_key: SecretStr | None = None

    # ---- LangSmith tracing (accept both new + legacy env-var names) ----
    langsmith_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"),
    )
    langsmith_project: str = Field(
        default="all-agentic-architectures",
        validation_alias=AliasChoices("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"),
    )
    langchain_tracing_v2: bool = True

    # ---- Vector / graph stores ----
    # Default to in-process backends so notebooks run with zero external infrastructure.
    # Switch to qdrant/neo4j when you want to scale beyond a single notebook run.
    vector_backend: VectorBackend = "faiss"
    graph_backend: GraphBackend = "networkx"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: SecretStr | None = None

    # ---- Embeddings (defaults inherited from LLM provider) ----
    embeddings_provider: str = Field(default="")
    embeddings_model: str = Field(default="")

    model_config = SettingsConfigDict(
        env_file=_locate_env_file() or ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def api_key_for(self, provider: Provider) -> SecretStr | None:
        """Return the configured key for a given provider, or None if missing."""
        return {
            "nebius": self.nebius_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "groq": self.groq_api_key,
            "together": self.together_api_key,
            "fireworks": self.fireworks_api_key,
            "mistralai": self.mistral_api_key,
            "google": self.google_api_key,
            "huggingface": self.huggingfacehub_api_token,
            "ollama": None,  # local, no key
        }.get(provider)


settings = Settings()
