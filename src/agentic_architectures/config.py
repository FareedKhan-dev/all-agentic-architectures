"""Environment-driven configuration shared by every architecture.

A single `Settings` object reads `.env` (or process env) once and is reused
everywhere. Notebooks never read `os.environ` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import find_dotenv
from pydantic import AliasChoices, Field, SecretStr, model_validator
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
    "deepseek",
    "openrouter",
]

VectorBackend = Literal["memory", "faiss", "chroma", "qdrant"]
GraphBackend = Literal["networkx", "neo4j"]


class Settings(BaseSettings):
    """All env vars in one place. Read from `.env` at the repo root."""

    # ---- LLM selection ----
    llm_provider: Provider = "nebius"
    # Empty means "the provider's recommended model" (llm/providers.py), filled in below, so
    # switching LLM_PROVIDER alone never sends one provider's model name to another.
    llm_model: str = ""
    # The model the notebooks use where a reasoning ("thinking") model makes the difference,
    # e.g. deepseek-v4-pro, or qwen3:14b on Ollama. Empty means the same model as llm_model.
    llm_reasoning_model: str = ""
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
    deepseek_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"

    # ---- Tools ----
    tavily_api_key: SecretStr | None = None
    # The default web search: Tavily, or "mcp" for the MCP search server at search_mcp_url
    # (Parallel's free, keyless server), which needs no key and no signup.
    search_backend: Literal["tavily", "mcp"] = "tavily"
    search_mcp_url: str = "https://search.parallel.ai/mcp"
    # Sent as a bearer token. Parallel's keyless tier is rate-limited; a key lifts the limit.
    search_mcp_api_key: SecretStr | None = None

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
    # "memory" is langchain-core's own InMemoryVectorStore, so it needs no extra package.
    # Switch to qdrant/neo4j when you want to scale beyond a single notebook run.
    vector_backend: VectorBackend = "memory"
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
        # `.env.example` lists every key with an empty value; an empty value means "not set",
        # so TAVILY_API_KEY= gives the clear missing-key error, not a 401 at the first call.
        env_ignore_empty=True,
    )

    @model_validator(mode="after")
    def _default_model_for_provider(self) -> Settings:
        if not self.llm_model:
            self.llm_model = default_model_for(self.llm_provider)
        if not self.llm_reasoning_model:
            self.llm_reasoning_model = self.llm_model
        return self

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
            "deepseek": self.deepseek_api_key,
            "openrouter": self.openrouter_api_key,
            "ollama": None,  # local, no key
        }.get(provider)


def default_model_for(provider: str) -> str:
    """The recommended model of ``provider``, used when no model is configured.

    Empty for a provider without a recommendation (Hugging Face, where the model is the
    whole choice); `get_llm` then asks for LLM_MODEL instead of guessing.
    """
    from agentic_architectures.llm.providers import PROVIDERS

    info = PROVIDERS.get(provider)
    return info.recommended_model if info is not None else ""


settings = Settings()
