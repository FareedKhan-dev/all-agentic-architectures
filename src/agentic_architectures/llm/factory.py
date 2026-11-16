"""Single switch point for picking an LLM (or embeddings model) by provider.

Every notebook and every Architecture in the library calls `get_llm()` —
nothing else. To switch providers, set `LLM_PROVIDER` in `.env` or pass
`provider="..."` here.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from agentic_architectures.config import Provider, default_model_for, settings
from agentic_architectures.llm.providers import PROVIDERS

if TYPE_CHECKING:  # avoid hard import at module load time
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel


# ----------------------------------------------------------------------------
# Capability matrix — what each provider can reliably do.
# Architectures that need tool-calling or structured output check this before
# running on a given provider and skip with a clear message if unsupported.
# ----------------------------------------------------------------------------
_PROVIDER_CAPABILITIES: dict[str, dict[str, bool]] = {
    "nebius": {"tools": True, "structured_output": True},
    "openai": {"tools": True, "structured_output": True},
    "anthropic": {"tools": True, "structured_output": True},
    "groq": {"tools": True, "structured_output": True},
    "together": {"tools": True, "structured_output": True},
    "fireworks": {"tools": True, "structured_output": True},
    "mistralai": {"tools": True, "structured_output": True},
    "google": {"tools": True, "structured_output": True},
    # Ollama serves JSON-schema structured output (0.5+); every architecture that uses it ran on qwen3.
    "ollama": {"tools": True, "structured_output": True},
    "huggingface": {"tools": False, "structured_output": False},
    "deepseek": {"tools": True, "structured_output": True},
    "openrouter": {"tools": True, "structured_output": True},
}

# Where this library's provider name differs from the one langchain's init_chat_model takes.
_LANGCHAIN_PROVIDER_NAME: dict[str, str] = {"google": "google_genai"}

# The extra that installs each provider's integration, for the error a missing one raises.
_PROVIDER_EXTRA: dict[str, str] = {"mistralai": "mistral"}


def provider_supports_tools(provider: str | None = None) -> bool:
    return _PROVIDER_CAPABILITIES.get(provider or settings.llm_provider, {}).get("tools", False)


def provider_supports_structured_output(provider: str | None = None) -> bool:
    return _PROVIDER_CAPABILITIES.get(provider or settings.llm_provider, {}).get("structured_output", False)


def get_llm(
    provider: Provider | None = None,
    model: str | None = None,
    temperature: float | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Return a configured chat model for the given provider.

    Resolution order: explicit args -> .env / Settings -> hard defaults. `LLM_MODEL`
    belongs to `LLM_PROVIDER`, so asking for another provider without a model gets that
    provider's recommended model, never the configured provider's model name.
    Raises ImportError with a helpful pip-install hint if the provider's
    integration package isn't installed.
    """
    provider = provider or settings.llm_provider
    if not model:
        model = settings.llm_model if provider == settings.llm_provider else default_model_for(provider)
    if not model:
        raise ValueError(f"No model configured for provider {provider!r}: set LLM_MODEL in .env or pass model=")
    info = PROVIDERS.get(provider)
    if info is not None and model == info.recommended_model:
        for key_name, value in info.model_kwargs.items():  # e.g. what gpt-6-luna needs for tool calls
            kwargs.setdefault(key_name, value)
    temperature = temperature if temperature is not None else settings.llm_temperature

    key = settings.api_key_for(provider)
    api_key = key.get_secret_value() if key is not None else None

    if provider == "nebius":
        try:
            from langchain_nebius import ChatNebius
        except ImportError as e:
            raise ImportError("Nebius provider requires `pip install agentic-architectures[nebius]`") from e
        return ChatNebius(
            model=model,
            temperature=temperature,
            api_key=api_key,
            **kwargs,
        )

    # All other providers go through langchain's init_chat_model.
    try:
        from langchain.chat_models import init_chat_model
    except ImportError as e:
        raise ImportError("langchain>=0.3 is required") from e

    # init_chat_model reads API keys from process env, so push the configured
    # key in (without persisting to disk).
    _ensure_provider_env(provider, api_key)
    if provider == "ollama":
        kwargs.setdefault("base_url", settings.ollama_base_url)  # OLLAMA_BASE_URL, as for embeddings
        # Ollama's own default window is 4,096 tokens, and it silently drops the start of a longer
        # prompt; agent prompts (retrieved documents, debate transcripts) run past that easily.
        kwargs.setdefault("num_ctx", 16384)

    try:
        return init_chat_model(
            model=model,
            model_provider=_LANGCHAIN_PROVIDER_NAME.get(provider, provider),
            temperature=temperature,
            **kwargs,
        )
    except ImportError as e:
        extra = _PROVIDER_EXTRA.get(provider, provider)
        raise ImportError(f"The {provider} provider requires `pip install agentic-architectures[{extra}]`") from e


def _ensure_provider_env(provider: str, api_key: str | None) -> None:
    """Populate the env var that LangChain expects for this provider."""
    if api_key is None:
        return
    env_var = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
        "together": "TOGETHER_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "mistralai": "MISTRAL_API_KEY",
        "google": "GOOGLE_API_KEY",
        "huggingface": "HUGGINGFACEHUB_API_TOKEN",
        "deepseek": "DEEPSEEK_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(provider)
    if env_var and not os.environ.get(env_var):
        os.environ[env_var] = api_key


# ----------------------------------------------------------------------------
# Embeddings — same idea, separate function.
# ----------------------------------------------------------------------------

_DEFAULT_EMBEDDINGS_BY_PROVIDER: dict[str, tuple[str, str]] = {
    "nebius": ("nebius", "Qwen/Qwen3-Embedding-8B"),
    "openai": ("openai", "text-embedding-3-small"),
    "google": ("google", "models/gemini-embedding-2"),  # text-embedding-004 was shut down Jan 14, 2026
    "huggingface": ("huggingface", "sentence-transformers/all-MiniLM-L6-v2"),
    "ollama": ("ollama", "nomic-embed-text"),
}


def get_embeddings(
    provider: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> Embeddings:
    """Return an embeddings model. Defaults inherit from `LLM_PROVIDER`.

    An LLM provider without an embeddings model (Anthropic, Groq, DeepSeek, ...) falls back
    to a free local sentence-transformers model, unless a provider was asked for by name.
    """
    explicit = provider or settings.embeddings_provider
    provider = explicit or settings.llm_provider
    if not explicit and provider not in _DEFAULT_EMBEDDINGS_BY_PROVIDER:
        provider = "huggingface"
    default_model = _DEFAULT_EMBEDDINGS_BY_PROVIDER.get(provider, ("", ""))[1]
    model = model or settings.embeddings_model or default_model

    if provider == "nebius":
        try:
            from langchain_nebius import NebiusEmbeddings
        except ImportError as e:
            raise ImportError("pip install agentic-architectures[nebius]") from e
        key = settings.nebius_api_key
        return NebiusEmbeddings(
            model=model,
            api_key=key.get_secret_value() if key is not None else None,
            **kwargs,
        )
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        _ensure_provider_env(
            "openai",
            settings.openai_api_key.get_secret_value() if settings.openai_api_key else None,
        )
        return OpenAIEmbeddings(model=model, **kwargs)
    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as e:
            why = "" if explicit else f" ({settings.llm_provider} has no embeddings model, so a local one is used)"
            raise ImportError(
                f"Embeddings need `pip install agentic-architectures[huggingface] sentence-transformers`{why}, "
                "or set EMBEDDINGS_PROVIDER to one of: " + ", ".join(_DEFAULT_EMBEDDINGS_BY_PROVIDER)
            ) from e

        return HuggingFaceEmbeddings(model_name=model, **kwargs)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=model,
            base_url=settings.ollama_base_url,
            **kwargs,
        )
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        _ensure_provider_env(
            "google",
            settings.google_api_key.get_secret_value() if settings.google_api_key else None,
        )
        return GoogleGenerativeAIEmbeddings(model=model, **kwargs)

    raise ValueError(
        f"No embeddings implementation registered for provider={provider!r}. "
        f"Supported: {list(_DEFAULT_EMBEDDINGS_BY_PROVIDER)}"
    )
