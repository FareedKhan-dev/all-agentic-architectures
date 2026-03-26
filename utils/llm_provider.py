"""
Multi-provider LLM factory for All Agentic Architectures.

Supports seamless switching between LLM providers via environment variables,
keeping all 17 notebooks provider-agnostic.

Supported providers:
  - nebius   (default) — via langchain-nebius
  - minimax  — MiniMax M2.7 / M2.7-highspeed via OpenAI-compatible API
  - openai   — OpenAI GPT models via langchain-openai

Usage:
    from utils.llm_provider import get_llm
    llm = get_llm()                          # uses LLM_PROVIDER env var
    llm = get_llm(provider="minimax")        # explicit override
    llm = get_llm(temperature=0.5)           # pass-through kwargs
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Provider defaults
# ---------------------------------------------------------------------------
PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "nebius": {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "temperature": 0.0,
    },
    "minimax": {
        "model": "MiniMax-M2.7",
        "temperature": 0.1,
    },
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
    },
}

SUPPORTED_PROVIDERS = list(PROVIDER_DEFAULTS.keys())


def _clamp_temperature(temperature: float) -> float:
    """MiniMax requires temperature in (0.0, 1.0]."""
    if temperature <= 0.0:
        return 0.01
    if temperature > 1.0:
        return 1.0
    return temperature


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    **kwargs: Any,
):
    """Return a LangChain chat model for the requested provider.

    Parameters
    ----------
    provider : str, optional
        One of ``"nebius"``, ``"minimax"``, ``"openai"``.
        Falls back to the ``LLM_PROVIDER`` env var, then ``"nebius"``.
    model : str, optional
        Override the default model for the chosen provider.
    temperature : float, optional
        Sampling temperature.  Clamped to (0, 1] for MiniMax.
    **kwargs
        Forwarded to the underlying LangChain chat model constructor.
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "nebius").lower().strip()

    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider: {provider!r}. "
            f"Choose from {SUPPORTED_PROVIDERS}."
        )

    defaults = PROVIDER_DEFAULTS[provider]
    model = model or os.getenv("LLM_MODEL") or defaults["model"]
    temperature = temperature if temperature is not None else defaults["temperature"]

    # ----- Nebius -----
    if provider == "nebius":
        from langchain_nebius import ChatNebius

        return ChatNebius(model=model, temperature=temperature, **kwargs)

    # ----- MiniMax (OpenAI-compatible) -----
    if provider == "minimax":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "MINIMAX_API_KEY environment variable is required for the "
                "MiniMax provider. Get one at https://www.minimax.io"
            )

        temperature = _clamp_temperature(temperature)
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base="https://api.minimax.io/v1",
            **kwargs,
        )

    # ----- OpenAI -----
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, **kwargs)

    # Should not reach here
    raise ValueError(f"Unhandled provider: {provider!r}")
