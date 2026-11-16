"""Provider metadata: recommended models, docs URLs, signup links.

Used by the docs site and the `00_setup_and_providers` notebook to render a
provider comparison table, and by the settings: an unset LLM_MODEL means the
provider's `recommended_model`.

Each recommended model is one the provider serves on its normal pay-per-token API, checked
against the provider's own model and deprecation pages in November 2026. Providers retire
models often (six of the defaults here had been retired or deprecated by then), so when a
default stops working, set LLM_MODEL to the model the provider's deprecation page names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProviderInfo:
    id: str
    name: str
    signup_url: str
    docs_url: str
    recommended_model: str
    is_local: bool = False
    notes: str = ""
    # Passed to the chat model whenever `recommended_model` is used (explicit kwargs win).
    model_kwargs: dict[str, Any] = field(default_factory=dict)


PROVIDERS: dict[str, ProviderInfo] = {
    "nebius": ProviderInfo(
        id="nebius",
        name="Nebius Token Factory",
        signup_url="https://tokenfactory.nebius.com",
        docs_url="https://docs.tokenfactory.nebius.com",
        recommended_model="nvidia/Nemotron-3_5-Lightning",
        notes="Nebius' own replacement for Llama 3.3 70B, which left its serverless API on Aug 31, 2026. "
        "Tool calling, low price.",
    ),
    "openai": ProviderInfo(
        id="openai",
        name="OpenAI",
        signup_url="https://platform.openai.com/signup",
        docs_url="https://developers.openai.com/api/docs/models",
        recommended_model="gpt-6-luna",
        notes="OpenAI's replacement for gpt-4o-mini (deprecated, shut down Jan 14, 2027). On Chat Completions "
        "it calls tools only with reasoning_effort='none', which is the default here for this model.",
        model_kwargs={"reasoning_effort": "none"},
    ),
    "anthropic": ProviderInfo(
        id="anthropic",
        name="Anthropic Claude",
        signup_url="https://platform.claude.com",
        docs_url="https://platform.claude.com/docs",
        recommended_model="claude-sonnet-5-5",
        notes="Strong at long context and agentic tool use. claude-haiku-5-5 is the cheaper choice.",
    ),
    "groq": ProviderInfo(
        id="groq",
        name="Groq",
        signup_url="https://console.groq.com",
        docs_url="https://console.groq.com/docs",
        recommended_model="openai/gpt-oss-120b",
        notes="Extremely low latency, great for ReAct loops. Groq's replacement for llama-3.3-70b-versatile.",
    ),
    "ollama": ProviderInfo(
        id="ollama",
        name="Ollama (local)",
        signup_url="https://ollama.com/download",
        docs_url="https://github.com/ollama/ollama",
        recommended_model="qwen3:8b",
        is_local=True,
        notes="Fully local. No API key. Every architecture ran on qwen3:8b (a 5.2 GB download) in the real-model "
        "sweep; qwen3:14b is stronger.",
    ),
    "together": ProviderInfo(
        id="together",
        name="Together AI",
        signup_url="https://api.together.xyz",
        docs_url="https://docs.together.ai",
        recommended_model="openai/gpt-oss-120b",
    ),
    "fireworks": ProviderInfo(
        id="fireworks",
        name="Fireworks AI",
        signup_url="https://fireworks.ai",
        docs_url="https://docs.fireworks.ai",
        recommended_model="accounts/fireworks/models/glm-5p3-flash",
    ),
    "mistralai": ProviderInfo(
        id="mistralai",
        name="Mistral AI",
        signup_url="https://console.mistral.ai",
        docs_url="https://docs.mistral.ai",
        recommended_model="mistral-large-latest",
        notes="The alias points to Mistral Large 4.",
    ),
    "deepseek": ProviderInfo(
        id="deepseek",
        name="DeepSeek",
        signup_url="https://platform.deepseek.com",
        docs_url="https://api-docs.deepseek.com",
        recommended_model="deepseek-flash",
        notes="deepseek-chat and deepseek-reasoner were retired on Jul 24, 2026. deepseek-v4-pro is the larger model.",
    ),
    "google": ProviderInfo(
        id="google",
        name="Google Gemini",
        signup_url="https://aistudio.google.com",
        docs_url="https://ai.google.dev/gemini-api/docs",
        recommended_model="gemini-3.5-flash-lite",
        notes="gemini-2.0-flash was shut down on Jun 1, 2026. Free tier available.",
    ),
    "openrouter": ProviderInfo(
        id="openrouter",
        name="OpenRouter",
        signup_url="https://openrouter.ai",
        docs_url="https://openrouter.ai/docs",
        recommended_model="openai/gpt-oss-120b",
        notes="One key for hundreds of models from many providers (MiniMax, Qwen, DeepSeek, ...).",
    ),
}
