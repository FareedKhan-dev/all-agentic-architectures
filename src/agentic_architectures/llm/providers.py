"""Provider metadata — recommended models, docs URLs, signup links.

Used by the docs site and the `00_setup_and_providers` notebook to render a
provider comparison table.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderInfo:
    id: str
    name: str
    signup_url: str
    docs_url: str
    recommended_model: str
    is_local: bool = False
    notes: str = ""


PROVIDERS: dict[str, ProviderInfo] = {
    "nebius": ProviderInfo(
        id="nebius",
        name="Nebius AI Studio",
        signup_url="https://studio.nebius.com",
        docs_url="https://docs.nebius.com/studio/inference",
        recommended_model="meta-llama/Llama-3.3-70B-Instruct",
        notes="Generous free tier, OpenAI-compatible API. Also offers Qwen3, DeepSeek, Mistral.",
    ),
    "openai": ProviderInfo(
        id="openai",
        name="OpenAI",
        signup_url="https://platform.openai.com/signup",
        docs_url="https://platform.openai.com/docs/models",
        recommended_model="gpt-4o-mini",
        notes="Industry standard. Excellent tool-calling and structured output.",
    ),
    "anthropic": ProviderInfo(
        id="anthropic",
        name="Anthropic Claude",
        signup_url="https://console.anthropic.com",
        docs_url="https://docs.anthropic.com",
        recommended_model="claude-sonnet-4-5",
        notes="Best-in-class for long context and agentic tool use.",
    ),
    "groq": ProviderInfo(
        id="groq",
        name="Groq",
        signup_url="https://console.groq.com",
        docs_url="https://console.groq.com/docs",
        recommended_model="llama-3.3-70b-versatile",
        notes="Extremely low latency — great for ReAct loops.",
    ),
    "ollama": ProviderInfo(
        id="ollama",
        name="Ollama (local)",
        signup_url="https://ollama.com/download",
        docs_url="https://github.com/ollama/ollama",
        recommended_model="llama3.1:8b",
        is_local=True,
        notes="Fully local. No API key. Run any open model on your machine.",
    ),
    "together": ProviderInfo(
        id="together",
        name="Together AI",
        signup_url="https://api.together.xyz",
        docs_url="https://docs.together.ai",
        recommended_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ),
    "fireworks": ProviderInfo(
        id="fireworks",
        name="Fireworks AI",
        signup_url="https://fireworks.ai",
        docs_url="https://docs.fireworks.ai",
        recommended_model="accounts/fireworks/models/llama-v3p1-70b-instruct",
    ),
    "mistralai": ProviderInfo(
        id="mistralai",
        name="Mistral AI",
        signup_url="https://console.mistral.ai",
        docs_url="https://docs.mistral.ai",
        recommended_model="mistral-large-latest",
    ),
    "google": ProviderInfo(
        id="google",
        name="Google Gemini",
        signup_url="https://aistudio.google.com",
        docs_url="https://ai.google.dev/gemini-api/docs",
        recommended_model="gemini-2.0-flash",
    ),
}
