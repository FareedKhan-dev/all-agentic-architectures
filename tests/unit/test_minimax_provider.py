"""Tests for MiniMax provider registration, routing, and request URLs."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from agentic_architectures.config import Settings
from agentic_architectures.llm import factory
from agentic_architectures.llm.providers import PROVIDERS


def test_minimax_settings_and_registry() -> None:
    configured = Settings(
        _env_file=None,
        llm_provider="minimax",
        minimax_api_key=SecretStr("test-key"),
        minimax_base_url="https://api.minimaxi.com/v1",
    )

    key = configured.api_key_for("minimax")
    assert key is not None
    assert key.get_secret_value() == "test-key"
    assert configured.minimax_base_url == "https://api.minimaxi.com/v1"

    provider = PROVIDERS["minimax"]
    assert provider.id == "minimax"
    assert provider.recommended_model == "MiniMax-M3"
    assert "MiniMax-M2.7" in provider.notes
    assert factory.provider_supports_tools("minimax")
    assert factory.provider_supports_structured_output("minimax")


def test_minimax_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory.settings, "minimax_api_key", None)

    with pytest.raises(ValueError, match="MINIMAX_API_KEY is not configured"):
        factory.get_llm(provider="minimax", model="MiniMax-M3")


@pytest.mark.parametrize(
    ("base_url", "expected_host"),
    [
        ("https://api.minimax.io/v1", "api.minimax.io"),
        ("https://api.minimaxi.com/v1", "api.minimaxi.com"),
    ],
)
def test_minimax_openai_request_path(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
    expected_host: str,
) -> None:
    class ChatOpenAIStub:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=ChatOpenAIStub))
    monkeypatch.setattr(factory.settings, "minimax_api_key", SecretStr("test-key"))
    monkeypatch.setattr(factory.settings, "minimax_base_url", base_url)

    llm = factory.get_llm(provider="minimax", model="MiniMax-M3", request_timeout=30)
    assert isinstance(llm, ChatOpenAIStub)
    assert llm.kwargs == {
        "model": "MiniMax-M3",
        "temperature": 0.0,
        "api_key": "test-key",
        "base_url": base_url,
        "request_timeout": 30,
    }
    assert expected_host in llm.kwargs["base_url"]


@pytest.mark.parametrize(
    ("base_url", "expected_host"),
    [
        ("https://api.minimax.io/anthropic", "api.minimax.io"),
        ("https://api.minimaxi.com/anthropic", "api.minimaxi.com"),
    ],
)
def test_minimax_anthropic_request_path(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
    expected_host: str,
) -> None:
    class ChatAnthropicStub:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "langchain_anthropic", SimpleNamespace(ChatAnthropic=ChatAnthropicStub))
    monkeypatch.setattr(factory.settings, "minimax_api_key", SecretStr("test-key"))
    monkeypatch.setattr(factory.settings, "minimax_base_url", base_url)

    llm = factory.get_llm(
        provider="minimax",
        model="MiniMax-M3",
        max_tokens=128,
    )
    assert isinstance(llm, ChatAnthropicStub)
    assert llm.kwargs == {
        "model_name": "MiniMax-M3",
        "temperature": 0.0,
        "api_key": "test-key",
        "base_url": base_url,
        "max_tokens": 128,
    }
    assert expected_host in llm.kwargs["base_url"]
