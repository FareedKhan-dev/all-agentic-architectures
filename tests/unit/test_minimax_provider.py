"""Tests for MiniMax provider registration, routing, and request URLs."""

from __future__ import annotations

import httpx
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
    pytest.importorskip("langchain_openai")
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "MiniMax-M3",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(factory.settings, "minimax_api_key", SecretStr("test-key"))
    monkeypatch.setattr(factory.settings, "minimax_base_url", base_url)

    llm = factory.get_llm(provider="minimax", model="MiniMax-M3", http_client=client)
    assert llm.invoke("Hello").content == "ok"
    assert [(request.url.host, request.url.path) for request in requests] == [(expected_host, "/v1/chat/completions")]
    client.close()


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
    pytest.importorskip("langchain_anthropic")
    from langchain_anthropic import chat_models

    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "MiniMax-M3",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(chat_models, "_get_default_httpx_client", lambda **_: client)
    monkeypatch.setattr(factory.settings, "minimax_api_key", SecretStr("test-key"))
    monkeypatch.setattr(factory.settings, "minimax_base_url", base_url)

    llm = factory.get_llm(
        provider="minimax",
        model="MiniMax-M3",
        max_tokens=128,
    )
    assert llm.anthropic_api_url == base_url
    assert llm.invoke("Hello").content == "ok"
    assert [(request.url.host, request.url.path) for request in requests] == [(expected_host, "/anthropic/v1/messages")]
    client.close()
