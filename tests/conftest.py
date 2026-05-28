"""Shared test fixtures + helpers.

Provides:
  - `MockLLM`        : a `BaseChatModel`-shaped object that returns canned responses.
  - `mock_llm`       : pytest fixture wrapping MockLLM with sensible defaults.
  - `mock_structured`: helper for when an architecture calls `llm.with_structured_output(Schema)`.
  - `RUN_INTEGRATION`: env-gated marker for integration tests.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


RUN_INTEGRATION = os.environ.get("RUN_INTEGRATION") == "1"


class MockLLM(FakeMessagesListChatModel):
    """Mock chat model with canned responses.

    Subclasses LangChain's `FakeMessagesListChatModel` so `with_structured_output()`
    and `bind()` work without monkeypatching.

    Usage:
        llm = MockLLM(responses=["hello", "world"])      # returns each in order
        llm = MockLLM(responses=[AIMessage(content="x")])

    For structured-output testing, use `mock_structured_llm()` below — it returns
    a model whose `with_structured_output(Schema)` yields instances of `Schema`
    pre-populated from the canned dicts.
    """

    @classmethod
    def from_strings(cls, *texts: str) -> "MockLLM":
        return cls(responses=[AIMessage(content=t) for t in texts])


class _StructuredMockRunnable:
    """Runnable shape that `with_structured_output()` returns. Invoking it
    yields the next canned schema instance.
    """

    def __init__(self, schema: type, canned: list[dict[str, Any]]):
        self._schema = schema
        self._canned = list(canned)
        self._i = 0

    def invoke(self, _: Any, **__: Any) -> Any:
        if not self._canned:
            raise RuntimeError(
                f"MockLLM ran out of canned responses for schema {self._schema.__name__}"
            )
        data = self._canned[self._i % len(self._canned)]
        self._i += 1
        return self._schema(**data)

    async def ainvoke(self, x: Any, **kw: Any) -> Any:
        return self.invoke(x, **kw)


class StructuredMockLLM(MockLLM):
    """MockLLM whose `with_structured_output(Schema)` returns pre-baked Pydantic objects.

    Args:
        structured_responses: dict mapping schema-class-name → list of kwarg dicts.
            Each call to with_structured_output(Schema).invoke(...) pops the next
            dict and returns Schema(**dict).
        text_responses: passed to FakeMessagesListChatModel for plain .invoke() calls.
    """

    def __init__(
        self,
        structured_responses: dict[str, list[dict[str, Any]]] | None = None,
        text_responses: list[str] | None = None,
    ):
        text_responses = text_responses or ["mock text response"]
        super().__init__(responses=[AIMessage(content=t) for t in text_responses])
        # Store structured responses on instance dict (model_config allows extras)
        self.__dict__["_structured_responses"] = structured_responses or {}

    def with_structured_output(  # type: ignore[override]
        self, schema: type, *_: Any, **__: Any
    ) -> _StructuredMockRunnable:
        canned = self.__dict__["_structured_responses"].get(schema.__name__, [])
        return _StructuredMockRunnable(schema, canned)

    def bind(self, **_: Any) -> "StructuredMockLLM":  # type: ignore[override]
        return self

    def bind_tools(  # type: ignore[override]
        self, tools: Any = None, *_: Any, **__: Any
    ) -> "StructuredMockLLM":
        """No-op for tool binding; mock ignores tools."""
        return self


@pytest.fixture
def mock_llm() -> "StructuredMockLLM":
    """A no-op StructuredMockLLM. Plain `.invoke()` returns 'mock'.
    `.with_structured_output(Schema)` returns a runnable that errors only when
    `.invoke()` is called without canned responses — instantiation-time
    `with_structured_output` calls in architecture __init__s succeed silently.
    """
    return StructuredMockLLM(text_responses=["mock"])


@pytest.fixture
def structured_mock() -> Callable[..., StructuredMockLLM]:
    """Factory for StructuredMockLLM. Use:

        def test_x(structured_mock):
            llm = structured_mock(structured_responses={'_Schema': [{'field': 'value'}]})
    """
    return StructuredMockLLM
