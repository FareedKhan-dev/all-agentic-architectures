"""Provider-agnostic LLM and embeddings factory.

The factory is imported on first use, not with this package: the settings read the
provider registry (`llm/providers.py`) while they are being built, and the factory itself
imports the settings, so loading it eagerly here would be a circular import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic_architectures.llm.factory import (
        get_embeddings,
        get_llm,
        provider_supports_structured_output,
        provider_supports_tools,
    )

__all__ = [
    "get_embeddings",
    "get_llm",
    "provider_supports_structured_output",
    "provider_supports_tools",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from agentic_architectures.llm import factory

        return getattr(factory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
