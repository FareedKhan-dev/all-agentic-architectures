"""agentic_architectures — 35+ production-grade agentic AI architectures, provider-agnostic."""

from __future__ import annotations

from agentic_architectures.config import Settings, settings
from agentic_architectures.llm.factory import get_embeddings, get_llm
from agentic_architectures.tracing.langsmith import enable_langsmith

__version__ = "0.2.0"

__all__ = [
    "Settings",
    "__version__",
    "enable_langsmith",
    "get_embeddings",
    "get_llm",
    "settings",
]
