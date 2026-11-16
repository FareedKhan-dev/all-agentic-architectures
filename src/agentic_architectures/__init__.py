"""agentic_architectures — 35+ production-grade agentic AI architectures, provider-agnostic."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from agentic_architectures.config import Settings, settings
from agentic_architectures.llm.factory import get_embeddings, get_llm
from agentic_architectures.tracing.langsmith import enable_langsmith

try:  # the installed distribution's version, which release-please bumps in pyproject.toml
    __version__ = _version("agentic-architectures")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = [
    "Settings",
    "__version__",
    "enable_langsmith",
    "get_embeddings",
    "get_llm",
    "settings",
]
