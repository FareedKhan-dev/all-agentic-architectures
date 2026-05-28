"""One-call LangSmith setup. Notebooks just do:

    from agentic_architectures import enable_langsmith
    enable_langsmith()
"""

from __future__ import annotations

import os

from agentic_architectures.config import settings


def enable_langsmith(project: str | None = None) -> bool:
    """Configure LangSmith tracing if a key is present.

    Returns True if tracing was enabled, False if no key was found (silent no-op).
    """
    key = settings.langsmith_api_key
    if key is None:
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true" if settings.langchain_tracing_v2 else "false"
    os.environ["LANGCHAIN_API_KEY"] = key.get_secret_value()
    os.environ["LANGSMITH_API_KEY"] = key.get_secret_value()
    os.environ["LANGCHAIN_PROJECT"] = project or settings.langsmith_project
    os.environ["LANGSMITH_PROJECT"] = project or settings.langsmith_project
    return True
