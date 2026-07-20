"""Web-search tool — wraps `langchain_tavily.TavilySearch` with retry + error handling.

Replaces deprecated `TavilySearchResults` import in old notebooks 02/03.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tenacity import retry, stop_after_attempt, wait_exponential

from agentic_architectures.config import settings

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


def web_search_tool(
    max_results: int = 5,
    search_depth: str = "advanced",
    **kwargs: Any,
) -> BaseTool:
    """Return a configured Tavily search tool, or a no-op stub if the API key is missing."""
    try:
        from langchain_tavily import TavilySearch
    except ImportError as e:
        raise ImportError("pip install agentic-architectures[tavily]") from e

    key = settings.tavily_api_key
    if key is None:
        from langchain_core.tools import tool

        @tool("web_search")
        def _noop_tool(query: str) -> str:
            """Search the web (no-op — TAVILY_API_KEY not set)."""
            return f"[web_search unavailable: TAVILY_API_KEY not set] Would search: {query}"

        return _noop_tool  # type: ignore[return-value]

    return TavilySearch(
        max_results=max_results,
        search_depth=search_depth,
        tavily_api_key=key.get_secret_value(),
        **kwargs,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def search_with_retry(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Direct callable form — invoke the tool once with exponential backoff."""
    tool = web_search_tool(max_results=max_results)
    result = tool.invoke({"query": query})
    # TavilySearch returns dict with "results" key
    if isinstance(result, dict) and "results" in result:
        return result["results"]
    return [result] if not isinstance(result, list) else result
