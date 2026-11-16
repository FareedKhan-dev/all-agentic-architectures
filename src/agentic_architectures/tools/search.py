"""Web-search tool: Tavily, or any MCP search server with no key at all.

`web_search_tool()` is the search tool the architectures use when they are not given
tools of their own. It is Tavily (TAVILY_API_KEY) unless SEARCH_BACKEND=mcp, which takes
the search tool of the MCP server at SEARCH_MCP_URL instead: Parallel's free, keyless
search server by default, so the search architectures run without signing up anywhere.
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
    """Return the configured web search tool.

    With SEARCH_BACKEND=mcp, ``max_results`` and ``search_depth`` do not apply: the MCP
    server decides how much it returns.

    Raises:
        RuntimeError: Tavily is selected and TAVILY_API_KEY is not set. Searching nothing
            quietly would let an agent answer from memory while looking like it searched.
    """
    if settings.search_backend == "mcp":
        from agentic_architectures.tools.mcp import mcp_tools

        key = settings.search_mcp_api_key
        headers = {"Authorization": f"Bearer {key.get_secret_value()}"} if key is not None else None
        tools = mcp_tools(settings.search_mcp_url, headers=headers)
        return next((t for t in tools if "search" in t.name.lower()), tools[0])

    try:
        from langchain_tavily import TavilySearch
    except ImportError as e:
        raise ImportError("pip install agentic-architectures[tavily]") from e

    key = settings.tavily_api_key
    if key is None:
        raise RuntimeError(
            "TAVILY_API_KEY not set. Sign up free at https://app.tavily.com and add the key to your .env file, "
            "or set SEARCH_BACKEND=mcp to search through a free, keyless MCP server "
            "(pip install agentic-architectures[mcp])."
        )

    return TavilySearch(
        max_results=max_results,
        search_depth=search_depth,
        tavily_api_key=key.get_secret_value(),
        **kwargs,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def search_with_retry(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Direct callable form: invoke the Tavily tool once with exponential backoff."""
    tool = web_search_tool(max_results=max_results)
    result = tool.invoke({"query": query})
    # TavilySearch returns dict with "results" key
    if isinstance(result, dict) and "results" in result:
        return result["results"]
    return [result] if not isinstance(result, list) else result
