"""Opt-in web search through Parallel's free, keyless Search MCP server."""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


def parallel_search_tool(timeout: float = 60.0) -> BaseTool:
    """Return a Parallel search tool for explicit architecture tool selection.

    Args:
        timeout: Seconds before cancelling the MCP session. Cleanup is awaited.

    Returns:
        A tool accepting a single ``query`` and returning source URLs and excerpts.

    Raises:
        ImportError: If the optional ``parallel`` extra is missing.
        ValueError: If timeout is not finite and positive.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")

    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as e:
        raise ImportError("pip install agentic-architectures[parallel]") from e

    from langchain_core.tools import StructuredTool, ToolException
    from pydantic import BaseModel, Field

    from agentic_architectures import __version__

    class SearchInput(BaseModel):
        query: str = Field(min_length=1, description="Web search query or research question.")

    # Stable for this tool instance, including across its separate MCP sessions.
    session_id = uuid4().hex

    async def search(query: str) -> dict[str, Any]:
        async def call() -> dict[str, Any]:
            async with (
                streamablehttp_client(
                    "https://search.parallel.ai/mcp",
                    headers={"User-Agent": f"agentic-architectures/{__version__}"},
                ) as (read, write, _),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                result = await session.call_tool(
                    "web_search",
                    {"objective": query, "search_queries": [query], "session_id": session_id},
                )
            if result.isError:
                message = "\n".join(getattr(block, "text", "") for block in result.content)
                raise ToolException(f"Parallel search failed: {message}")
            if result.structuredContent is None:
                raise ToolException("Parallel search returned no structured results")
            return result.structuredContent

        return await asyncio.wait_for(call(), timeout=timeout)

    def search_sync(query: str) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(search(query))
        raise RuntimeError("Use await tool.ainvoke(...) inside an active event loop")

    return StructuredTool.from_function(
        func=search_sync,
        coroutine=search,
        name="parallel_web_search",
        description="Search the web for facts. Returns source URLs and relevant excerpts for cited answers.",
        args_schema=SearchInput,
    )
