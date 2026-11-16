"""Tools from any MCP (Model Context Protocol) server, for every architecture.

MCP is the open protocol tool providers now publish their tools through. Point
`mcp_tools()` at a server and pass the result to any architecture that takes tools:

    from agentic_architectures.architectures import ReAct
    from agentic_architectures.tools import mcp_tools

    # Parallel's free, keyless web search, so ReAct can search without a Tavily key
    tools = mcp_tools("https://search.parallel.ai/mcp")
    ReAct(tools=tools).run("Who won the 2026 Nobel Prize in Physics?")

The tools come from LangChain's official MCP adapters (`pip install
agentic-architectures[mcp]`). MCP tools are asynchronous; these also work from the
synchronous `run()` of every architecture, in a script and in a notebook alike (a notebook
already has an event loop running, so the call is made from a helper thread there). Each
call opens its own short session, so nothing stays connected between calls.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

T = TypeVar("T")


def mcp_tools(
    url: str,
    *,
    transport: str = "streamable_http",
    headers: dict[str, str] | None = None,
    names: list[str] | None = None,
) -> list[BaseTool]:
    """Return the tools an MCP server offers, ready for any architecture.

    Args:
        url: The server's URL, e.g. ``https://search.parallel.ai/mcp``.
        transport: ``"streamable_http"`` (the current MCP transport) or ``"sse"``.
        headers: Extra HTTP headers, e.g. ``{"Authorization": "Bearer ..."}`` for a server
            that needs a key.
        names: Keep only these tools. An unknown name is an error that lists the real ones.

    Raises:
        ImportError: If the ``mcp`` extra is not installed.
        ValueError: If the server offers no tools, or not the ones in ``names``.
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as e:
        raise ImportError("MCP tools need `pip install agentic-architectures[mcp]`") from e

    connection: dict[str, Any] = {"url": url, "transport": transport}
    if headers:
        connection["headers"] = headers
    client = MultiServerMCPClient({"server": connection})  # type: ignore[dict-item]
    tools = _run(client.get_tools())
    offered = sorted(t.name for t in tools)
    if names is not None:
        missing = sorted(set(names) - set(offered))
        if missing:
            raise ValueError(f"{url} has no tool named {missing}; it offers {offered}")
        tools = [t for t in tools if t.name in names]
    if not tools:
        raise ValueError(f"{url} offers no tools")
    return [_with_sync_call(t) for t in tools]


def _run(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion from synchronous code, event loop or not."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # A loop is already running in this thread (Jupyter, or an async caller that used the
    # sync API): asyncio.run would refuse, so give the coroutine a loop of its own.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _with_sync_call(tool: BaseTool) -> BaseTool:
    """The same tool, callable with ``invoke`` as well as ``ainvoke``."""
    from langchain_core.tools import StructuredTool

    if not isinstance(tool, StructuredTool) or tool.func is not None or tool.coroutine is None:
        return tool
    coroutine = tool.coroutine

    def call(**kwargs: Any) -> Any:
        return _run(coroutine(**kwargs))

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=call,
        coroutine=coroutine,
        response_format=tool.response_format,
        metadata=tool.metadata,
    )
