"""Exercise the MCP wire protocol and the maintained ToolUse caller offline."""

from __future__ import annotations

import asyncio
import builtins
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException

from agentic_architectures.architectures import ReAct, ToolUse
from agentic_architectures.tools import parallel_search_tool

pytest.importorskip("mcp")
from mcp.client import streamable_http

RESPONSE = {
    "results": [{"url": "https://modelcontextprotocol.io", "excerpts": ["MCP connects AI applications to tools."]}],
    "session_id": "test-session",
}


class SearchAnswerModel:
    """Deterministic tool-calling model that builds its answer from tool output."""

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if results:
            data = json.loads(results[-1].content)
            source = data["results"][0]
            return AIMessage(content=f"{source['excerpts'][0]} Source: {source['url']}")
        return AIMessage(
            content="",
            tool_calls=[{"name": self.tools[0].name, "args": {"query": "Model Context Protocol"}, "id": "search-1"}],
        )


@pytest.fixture
def wire(monkeypatch):
    requests = []
    state = {"response": {"content": [], "structuredContent": RESPONSE}, "delay": 0, "closed": False}
    original = streamable_http.streamablehttp_client

    async def handle(request):
        requests.append(request)
        body = json.loads(request.content)
        if "id" not in body:
            return httpx.Response(202)
        if body["method"] == "initialize":
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-search", "version": "1"},
            }
        elif body["method"] == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        else:
            assert body["method"] == "tools/call"
            await asyncio.sleep(state["delay"])
            result = state["response"]
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    @asynccontextmanager
    async def transport(url, **kwargs):
        def client_factory(headers=None, timeout=None, auth=None):
            return httpx.AsyncClient(headers=headers, timeout=timeout, auth=auth, transport=httpx.MockTransport(handle))

        try:
            async with original(url, **kwargs, httpx_client_factory=client_factory) as streams:
                yield streams
        finally:
            state["closed"] = True

    monkeypatch.setattr(streamable_http, "streamablehttp_client", transport)
    return requests, state


def test_sync_search_wire_and_stable_session(wire):
    requests, _ = wire
    tool = parallel_search_tool()
    assert tool.invoke({"query": "MCP specification"}) == RESPONSE
    assert tool.invoke({"query": "MCP tools"}) == RESPONSE
    calls = [json.loads(r.content) for r in requests if json.loads(r.content)["method"] == "tools/call"]
    args = calls[0]["params"]["arguments"]
    assert calls[0]["params"]["name"] == "web_search"
    assert args["objective"] == "MCP specification"
    assert args["search_queries"] == ["MCP specification"]
    assert len(args["session_id"]) == 32
    assert args["session_id"] == calls[1]["params"]["arguments"]["session_id"]
    for request in requests:
        assert str(request.url) == "https://search.parallel.ai/mcp"
        assert request.headers["user-agent"] == "agentic-architectures/0.3.0"
        assert "authorization" not in request.headers


async def test_async_search_and_active_loop_guidance(wire):
    tool = parallel_search_tool()
    assert await tool.ainvoke({"query": "MCP"}) == RESPONSE
    with pytest.raises(RuntimeError, match="ainvoke"):
        tool.invoke({"query": "MCP"})


@pytest.mark.parametrize(
    "response,match",
    [
        ({"content": [{"type": "text", "text": "rate limited"}], "isError": True}, "rate limited"),
        ({"content": []}, "no structured results"),
    ],
)
def test_mcp_errors_propagate(wire, response, match):
    wire[1]["response"] = response
    with pytest.raises(ToolException, match=match):
        parallel_search_tool().invoke({"query": "MCP"})


async def test_timeout_cancels_session(wire):
    wire[1]["delay"] = 1
    with pytest.raises(TimeoutError):
        await parallel_search_tool(timeout=0.05).ainvoke({"query": "MCP"})
    assert wire[1]["closed"]


async def test_caller_cancellation_closes_session(wire):
    wire[1]["delay"] = 1
    task = asyncio.create_task(parallel_search_tool().ainvoke({"query": "MCP"}))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert wire[1]["closed"]


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        parallel_search_tool(timeout=timeout)


def test_missing_optional_dependency(monkeypatch):
    original = builtins.__import__

    def without_mcp(name, *args, **kwargs):
        if name == "mcp":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_mcp)
    with pytest.raises(ImportError, match=r"agentic-architectures\[parallel\]"):
        parallel_search_tool()


@pytest.mark.parametrize("architecture", [ToolUse, ReAct])
def test_architecture_caller_returns_cited_answer(wire, architecture):
    agent = architecture(llm=SearchAnswerModel(), tools=[parallel_search_tool()])
    result = agent.run("What is MCP?")
    assert result.output == "MCP connects AI applications to tools. Source: https://modelcontextprotocol.io"
    assert result.metadata["tools_used"] == ["parallel_web_search"]
    assert result.metadata["tool_calls"] == 1
    assert any(t["type"] == "tool_result" for t in result.trace)


@pytest.mark.parametrize("architecture", [ToolUse, ReAct])
def test_default_search_still_uses_tavily(monkeypatch, architecture):
    from agentic_architectures.tools import search

    incumbent = parallel_search_tool()  # sentinel only; no network call
    selected = []

    def tavily_factory(**kwargs):
        selected.append(kwargs)
        return incumbent

    monkeypatch.setattr(search, "web_search_tool", tavily_factory)
    agent = architecture(llm=SearchAnswerModel())
    assert selected == [{"max_results": 5}]
    assert agent.tools == [incumbent]
