# Tools

::: agentic_architectures.tools.web_search_tool
::: agentic_architectures.tools.parallel_search_tool
::: agentic_architectures.tools.python_repl_tool
::: agentic_architectures.tools.SimulatorTool

## Opt-in Parallel search

Install the optional MCP client:

```bash
pip install "agentic-architectures[parallel]"
```

Parallel's free, keyless Search MCP server uses Streamable HTTP at
`https://search.parallel.ai/mcp`. Select it explicitly through the existing
architecture tools argument:

```python
from agentic_architectures.architectures import ToolUse
from agentic_architectures.tools import parallel_search_tool

tool = parallel_search_tool(timeout=60)
agent = ToolUse(tools=[tool])
result = agent.run("What is the Model Context Protocol? Cite your sources.")
print(result.output)
```

Configure a tool-capable LLM provider and its credentials as described in
[Providers](../getting-started/providers.md). The LLM still requires its usual
setup; search does not require a Parallel API key. `ReAct(tools=[tool])` also
accepts this tool. Existing Tavily defaults and configuration are unchanged.

For direct use, `tool.invoke({"query": "Model Context Protocol specification"})`
returns the server's structured response, including `results` with source URLs
and `excerpts`. In async code use
`await tool.ainvoke({"query": "Model Context Protocol specification"})`.
The tool maps `query` to the MCP `web_search` objective and one search query,
keeps a stable search session identifier per tool instance, and opens and closes
an MCP connection for each invocation. The positive, finite `timeout` cancels the
session when it expires, then awaits cleanup. MCP errors and timeouts propagate to the caller, with no fallback to
another provider. Free service rate limits may apply.
