"""Tool Use — give the agent access to external tools (web search by default).

The simplest "agentic" pattern that goes beyond a single forward pass: the LLM
can decide to *call* an external tool, read the tool's output, then either call
another tool or produce a final answer.

Origin: the *function/tool calling* APIs introduced by OpenAI (June 2023) and now
universal across providers. Conceptual ancestor is Toolformer (Schick et al., 2023).

Distinct from **ReAct** (notebook 03), which adds an explicit *thought* phase
before each action. Tool Use is "act-only"; ReAct is "think then act".

State machine
-------------
    [agent] -→ tool_calls present? -→ [tools] -→ [agent]
              ↓                                       ↑
              └─── no tool calls → END  ──────────────┘
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agentic_architectures.architectures.base import Architecture, ArchitectureResult
from agentic_architectures.llm.factory import provider_supports_tools


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ToolUseState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class ToolUse(Architecture):
    """Tool-calling loop. The agent decides when to use a tool and when to stop."""

    name = "tool_use"
    description = (
        "Give the LLM external tools (default: Tavily web search). It decides when to "
        "call them, reads the results, and either calls another tool or produces a "
        "final answer."
    )
    reference = "https://platform.openai.com/docs/guides/function-calling"

    DEFAULT_SYSTEM_PROMPT = (
        "You are a research assistant with access to web search.\n\n"
        "Rules:\n"
        "1. Use the search tool only when you need facts you don't already know.\n"
        "2. After at most 2-3 searches, STOP searching and answer using what you found.\n"
        "3. Cite your sources with URLs in the final answer.\n"
        "4. If a search returns enough information, do NOT search again - answer the user."
    )

    def __init__(
        self,
        tools: list[Any] | None = None,
        max_rounds: int = 6,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not provider_supports_tools():
            raise RuntimeError(
                "Tool Use requires a provider with tool-calling support. "
                "Switch LLM_PROVIDER to one of: nebius, openai, anthropic, groq, "
                "together, fireworks, mistralai, google, ollama."
            )

        if tools is None:
            from agentic_architectures.tools.search import web_search_tool

            tools = [web_search_tool(max_results=5)]
        self.tools = tools
        self.max_rounds = max_rounds
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._llm_with_tools = self.llm.bind_tools(self.tools)

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _agent(self, state: ToolUseState) -> dict[str, Any]:
        # Count tool calls already made in the conversation.
        tool_call_count = sum(
            len(getattr(m, "tool_calls", []) or [])
            for m in state["messages"]
        )
        # Hard cap: once we exceed max_rounds tool calls, invoke WITHOUT tools.
        # This forces the model to produce a final text answer.
        over_budget = tool_call_count >= self.max_rounds
        llm = self.llm if over_budget else self._llm_with_tools

        msgs = state["messages"]
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=self.system_prompt), *msgs]
        if over_budget:
            # Append a stern reminder so the model definitely answers.
            from langchain_core.messages import HumanMessage as _HM

            msgs = [*msgs, _HM(content=(
                f"You have used your search budget ({self.max_rounds} calls). "
                "Now answer the user's question using ONLY the information already "
                "in this conversation. Do NOT request more searches."
            ))]

        response = llm.invoke(msgs)
        new_messages: list[Any] = []
        if not state["messages"] or not isinstance(state["messages"][0], SystemMessage):
            new_messages.append(SystemMessage(content=self.system_prompt))
        new_messages.append(response)
        return {"messages": new_messages}

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(ToolUseState)
        g.add_node("agent", self._agent)
        g.add_node("tools", ToolNode(self.tools))

        g.add_edge(START, "agent")
        g.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "tools", END: END},
        )
        g.add_edge("tools", "agent")
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        # Generous budget — many models over-search even with a strict system prompt.
        # Each tool round = 2 nodes (agent + tools), plus the final agent. We add 6× headroom.
        config = {"recursion_limit": max(40, 6 * self.max_rounds + 10)}
        final_state = graph.invoke(
            {"messages": [HumanMessage(content=task)]},
            config=config,
        )
        messages = final_state["messages"]

        trace = _messages_to_trace(messages)
        tool_calls = [t for t in trace if t["type"] == "tool_call"]
        return ArchitectureResult(
            output=messages[-1].content if messages else "",
            state={"message_count": len(messages)},
            trace=trace,
            metadata={
                "tool_calls": len(tool_calls),
                "tools_used": sorted({t["tool"] for t in tool_calls if t.get("tool")}),
                "rounds": sum(1 for t in trace if t["type"] == "agent"),
            },
        )


def _messages_to_trace(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    """Flatten messages into a list of trace events the notebook can render."""
    trace: list[dict[str, Any]] = []
    for m in messages:
        role = getattr(m, "type", "") or m.__class__.__name__.lower()
        if role in {"human", "humanmessage"}:
            trace.append({"type": "user", "content": m.content})
        elif role in {"ai", "aimessage"}:
            # Explicit Thought marker (set by the `think` node in ReAct & friends).
            if getattr(m, "additional_kwargs", {}).get("react_step") == "thought":
                trace.append({"type": "thought", "content": str(m.content)})
                continue
            tool_calls = getattr(m, "tool_calls", None) or []
            if tool_calls:
                # Also capture any "implicit" thought (content alongside a tool call).
                content = (m.content if isinstance(m.content, str) else "").strip()
                if content:
                    trace.append({"type": "thought", "content": content})
                for tc in tool_calls:
                    args = tc.get("args", tc) if isinstance(tc, dict) else tc.args
                    name = tc.get("name", "?") if isinstance(tc, dict) else tc.name
                    trace.append(
                        {
                            "type": "tool_call",
                            "tool": name,
                            "args": args,
                        }
                    )
            else:
                trace.append({"type": "agent", "content": m.content})
        elif role in {"tool", "toolmessage"}:
            content = m.content if isinstance(m.content, str) else str(m.content)
            trace.append(
                {
                    "type": "tool_result",
                    "tool": getattr(m, "name", "?"),
                    "content": content,
                }
            )
    return trace
