"""ReAct — Reason + Act in alternation, with Thought architecturally guaranteed.

Naive implementations of ReAct on modern tool-calling APIs (including
`langgraph.prebuilt.create_react_agent`) place both the Thought and the
Action into a single AIMessage — the Thought in `content`, the Action in
`tool_calls`. The problem: many models (Llama 3.x, smaller OSS models)
happily emit a tool call with empty `content`, which silently degrades
ReAct into plain Tool Use (see notebook 02).

This implementation **guarantees** a Thought before every Action by
splitting the loop into two explicit nodes:

  - `think` runs the LLM **without** tools and asks for ONLY a Thought.
  - `act` runs the LLM **with** tools and chooses one tool call or the
    final answer.

Cost: 2 LLM calls per ReAct round instead of 1, but the Thought channel
cannot be skipped by the model.

Origin: Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*,
ICLR 2023 ([arXiv:2210.03629](https://arxiv.org/abs/2210.03629)).
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agentic_architectures.architectures.base import Architecture, ArchitectureResult
from agentic_architectures.architectures.tool_use import ToolUseState, _messages_to_trace
from agentic_architectures.llm.factory import provider_supports_tools


class ReAct(Architecture):
    """Explicit two-node ReAct loop: think → act → (tools → think)*  → END."""

    name = "react"
    description = (
        "Reason + Act in alternation. A dedicated `think` node produces a Thought "
        "without tool access; a separate `act` node makes the Thought-informed "
        "tool call (or finalises). Architecturally guarantees a Thought before "
        "every Action — unlike `create_react_agent`, which lets the model skip it."
    )
    reference = "https://arxiv.org/abs/2210.03629"

    DEFAULT_SYSTEM_PROMPT = (
        "You are a research assistant solving the user's task by alternating "
        "Thoughts and Actions. Always cite source URLs in the final answer."
    )

    THINK_INSTRUCTION = (
        "Produce EXACTLY ONE short paragraph beginning with the literal word "
        "'Thought:' that (a) reflects on what you have learned from the "
        "conversation so far, and (b) states the specific next step needed to "
        "advance toward an answer. Do NOT call any tools in this step — only "
        "produce the Thought."
    )

    ACT_INSTRUCTION = (
        "Based on your latest Thought above, take exactly ONE action:\n"
        "  (a) Call exactly ONE tool whose arguments implement your Thought, OR\n"
        "  (b) Write the final answer (no tool call) if you already have enough "
        "evidence. Include source URLs in the final answer."
    )

    def __init__(
        self,
        tools: list[Any] | None = None,
        max_rounds: int = 5,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not provider_supports_tools():
            raise RuntimeError(
                "ReAct requires a provider with tool-calling support. "
                "Switch LLM_PROVIDER to one that does."
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

    def _think(self, state: ToolUseState) -> dict[str, Any]:
        # Inline the instructions WITHOUT persisting them to state (so the
        # message history stays clean — only the actual Thought is kept).
        msgs = [
            SystemMessage(content=self.system_prompt),
            *state["messages"],
            HumanMessage(content=self.THINK_INSTRUCTION),
        ]
        # No tools bound — the model cannot make a tool call here.
        response = self.llm.invoke(msgs)
        thought = AIMessage(
            content=str(response.content),
            additional_kwargs={"react_step": "thought"},
        )
        return {"messages": [thought]}

    def _act(self, state: ToolUseState) -> dict[str, Any]:
        msgs = [
            SystemMessage(content=self.system_prompt),
            *state["messages"],
            HumanMessage(content=self.ACT_INSTRUCTION),
        ]
        response = self._llm_with_tools.invoke(msgs)
        return {"messages": [response]}

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(ToolUseState)
        g.add_node("think", self._think)
        g.add_node("act", self._act)
        g.add_node("tools", ToolNode(self.tools))

        g.add_edge(START, "think")
        g.add_edge("think", "act")
        g.add_conditional_edges(
            "act",
            tools_condition,
            {"tools": "tools", END: END},
        )
        g.add_edge("tools", "think")
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        # Each round = think + act + (maybe tools) = 3 node visits.
        # Generous budget — Llama-style models often want to keep searching.
        config = {"recursion_limit": max(40, 6 * self.max_rounds + 10)}
        final_state = graph.invoke(
            {"messages": [HumanMessage(content=task)]},
            config=config,
        )
        messages = final_state["messages"]

        trace = _messages_to_trace(messages)
        tool_calls = [t for t in trace if t["type"] == "tool_call"]
        thoughts = [t for t in trace if t["type"] == "thought"]

        return ArchitectureResult(
            output=messages[-1].content if messages else "",
            state={"message_count": len(messages)},
            trace=trace,
            metadata={
                "tool_calls": len(tool_calls),
                "tools_used": sorted({t["tool"] for t in tool_calls if t.get("tool")}),
                "thought_count": len(thoughts),
                "rounds": sum(1 for t in trace if t["type"] == "agent"),
            },
        )
