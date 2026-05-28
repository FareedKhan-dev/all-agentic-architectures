"""Multi-Agent Systems — a supervisor routes work to role-specialised agents.

A team of agents, each with its own role (system prompt) and tools, collaborates
to solve a problem too broad for any one of them. A **supervisor** agent reads
the conversation history and routes to the next specialist — or to a final
*Writer* that synthesises all specialists' outputs.

Why a team beats one big agent:
  - **Focused context** per specialist (their system prompt narrows the search space).
  - **Division of labour** lets you mix tools — the Financial agent has access to
    a stock-quote tool, the News agent has web search, the Code agent has a REPL.
  - **Inspectability** — each specialist's output is a separately-labelled artifact
    in the trace, much easier to debug than one giant monologue.

The supervisor is the *coordination protocol*. Variations:
  - **Manager-style supervisor** (this notebook): a router that picks `next`
    from a fixed roster, plus a Writer node.
  - **Hierarchical**: supervisors-of-supervisors, used for very large teams.
  - **Blackboard** (notebook 07): no central supervisor; agents subscribe to
    shared state and self-elect.

Origin: long history in classical MAS; modern LLM versions trace to AutoGen
(Microsoft, 2023), CrewAI (2023), and LangGraph's supervisor pattern.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, create_model

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


# ---------------------------------------------------------------------------
# Default specialist roster — easily overridden via constructor
# ---------------------------------------------------------------------------
DEFAULT_SPECIALISTS: dict[str, str] = {
    "news": (
        "You are a NEWS specialist. Find recent news, announcements, "
        "press coverage. Focus on events from the last 12 months. "
        "Output: bullet list of 3-5 concrete news items with dates and source URLs."
    ),
    "technical": (
        "You are a TECHNICAL/PRODUCT specialist. Find concrete product details, "
        "specifications, releases, capabilities. Avoid marketing language. "
        "Output: bullet list of 3-5 concrete technical facts with source URLs."
    ),
    "financial": (
        "You are a FINANCIAL specialist. Find revenue, growth, market cap, "
        "stock performance, key financial metrics. Use the most recent figures. "
        "Output: bullet list of 3-5 concrete financial facts with source URLs."
    ),
}


# ---------------------------------------------------------------------------
# Supervisor decision schema — dynamic, since roster varies per architecture instance
# ---------------------------------------------------------------------------
def _build_decision_schema(specialist_names: list[str]) -> type[BaseModel]:
    """Build a Pydantic schema whose `next` field is constrained to the
    actual specialist roster + writer/FINISH."""
    allowed = tuple(specialist_names) + ("writer", "FINISH")
    next_type = Literal[allowed]  # type: ignore[valid-type]
    return create_model(
        "SupervisorDecision",
        next=(
            next_type,
            Field(
                description=(
                    "Which agent should act next? "
                    "Pick a specialist if their domain hasn't been covered yet. "
                    "Pick 'writer' once all specialists have contributed. "
                    "Pick 'FINISH' only if the writer has already produced the final report."
                )
            ),
        ),
        reason=(str, Field(description="One sentence explaining the routing choice.")),
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class MultiAgentState(TypedDict, total=False):
    task: str
    specialist_outputs: Annotated[list[dict[str, str]], operator.add]
    next: str
    next_reason: str
    final_report: str


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class MultiAgent(Architecture):
    """Supervisor + N specialists + Writer."""

    name = "multi_agent"
    description = (
        "A supervisor routes the task to role-specialised agents (each with its "
        "own system prompt + tools), then a Writer synthesises their outputs into "
        "a final report."
    )
    reference = "https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/"

    def __init__(
        self,
        specialists: dict[str, str] | None = None,
        tools: list[Any] | None = None,
        specialist_rounds: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.specialists = specialists or DEFAULT_SPECIALISTS

        if tools is None:
            from agentic_architectures.tools.search import web_search_tool

            tools = [web_search_tool(max_results=4)]
        self.tools = tools
        self.specialist_rounds = specialist_rounds

        # Each specialist is a ToolUse instance with its own role prompt.
        # IMPORTANT: append the cap-search rule so the role prompt doesn't
        # silently disable ToolUse's runaway-search protection.
        _CAP_RULE = (
            "\n\nRULES:\n"
            "  - Use the search tool at most 1-2 times.\n"
            "  - After your second search, STOP searching and answer with what you have.\n"
            "  - Always cite source URLs."
        )
        from agentic_architectures.architectures.tool_use import ToolUse

        self._specialist_agents = {
            name: ToolUse(
                llm=self.llm,
                tools=self.tools,
                max_rounds=specialist_rounds,
                system_prompt=prompt + _CAP_RULE,
            )
            for name, prompt in self.specialists.items()
        }

        self._SupervisorDecision = _build_decision_schema(list(self.specialists.keys()))
        self._supervisor_llm = self.llm.with_structured_output(self._SupervisorDecision)

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _supervisor(self, state: MultiAgentState) -> dict[str, Any]:
        outputs = state.get("specialist_outputs", [])
        covered = {o["specialist"] for o in outputs}
        remaining = [name for name in self.specialists if name not in covered]
        writer_done = bool(state.get("final_report"))

        summary = (
            f"Task: {state['task']}\n\n"
            f"Specialists who have contributed: {sorted(covered) or 'none'}\n"
            f"Specialists still available: {remaining or 'none'}\n"
            f"Writer has produced final report: {writer_done}\n\n"
            "Decide who acts next. Rules:\n"
            "  - If any specialist hasn't contributed yet, route to one of them.\n"
            "  - If all specialists have contributed and the writer hasn't, route to 'writer'.\n"
            "  - If the writer has produced the final report, route to 'FINISH'.\n"
            "  - NEVER route the same specialist twice."
        )
        decision = self._supervisor_llm.invoke(summary)
        # Safety: clamp 'next' if model misbehaves.
        if decision.next in covered:
            # picked a specialist who already acted — fall through to next remaining
            decision.next = (
                remaining[0] if remaining else ("writer" if not writer_done else "FINISH")
            )
        return {"next": decision.next, "next_reason": decision.reason}

    def _make_specialist_node(self, name: str):
        agent = self._specialist_agents[name]

        def _node(state: MultiAgentState) -> dict[str, Any]:
            sub_task = (
                f"Research task: {state['task']}\n\n"
                "Focus only on your specialty (see system prompt). Be concise."
            )
            result = agent.run(sub_task)
            return {
                "specialist_outputs": [
                    {"specialist": name, "content": result.output}
                ]
            }

        _node.__name__ = f"specialist_{name}"
        return _node

    def _writer(self, state: MultiAgentState) -> dict[str, Any]:
        outputs = state.get("specialist_outputs", [])
        sections = "\n\n".join(
            f"## {o['specialist'].upper()} findings\n{o['content']}"
            for o in outputs
        )
        heading_examples = ", ".join(
            f"### {o['specialist'].title()}" for o in outputs
        )
        prompt = (
            f"You are the team Writer. Synthesise the specialists' findings below "
            f"into a coherent ~250-word report answering this task:\n\n"
            f"Task: {state['task']}\n\n"
            f"--- Specialist findings ---\n{sections}\n\n"
            "Rules:\n"
            f"  - Use one heading per specialist: {heading_examples}.\n"
            "  - Preserve ALL source URLs from the specialists.\n"
            "  - Do NOT add facts the specialists didn't provide.\n"
            "  - End with a 1-sentence overall conclusion."
        )
        report = self.llm.invoke(prompt).content
        return {"final_report": str(report)}

    # ------------------------------------------------------------------ #
    #  Router                                                             #
    # ------------------------------------------------------------------ #

    def _route(self, state: MultiAgentState) -> str:
        nxt = state.get("next", "FINISH")
        return nxt if nxt != "FINISH" else "END"

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(MultiAgentState)
        g.add_node("supervisor", self._supervisor)
        for name in self.specialists:
            g.add_node(name, self._make_specialist_node(name))
        g.add_node("writer", self._writer)

        g.add_edge(START, "supervisor")
        # Supervisor routes to one of: specialist names, "writer", or END
        routes = {name: name for name in self.specialists}
        routes["writer"] = "writer"
        routes["END"] = END
        g.add_conditional_edges("supervisor", self._route, routes)

        for name in self.specialists:
            g.add_edge(name, "supervisor")
        g.add_edge("writer", "supervisor")
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        # Supervisor + 3 specialists + writer + supervisor again → many edges
        n_max_specialists = len(self.specialists)
        config = {"recursion_limit": 4 * n_max_specialists + 10}
        final_state = graph.invoke({"task": task}, config=config)

        outputs = final_state.get("specialist_outputs", [])
        trace = [{"type": "specialist", **o} for o in outputs]
        if final_state.get("final_report"):
            trace.append({"type": "writer", "content": final_state["final_report"]})

        return ArchitectureResult(
            output=final_state.get("final_report", ""),
            state={
                "specialists_used": sorted({o["specialist"] for o in outputs}),
                "task": task,
            },
            trace=trace,
            metadata={
                "specialists_invoked": len(outputs),
                "specialists_available": len(self.specialists),
                "has_writer_output": bool(final_state.get("final_report")),
            },
        )
