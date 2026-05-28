"""Meta-Controller — an LLM router that picks WHICH architecture to use per task.

Multi-Agent (notebook 05) routes between specialised *agents*. Meta-Controller
routes between specialised *architectures*: ReAct for multi-hop research,
Planning for decomposable tasks, Reflection for quality-sensitive writing,
ToolUse for one-shot lookups, Mental Loop for deliberation, etc.

This is the **most composable** architecture in the repo — it treats every
other architecture as a black-box callable, exploiting the common
`Architecture.run(task)` contract.

When to reach for it: when you have a set of users / tasks with **diverse
shapes** and one fixed architecture is the wrong default for some of them.

Cost: 1 routing LLM call + 1 architecture-execution (which itself may make
many LLM calls). Generally cheaper than running every architecture in parallel
(Ensemble, notebook 13).
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import Field, create_model

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


# ---------------------------------------------------------------------------
# Default architecture roster — name → factory function returning an Architecture
# ---------------------------------------------------------------------------
def _default_roster(llm: Any) -> dict[str, Architecture]:
    from agentic_architectures.architectures.planning import Planning
    from agentic_architectures.architectures.react import ReAct
    from agentic_architectures.architectures.reflection import Reflection
    from agentic_architectures.architectures.tool_use import ToolUse

    return {
        "tool_use": ToolUse(llm=llm, max_rounds=3),
        "react": ReAct(llm=llm, max_rounds=3),
        "planning": Planning(llm=llm, max_replans=1, executor_rounds=3),
        "reflection": Reflection(llm=llm, max_iterations=2, target_score=9),
    }


# Descriptions the router sees when deciding
ARCHITECTURE_DESCRIPTIONS: dict[str, str] = {
    "tool_use": (
        "One-shot fact lookup using web search. Best for: 'What is X?' / "
        "'Who founded Y?' / 'Current price of Z?'. Single tool call usually suffices."
    ),
    "react": (
        "Multi-hop research with explicit Thought before each Action. Best for: "
        "questions that require chaining multiple lookups where each step "
        "depends on the previous result (e.g. 'When did the CEO of X's company "
        "join Y?')."
    ),
    "planning": (
        "Decompose a task into ordered steps upfront, execute each. Best for: "
        "comparisons, multi-aspect reports, anything with obvious decomposition "
        "(e.g. 'Compare X and Y on dimensions A, B, C')."
    ),
    "reflection": (
        "Generate → critique → refine loop. NO external tools. Best for: "
        "quality-sensitive creative or code tasks where the model should "
        "iterate (e.g. 'Write a Python function that ...', 'Polish this paragraph')."
    ),
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class MetaControllerState(TypedDict, total=False):
    task: str
    chosen_arch: str
    routing_reason: str
    sub_result: dict[str, Any]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class MetaController(Architecture):
    """LLM router that picks the right architecture per task."""

    name = "meta_controller"
    description = (
        "Routes each incoming task to the most appropriate architecture "
        "(ToolUse / ReAct / Planning / Reflection) based on task shape. "
        "Each sub-architecture is reused unchanged — Meta-Controller is the "
        "first architecture in the repo that composes others as black boxes."
    )
    reference = "Common pattern; canonical example: LangChain RouterChain"

    def __init__(
        self,
        roster: dict[str, Architecture] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.roster: dict[str, Architecture] = roster or _default_roster(self.llm)

        # Dynamic Literal type from roster keys so router output is constrained.
        names = tuple(self.roster.keys())
        next_t = Literal[names]  # type: ignore[valid-type]
        self._RouterDecision = create_model(
            "RouterDecision",
            chosen_arch=(
                next_t,
                Field(
                    description=(
                        "Which architecture is the best fit for this task? "
                        "Pick exactly one from the roster. See the descriptions "
                        "below to choose."
                    )
                ),
            ),
            reason=(
                str,
                Field(description="One sentence explaining the routing choice."),
            ),
        )
        self._router_llm = self.llm.with_structured_output(self._RouterDecision)

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _route(self, state: MetaControllerState) -> dict[str, Any]:
        descriptions = "\n".join(
            f"  - **{name}**: {ARCHITECTURE_DESCRIPTIONS.get(name, '(no description)')}"
            for name in self.roster
        )
        prompt = (
            "You are a Meta-Controller routing an incoming task to the most "
            "appropriate agentic architecture.\n\n"
            f"## Available architectures\n{descriptions}\n\n"
            f"## Task\n{state['task']}\n\n"
            "Pick the SINGLE best-fit architecture and explain in ONE sentence why."
        )
        decision = self._router_llm.invoke(prompt)
        return {
            "chosen_arch": decision.chosen_arch,
            "routing_reason": decision.reason,
        }

    def _execute(self, state: MetaControllerState) -> dict[str, Any]:
        arch_name = state["chosen_arch"]
        sub_arch = self.roster[arch_name]
        sub_result = sub_arch.run(state["task"])
        return {
            "sub_result": {
                "output": sub_result.output,
                "metadata": sub_result.metadata,
                "architecture": arch_name,
            }
        }

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(MetaControllerState)
        g.add_node("route", self._route)
        g.add_node("execute", self._execute)
        g.add_edge(START, "route")
        g.add_edge("route", "execute")
        g.add_edge("execute", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke({"task": task})
        sub = final_state.get("sub_result", {})
        return ArchitectureResult(
            output=sub.get("output", ""),
            state={
                "chosen_arch": final_state.get("chosen_arch", ""),
                "routing_reason": final_state.get("routing_reason", ""),
            },
            trace=[
                {
                    "type": "route",
                    "chosen": final_state.get("chosen_arch", ""),
                    "reason": final_state.get("routing_reason", ""),
                },
                {
                    "type": "sub_arch_run",
                    "architecture": sub.get("architecture", ""),
                    "output_preview": sub.get("output", "")[:200],
                    "sub_metadata": sub.get("metadata", {}),
                },
            ],
            metadata={
                "chosen_arch": final_state.get("chosen_arch", ""),
                "routing_reason": final_state.get("routing_reason", ""),
                "roster_size": len(self.roster),
            },
        )
