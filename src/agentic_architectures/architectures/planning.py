"""Planning — decompose the task upfront, then execute the plan step-by-step.

A.k.a. *Plan-and-Execute* (LangGraph tutorial pattern) or *Plan-Execute-Replan*.
The agent generates an explicit ordered plan **before** taking any action;
each step is then handed to an executor (a sub-agent with tools); after the
last step, a replanner decides whether to finalize or extend the plan.

Why this beats ReAct (notebook 03) for the right tasks:
  - One bird's-eye-view planning pass usually beats N greedy local decisions
    when the task has natural structure (write a report, prep a meal, run a
    benchmark).
  - The plan is a *contract* — easy to inspect, modify, or replace.

Why ReAct beats this for the wrong tasks:
  - For simple "look up one fact" questions, planning is pure overhead.
  - When the task can't be decomposed in advance (open-ended exploration),
    planning is harmful — you commit to bad sub-goals.

Origin: long lineage; the modern LangGraph idiom is documented at
https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


# ---------------------------------------------------------------------------
# Structured-output schemas
# ---------------------------------------------------------------------------
class Plan(BaseModel):
    """Ordered, actionable steps to solve the task."""

    steps: list[str] = Field(
        description=(
            "Ordered list of 3-7 atomic, actionable steps. Each step must be "
            "executable on its own given the previous steps' results. Avoid "
            "vague verbs like 'analyze' — say 'compute X from Y' or 'look up Z'."
        ),
        min_length=1,
    )


class ReplanDecision(BaseModel):
    """Decision after executing all currently-planned steps."""

    is_done: bool = Field(
        description=(
            "True iff the executed steps' results contain enough information "
            "to produce the final answer."
        )
    )
    final_response: str | None = Field(
        default=None,
        description="The final answer to the user. Required iff is_done is True.",
    )
    additional_steps: list[str] | None = Field(
        default=None,
        description=(
            "Additional steps to extend the plan. Required iff is_done is False. "
            "Do NOT repeat steps already executed."
        ),
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class PlanningState(TypedDict, total=False):
    input: str
    plan: list[str]
    past_steps: Annotated[list[tuple[str, str]], operator.add]
    response: str
    replan_count: int


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class Planning(Architecture):
    """Plan-Execute-Replan loop."""

    name = "planning"
    description = (
        "Decompose the task into an ordered plan upfront, execute each step "
        "with a sub-agent, then optionally replan based on results."
    )
    reference = "https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/"

    def __init__(
        self,
        tools: list[Any] | None = None,
        max_replans: int = 2,
        executor_rounds: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if tools is None:
            from agentic_architectures.tools.search import web_search_tool

            tools = [web_search_tool(max_results=4)]
        self.tools = tools
        self.max_replans = max_replans
        self.executor_rounds = executor_rounds

        self._planner = self.llm.with_structured_output(Plan)
        self._replanner = self.llm.with_structured_output(ReplanDecision)

        # Executor is a fresh ReAct sub-agent — composability in action.
        from agentic_architectures.architectures.tool_use import ToolUse

        self._executor = ToolUse(
            llm=self.llm.bind_tools(tools) if False else self.llm,  # passed through
            tools=tools,
            max_rounds=executor_rounds,
        )

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _plan(self, state: PlanningState) -> dict[str, Any]:
        prompt = (
            "Decompose the following task into a short ordered plan (3-7 atomic "
            "steps). Each step should make progress and be executable on its "
            "own using web search or general reasoning.\n\n"
            f"Task: {state['input']}"
        )
        plan = self._planner.invoke(prompt)
        return {"plan": list(plan.steps), "past_steps": [], "replan_count": 0}

    def _execute(self, state: PlanningState) -> dict[str, Any]:
        next_step = state["plan"][0]
        plan_remaining = state["plan"]
        history = state.get("past_steps", [])

        context_block = ""
        if history:
            context_block = "Previously executed steps and their results:\n" + "\n".join(
                f"  {i+1}. {s}\n     → {(r or '')[:300]}{'…' if r and len(r) > 300 else ''}"
                for i, (s, r) in enumerate(history)
            ) + "\n\n"
        plan_block = "Full remaining plan:\n" + "\n".join(
            f"  {i+1}. {s}" for i, s in enumerate(plan_remaining)
        )

        executor_input = (
            f"You are executing step {len(history) + 1} of a larger plan.\n\n"
            f"Original task: {state['input']}\n\n"
            f"{context_block}{plan_block}\n\n"
            f"Now execute *only* this step (do not start later steps):\n"
            f"  → {next_step}\n\n"
            "Return the result of this step concisely (~3 sentences with sources if applicable)."
        )
        result = self._executor.run(executor_input)

        return {
            "plan": plan_remaining[1:],
            "past_steps": [(next_step, result.output)],
        }

    def _replan(self, state: PlanningState) -> dict[str, Any]:
        history = state.get("past_steps", [])
        plan_remaining = state.get("plan", [])
        replan_count = state.get("replan_count", 0)

        # If steps are still queued, just keep executing — no replan needed.
        if plan_remaining:
            return {}

        # Forced finalisation after max_replans extensions.
        force_finalise = replan_count >= self.max_replans

        history_block = "\n".join(
            f"  {i+1}. {s}\n     → {(r or '')[:400]}{'…' if r and len(r) > 400 else ''}"
            for i, (s, r) in enumerate(history)
        )
        prompt = (
            f"Original task: {state['input']}\n\n"
            f"Steps executed so far:\n{history_block}\n\n"
            f"Replan budget used: {replan_count}/{self.max_replans}.\n"
            + (
                "Replan budget is exhausted — you MUST set is_done=True and "
                "produce a final_response from the evidence above.\n\n"
                if force_finalise
                else "Decide: is the task complete? If so, set is_done=True and "
                "provide final_response. Otherwise set is_done=False and provide "
                "additional_steps (do NOT repeat already-executed steps).\n\n"
            )
            + "Be honest: if the evidence is weak, still finalise rather than spinning."
        )

        decision = self._replanner.invoke(prompt)

        if decision.is_done or force_finalise:
            return {
                "response": (
                    decision.final_response
                    or self._synthesize_from_history(state["input"], history)
                )
            }

        # Extend the plan and increment replan count.
        more = list(decision.additional_steps or [])
        return {"plan": more, "replan_count": replan_count + 1}

    def _synthesize_from_history(
        self, task: str, history: list[tuple[str, str]]
    ) -> str:
        """Fallback synthesis when the replanner forgets to fill final_response."""
        prompt = (
            f"Task: {task}\n\n"
            "Executed steps and results:\n"
            + "\n".join(f"  - {s}: {r}" for s, r in history)
            + "\n\nWrite a concise final answer with source URLs."
        )
        return str(self.llm.invoke(prompt).content)

    # ------------------------------------------------------------------ #
    #  Router                                                             #
    # ------------------------------------------------------------------ #

    def _route_after_replan(self, state: PlanningState) -> str:
        if state.get("response"):
            return "end"
        if state.get("plan"):
            return "execute"
        return "end"

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(PlanningState)
        g.add_node("plan", self._plan)
        g.add_node("execute", self._execute)
        g.add_node("replan", self._replan)
        g.add_edge(START, "plan")
        g.add_edge("plan", "execute")
        g.add_edge("execute", "replan")
        g.add_conditional_edges(
            "replan",
            self._route_after_replan,
            {"execute": "execute", "end": END},
        )
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        # plan(1) + N×(execute+replan)(2) + maybe 1 extra edge = 2N+3 nodes
        # With max_steps loosely bounded by initial plan + extensions:
        n_max = 12  # safe headroom — each execute itself contains a sub-graph
        config = {"recursion_limit": 3 * n_max + 5}
        final_state = graph.invoke({"input": task}, config=config)

        history = final_state.get("past_steps", [])
        trace = [
            {"type": "plan_step", "step": s, "result": r}
            for s, r in history
        ]
        return ArchitectureResult(
            output=final_state.get("response", "")
            or self._synthesize_from_history(task, history),
            state={"input": task, "remaining_plan": final_state.get("plan", [])},
            trace=trace,
            metadata={
                "steps_executed": len(history),
                "replans": final_state.get("replan_count", 0),
                "max_replans": self.max_replans,
            },
        )
