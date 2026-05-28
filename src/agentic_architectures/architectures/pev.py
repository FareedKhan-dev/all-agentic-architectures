"""PEV — Plan-Execute-Verify with per-step retry-on-failure.

Planning (notebook 04) trusts every step's result and only re-plans when the
*whole plan* runs out. That's fine if your tools are reliable. PEV adds a
**Verifier** between Execute and Accept: each step's result is judged against
the step's intent, and a failure triggers a *retry with critique* on the same
step. After `max_retries_per_step` failures the step is force-accepted (with
a `fail-accepted` verdict) so the plan can still finish.

When to reach for PEV instead of Planning:
  - Tools are flaky (network errors, rate limits, partial results).
  - High-stakes domains where an unverified step is dangerous.
  - You want to *measure* per-step success rate as a quality signal.

State machine
-------------
    plan → execute → verify ─┬─ pass + more steps ─→ execute
                              ├─ pass + done ──────→ finalize
                              ├─ fail + retries left → execute (retry w/ critique)
                              └─ fail + budget gone → finalize (fail-accepted)
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult
from agentic_architectures.architectures.planning import Plan  # reuse the schema
from agentic_architectures.evaluators import LLMJudge


# ---------------------------------------------------------------------------
# Verifier rubric
# ---------------------------------------------------------------------------
class _StepVerification(BaseModel):
    """Verifier's verdict on a single executed plan step."""

    is_satisfactory: bool = Field(
        description=(
            "True iff the step's result fully addresses the step's intent — "
            "concretely, contains the requested fact / computation / artifact "
            "and is grounded in evidence (cited URL or computation shown)."
        )
    )
    issues: str | None = Field(
        default=None,
        description=(
            "If is_satisfactory is False, describe concrete issues "
            "(missing fact, ungrounded claim, wrong format). One short paragraph. "
            "Required iff is_satisfactory is False."
        ),
    )
    confidence: int = Field(
        ge=1,
        le=5,
        description="Confidence in this verdict, 1 (low) to 5 (high).",
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class PEVState(TypedDict, total=False):
    input: str
    plan: list[str]
    past_steps: Annotated[list[dict[str, Any]], operator.add]
    # Per-step scratchpad (cleared on accept):
    pending_step: str
    pending_result: str
    pending_critique: str
    attempts: int
    response: str


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class PEV(Architecture):
    """Plan-Execute-Verify with per-step retry-with-critique."""

    name = "pev"
    description = (
        "Planning + a Verifier that gates each step. Failed steps trigger a "
        "retry on the SAME step with the verifier's critique attached, up to "
        "max_retries_per_step. The plan continues even if a step exhausts retries."
    )
    reference = "https://arxiv.org/abs/2305.10142"

    def __init__(
        self,
        tools: list[Any] | None = None,
        max_retries_per_step: int = 2,
        executor_rounds: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if tools is None:
            from agentic_architectures.tools.search import web_search_tool

            tools = [web_search_tool(max_results=4)]
        self.tools = tools
        self.max_retries_per_step = max_retries_per_step
        self.executor_rounds = executor_rounds

        from agentic_architectures.architectures.tool_use import ToolUse

        # Strict per-step prompt: PEV steps are atomic, so 1-2 searches suffice.
        self._executor = ToolUse(
            llm=self.llm,
            tools=tools,
            max_rounds=executor_rounds,
            system_prompt=(
                "You are executing one atomic step of a larger plan. "
                "Use the search tool AT MOST ONCE per step (twice only if the "
                "first result was empty). After your search, immediately answer "
                "with the specific fact/value requested and a source URL. "
                "Do NOT preamble, do NOT explain — just the requested fact and a URL."
            ),
        )
        self._planner = self.llm.with_structured_output(Plan)
        self._verifier: LLMJudge[_StepVerification] = LLMJudge(
            schema=_StepVerification,
            rubric=(
                "Judge whether the executed step's result fully addresses the "
                "step's intent given the original task. "
                "A satisfactory result must (1) contain the specific fact / "
                "computation / artifact the step asks for, and (2) be grounded "
                "(cite a URL OR show the explicit computation)."
            ),
            llm=self.llm,
        )

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _plan(self, state: PEVState) -> dict[str, Any]:
        prompt = (
            "Decompose this task into 3-6 atomic, verifiable steps. Each step "
            "must produce a concrete fact, value, or artifact that can be checked.\n\n"
            f"Task: {state['input']}"
        )
        plan = self._planner.invoke(prompt)
        return {
            "plan": list(plan.steps),
            "past_steps": [],
            "pending_step": "",
            "pending_critique": "",
            "attempts": 0,
        }

    def _execute(self, state: PEVState) -> dict[str, Any]:
        attempts = state.get("attempts", 0)
        is_retry = attempts > 0 and bool(state.get("pending_step"))

        if is_retry:
            step = state["pending_step"]
            critique = state.get("pending_critique", "")
            plan_unchanged = state.get("plan", [])
            sub_task = (
                f"Original overall task: {state['input']}\n\n"
                f"You are RETRYING this step (attempt {attempts + 1}):\n  → {step}\n\n"
                f"Your previous attempt was rejected by the Verifier. Critique:\n"
                f"  > {critique}\n\n"
                "Re-execute the step addressing the critique. Be concrete and grounded."
            )
        else:
            step = state["plan"][0]
            plan_unchanged = state["plan"][1:]
            sub_task = (
                f"Original overall task: {state['input']}\n\n"
                f"Execute this step (return result with sources/computation):\n  → {step}"
            )

        result = self._executor.run(sub_task)
        return {
            "plan": plan_unchanged,
            "pending_step": step,
            "pending_result": result.output,
            "attempts": attempts + 1,
        }

    def _verify(self, state: PEVState) -> dict[str, Any]:
        verdict = self._verifier.evaluate(
            candidate=state.get("pending_result", ""),
            context={
                "step": state.get("pending_step", ""),
                "original_task": state.get("input", ""),
            },
        )

        attempts = state.get("attempts", 1)
        if verdict.is_satisfactory:
            # Accept and commit
            return {
                "past_steps": [
                    {
                        "step": state["pending_step"],
                        "result": state["pending_result"],
                        "verdict": "pass",
                        "attempts": attempts,
                        "confidence": verdict.confidence,
                    }
                ],
                "pending_step": "",
                "pending_critique": "",
                "attempts": 0,
            }

        # Failed.
        if attempts < self.max_retries_per_step + 1:
            # Keep pending — _execute will retry with critique.
            return {"pending_critique": verdict.issues or "(no critique provided)"}

        # Budget exhausted — force-accept with fail verdict so plan moves on.
        return {
            "past_steps": [
                {
                    "step": state["pending_step"],
                    "result": state["pending_result"],
                    "verdict": "fail-accepted",
                    "attempts": attempts,
                    "confidence": verdict.confidence,
                    "last_critique": verdict.issues or "",
                }
            ],
            "pending_step": "",
            "pending_critique": "",
            "attempts": 0,
        }

    def _finalize(self, state: PEVState) -> dict[str, Any]:
        history = state.get("past_steps", [])
        rows = "\n".join(f"  - [{p['verdict']}] {p['step']}\n     → {(p['result'] or '')[:300]}" for p in history)
        prompt = (
            f"Original task: {state['input']}\n\n"
            f"Verified plan execution log:\n{rows}\n\n"
            "Write a concise final answer. Be honest about any steps that "
            "ended with verdict 'fail-accepted' — note that data point is "
            "lower-confidence. Preserve all source URLs."
        )
        return {"response": str(self.llm.invoke(prompt).content)}

    # ------------------------------------------------------------------ #
    #  Router                                                             #
    # ------------------------------------------------------------------ #

    def _route_after_verify(self, state: PEVState) -> str:
        # If we still have a pending step + critique, that's a retry signal.
        if state.get("pending_step") and state.get("pending_critique"):
            return "execute"
        # Step committed (pass or fail-accepted). What's next?
        if state.get("plan"):
            return "execute"  # next step in plan
        return "finalize"

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(PEVState)
        g.add_node("plan", self._plan)
        g.add_node("execute", self._execute)
        g.add_node("verify", self._verify)
        g.add_node("finalize", self._finalize)
        g.add_edge(START, "plan")
        g.add_edge("plan", "execute")
        g.add_edge("execute", "verify")
        g.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {"execute": "execute", "finalize": "finalize"},
        )
        g.add_edge("finalize", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        # Plan + (Execute + Verify) per step + retries. Generous budget.
        config = {"recursion_limit": 60}
        final_state = graph.invoke({"input": task}, config=config)

        history = final_state.get("past_steps", [])
        trace = [{"type": "step", **p} for p in history]
        return ArchitectureResult(
            output=final_state.get("response", ""),
            state={"input": task, "leftover_plan": final_state.get("plan", [])},
            trace=trace,
            metadata={
                "steps_total": len(history),
                "steps_passed": sum(1 for p in history if p["verdict"] == "pass"),
                "steps_fail_accepted": sum(1 for p in history if p["verdict"] == "fail-accepted"),
                "total_attempts": sum(p["attempts"] for p in history),
                "max_retries_per_step": self.max_retries_per_step,
            },
        )
