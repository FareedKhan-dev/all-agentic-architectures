"""Reflection — generate, critique, refine.

A baseline single-agent enhancement. The agent produces a draft, a Critic LLM
(usually the same model, sometimes a stronger one) scores and critiques it, then
the agent refines using the critique. Repeat until the critic is satisfied or
a budget is exhausted.

Origin: Madaan et al., *Self-Refine: Iterative Refinement with Self-Feedback* (2023).
Distinct from **Reflexion** (notebook 18), which adds episodic memory of past mistakes.

State machine
-------------
    [generate] → [critique] ─┐
            ↑                │  satisfied? → [final]
            └─ [refine] ←────┘  else
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult
from agentic_architectures.evaluators import LLMJudge


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ReflectionState(TypedDict, total=False):
    task: str
    draft: str
    critique: str
    score: int
    iteration: Annotated[int, "current refinement loop index, starts at 0"]
    max_iterations: int
    target_score: int
    history: list[dict[str, Any]]  # one entry per (draft, critique, score) round
    final_output: str


# ---------------------------------------------------------------------------
# Critic rubric — what the Critic LLM produces
# ---------------------------------------------------------------------------
class _ReflectionCritique(BaseModel):
    """Critic output."""

    score: int = Field(ge=1, le=10, description="Overall quality on a 1-10 scale.")
    critique: str = Field(description="Concrete, actionable critique of the draft.")


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class Reflection(Architecture):
    """Self-Refine style reflection loop."""

    name = "reflection"
    description = (
        "Generate → critique → refine loop. Improves output quality by treating "
        "the LLM as both author and editor in alternation."
    )
    reference = "https://arxiv.org/abs/2303.17651"

    def __init__(
        self,
        max_iterations: int = 3,
        target_score: int = 9,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_iterations = max_iterations
        self.target_score = target_score
        self._judge: LLMJudge[_ReflectionCritique] = LLMJudge(
            schema=_ReflectionCritique,
            rubric=(
                "Score the candidate (1-10) and provide concrete, actionable critique. "
                "Focus on what is missing, wrong, unclear, or could be stronger. "
                "Reserve scores ≥9 for genuinely excellent work."
            ),
            llm=self.llm,
        )

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _generate(self, state: ReflectionState) -> dict[str, Any]:
        prompt = (
            f"# Task\n{state['task']}\n\n"
            f"Produce the best response you can. Be concrete and complete."
        )
        draft = self.llm.invoke(prompt).content
        return {
            "draft": str(draft),
            "iteration": 0,
            "history": [],
        }

    def _critique(self, state: ReflectionState) -> dict[str, Any]:
        evaluation = self._judge.evaluate(
            candidate=state["draft"],
            context={"task": state["task"]},
        )
        history = state.get("history", []) + [
            {
                "iteration": state.get("iteration", 0),
                "draft": state["draft"],
                "score": evaluation.score,
                "critique": evaluation.critique,
            }
        ]
        return {
            "score": evaluation.score,
            "critique": evaluation.critique,
            "history": history,
        }

    def _refine(self, state: ReflectionState) -> dict[str, Any]:
        prompt = (
            f"# Task\n{state['task']}\n\n"
            f"# Previous draft\n{state['draft']}\n\n"
            f"# Critic feedback (score {state['score']}/10)\n{state['critique']}\n\n"
            f"Produce an improved version that addresses every point of the critique. "
            f"Don't just patch — rewrite as needed."
        )
        new_draft = self.llm.invoke(prompt).content
        return {
            "draft": str(new_draft),
            "iteration": state.get("iteration", 0) + 1,
        }

    def _finalize(self, state: ReflectionState) -> dict[str, Any]:
        return {"final_output": state["draft"]}

    # ------------------------------------------------------------------ #
    #  Router                                                             #
    # ------------------------------------------------------------------ #

    def _should_continue(self, state: ReflectionState) -> str:
        if state.get("score", 0) >= self.target_score:
            return "finalize"
        if state.get("iteration", 0) >= self.max_iterations:
            return "finalize"
        return "refine"

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(ReflectionState)
        g.add_node("generate", self._generate)
        g.add_node("critique", self._critique)
        g.add_node("refine", self._refine)
        g.add_node("finalize", self._finalize)

        g.add_edge(START, "generate")
        g.add_edge("generate", "critique")
        g.add_conditional_edges(
            "critique",
            self._should_continue,
            {"refine": "refine", "finalize": "finalize"},
        )
        g.add_edge("refine", "critique")
        g.add_edge("finalize", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke(
            {
                "task": task,
                "max_iterations": self.max_iterations,
                "target_score": self.target_score,
            }
        )
        return ArchitectureResult(
            output=final_state.get("final_output", final_state.get("draft", "")),
            state=dict(final_state),
            trace=final_state.get("history", []),
            metadata={
                "final_score": final_state.get("score"),
                "iterations": final_state.get("iteration", 0) + 1,
                "max_iterations": self.max_iterations,
            },
        )
