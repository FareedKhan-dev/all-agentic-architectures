"""RLHF-style self-improvement — editor critiques, generator revises, best outputs archived.

A misleading name (this isn't RL with human feedback — it's *editor-feedback* with
verbal critique), but it follows the same loop spirit: produce → critique →
revise → if good enough, archive as a high-quality example. The archive
becomes a reusable corpus of "things we've gotten right before" that can prime
future generations.

Compared to **Reflection** (notebook 01): same generate → critique → refine
loop, but Reflection produces ONE answer per task; RLHF here ACCUMULATES an
ARCHIVE of accepted outputs across tasks. The architecture is stateful across
calls.

Compared to **Reflexion** (notebook 18): Reflexion stores verbal *reflections
on past failures*; RLHF stores the *final accepted outputs* as positive examples.

State across calls:
  - `archive` (list of accepted outputs with metadata).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


# ---------------------------------------------------------------------------
# Editor critique schema — multi-dimensional, deterministic-scoring friendly
# ---------------------------------------------------------------------------
class _EditorCritique(BaseModel):
    """Multi-dimensional objective features the editor must commit to.

    The score that drives loop continuation and archive gating is COMPUTED IN
    PYTHON from these features, not from the LLM's `overall_score` field —
    sidesteps the LLM-as-Scorer flatness pathology (same fix as Mental Loop).
    """

    is_on_brief: bool = Field(description="True iff the output satisfies EVERY explicit constraint in the task.")
    word_count: int = Field(
        ge=0,
        description="Total word count of the output. Single point estimate.",
    )
    has_concrete_imagery: bool = Field(
        description="True iff the output uses specific concrete imagery (sensory detail, "
        "named entities) rather than vague abstractions ('quality', 'excellence')."
    )
    avoids_cliches: bool = Field(
        description="True iff the output AVOIDS hackneyed phrases like 'a journey of discovery', "
        "'where dreams come to life', 'unlock your potential', etc."
    )
    is_engaging: bool = Field(
        description="True iff a reasonable reader would WANT to keep reading / engaging "
        "after the first sentence. Boring-but-correct = False."
    )
    overall_score: int = Field(
        ge=1,
        le=10,
        description="Your subjective overall score 1-10. NOTE: this is preserved for "
        "comparison but is NOT used as the deciding signal — Python "
        "computes the deciding score from the boolean/numeric features above.",
    )
    critique: str = Field(description="Specific, actionable feedback pointing at concrete gaps.")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class RLHFState(TypedDict, total=False):
    task: str
    draft: str
    critique: str
    quality_score: int
    iteration: int
    max_iterations: int
    target_score: int
    history: Annotated[list[dict[str, Any]], operator.add]
    final_output: str
    archived: bool


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class RLHFSelfImprovement(Architecture):
    """Generate → critique → revise → maybe archive; archive persists across runs."""

    name = "rlhf_self_improvement"
    description = (
        "Editor-critic loop with persistent archive of accepted outputs. Each "
        "task: generate → critique → revise (up to max_iterations). If the final "
        "output crosses the quality threshold, append to the architecture's "
        "in-memory archive for reuse across future calls."
    )
    reference = "Self-Refine (Madaan 2023) + persistent archive pattern"

    def __init__(
        self,
        max_iterations: int = 2,
        target_score: int = 8,
        word_count_range: tuple[int, int] = (30, 100),
        **kwargs: Any,
    ) -> None:
        """
        Args:
            max_iterations: max refinement rounds per task.
            target_score: minimum COMPOSITE Python-score (0-10) for archive acceptance.
            word_count_range: (min, max) word count to receive the word-count points.
        """
        super().__init__(**kwargs)
        self.max_iterations = max_iterations
        self.target_score = target_score
        self.word_count_range = word_count_range
        self._editor = self.llm.with_structured_output(_EditorCritique)
        # Persistent archive across run() calls on the same instance.
        self.archive: list[dict[str, Any]] = []

    @staticmethod
    def _composite_score(features: dict[str, Any], wc_range: tuple[int, int]) -> int:
        """Python-computed score from objective editor features.

        Each boolean / range-check contributes a fixed weight. Total 0-10.
        This is the deterministic-scoring pattern (Mental Loop nb 10) generalized
        to multi-dimensional features — score has real spread because it depends
        on multiple INDEPENDENT booleans that the LLM has to commit to one by one.
        """
        score = 0
        if features.get("is_on_brief", False):
            score += 4
        wc_min, wc_max = wc_range
        if wc_min <= features.get("word_count", 0) <= wc_max:
            score += 2
        if features.get("has_concrete_imagery", False):
            score += 2
        if features.get("avoids_cliches", False):
            score += 1
        if features.get("is_engaging", False):
            score += 1
        return score

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _generate(self, state: RLHFState) -> dict[str, Any]:
        # Use the archive (if any) as positive examples in the prompt.
        examples_block = ""
        if self.archive:
            examples_block = (
                "\n## Recent high-quality examples from your archive\n"
                + "\n\n".join(
                    f"### Past task: {e['task'][:80]}\nOutput: {e['output'][:300]}…" for e in self.archive[-3:]
                )
                + "\n\nMatch or exceed the quality of these archived examples.\n"
            )
        prompt = (
            f"# Task\n{state['task']}\n{examples_block}\nProduce the best response you can. Be concrete and complete."
        )
        draft = str(self.llm.invoke(prompt).content)
        return {"draft": draft, "iteration": 0, "history": []}

    def _critique(self, state: RLHFState) -> dict[str, Any]:
        prompt = (
            "You are the EDITOR. Evaluate the candidate by committing to "
            "concrete OBJECTIVE features (booleans, word_count). The deciding "
            "score is computed in Python from your feature answers — don't "
            "try to game it by flat-rating everything 9/10.\n\n"
            f"## Task\n{state['task']}\n\n"
            f"## Candidate\n{state['draft']}\n\n"
            "Answer each feature honestly:\n"
            "  - is_on_brief: did the candidate satisfy EVERY explicit constraint?\n"
            "  - word_count: actual count\n"
            "  - has_concrete_imagery: specific sensory detail / named entities, NOT vague abstractions\n"
            "  - avoids_cliches: avoids hackneyed phrases\n"
            "  - is_engaging: would a reader want to keep reading?"
        )
        verdict = self._editor.invoke(prompt)
        features = verdict.model_dump()
        # Python composes the deciding score from objective features.
        composite = self._composite_score(features, self.word_count_range)
        history_entry = {
            "iteration": state.get("iteration", 0),
            "draft": state["draft"],
            "composite_score": composite,
            "llm_overall_score": verdict.overall_score,
            "is_on_brief": verdict.is_on_brief,
            "word_count": verdict.word_count,
            "has_concrete_imagery": verdict.has_concrete_imagery,
            "avoids_cliches": verdict.avoids_cliches,
            "is_engaging": verdict.is_engaging,
            "critique": verdict.critique,
        }
        return {
            "quality_score": composite,  # router uses this (composite, not LLM's flat number)
            "critique": verdict.critique,
            "history": [history_entry],
        }

    def _refine(self, state: RLHFState) -> dict[str, Any]:
        prompt = (
            f"# Task\n{state['task']}\n\n"
            f"# Previous draft\n{state['draft']}\n\n"
            f"# Editor feedback (score {state['quality_score']}/10)\n"
            f"{state['critique']}\n\n"
            "Produce an improved version that directly addresses the editor's "
            "feedback. Don't just patch — rewrite as needed."
        )
        new_draft = str(self.llm.invoke(prompt).content)
        return {"draft": new_draft, "iteration": state.get("iteration", 0) + 1}

    def _finalize(self, state: RLHFState) -> dict[str, Any]:
        # Archive decision: Python composite score >= target_score.
        # No dependence on LLM's `accept_for_archive` boolean — fully deterministic.
        final_score = state.get("quality_score", 0)
        should_archive = final_score >= self.target_score
        if should_archive:
            latest = state.get("history", [])[-1] if state.get("history") else {}
            self.archive.append(
                {
                    "task": state["task"],
                    "output": state["draft"],
                    "score": final_score,
                    "iterations": state.get("iteration", 0) + 1,
                    "features": {
                        k: latest.get(k)
                        for k in (
                            "is_on_brief",
                            "word_count",
                            "has_concrete_imagery",
                            "avoids_cliches",
                            "is_engaging",
                        )
                    },
                }
            )
        return {"final_output": state["draft"], "archived": should_archive}

    # ------------------------------------------------------------------ #
    #  Router                                                             #
    # ------------------------------------------------------------------ #

    def _should_continue(self, state: RLHFState) -> str:
        if state.get("quality_score", 0) >= self.target_score:
            return "finalize"
        if state.get("iteration", 0) >= self.max_iterations:
            return "finalize"
        return "refine"

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(RLHFState)
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
            state={
                "archive_size": len(self.archive),
                "final_score": final_state.get("quality_score", 0),
                "archived_this_run": final_state.get("archived", False),
            },
            trace=final_state.get("history", []),
            metadata={
                "iterations": final_state.get("iteration", 0) + 1,
                "final_score": final_state.get("quality_score", 0),
                "archived_this_run": final_state.get("archived", False),
                "archive_size": len(self.archive),
                "target_score": self.target_score,
            },
        )
