"""Tree of Thoughts — beam-search over a tree of LLM-generated reasoning steps.

Instead of taking one greedy reasoning step at a time (CoT), ToT generates
multiple candidate next-thoughts at each step, scores them, keeps the top
`beam_width`, and expands those for another layer. The result is a *tree* of
reasoning paths; the best complete leaf wins.

Why this beats Chain-of-Thought (CoT):
  - CoT commits early. A first-step mistake propagates and there's no backtracking.
  - ToT keeps multiple alternatives alive, scored, and prunes systematically.

Why it beats Reflection (notebook 01):
  - Reflection iterates on ONE draft. ToT explores BRANCHES of *different* drafts.
  - For tasks where the right approach is unclear ("which framing to take?"), ToT's
    parallel exploration produces a better answer than Reflection's serial refinement.

Origin: Yao et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language
Models*, NeurIPS 2023 ([arXiv:2305.10601](https://arxiv.org/abs/2305.10601)).

Default LLM: a reasoning model (Qwen3-Thinking) is the natural pairing — each "thought"
gets the benefit of internal `<think>` reasoning, producing higher-quality branches.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult
from agentic_architectures.evaluators import LLMJudge


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class _ThoughtCandidates(BaseModel):
    """K candidate next-thoughts at one tree node."""

    candidates: list[str] = Field(
        description=(
            "Substantively DIFFERENT next reasoning steps or partial solutions. "
            "Each must explore a different angle / approach / framing. "
            "Avoid producing variants that are paraphrases of each other."
        ),
        min_length=2,
    )


class _ThoughtScore(BaseModel):
    """Score for one candidate thought — STRICT rubric to force discrimination."""

    score: int = Field(
        ge=1,
        le=5,
        description=(
            "STRICT 1-5 scoring. Be discriminating — if you score everything 5, "
            "beam search has no signal to prune on.\n"
            "  1 = clearly off-track, contradicts the task, or contains a factual error.\n"
            "  2 = on-topic but weak: overlapping with a sibling, vague, or shallow.\n"
            "  3 = plausible but unproven; standard / unremarkable.\n"
            "  4 = strong: substantive, specific, clearly advances toward an excellent answer.\n"
            "  5 = RARE excellence — reserve for thoughts that are decisively better than "
            "everything else you'd produce on this task.\n"
            "Calibration rule: across a group of K sibling candidates, AT MOST ONE should "
            "earn a 5. If two thoughts look equally good, give the slightly weaker one a 4."
        ),
    )
    rationale: str = Field(
        description=(
            "One sentence explaining the score. Must reference SPECIFIC features of THIS "
            "thought (not generic praise) — e.g. 'introduces a concrete sensory detail "
            "(the smell of old paper)' rather than 'good opening sentence'."
        ),
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class TreeOfThoughtsState(TypedDict, total=False):
    task: str
    # Full tree as flat list. Each entry: {id, content, score, depth, parent_id}.
    thoughts: Annotated[list[dict[str, Any]], operator.add]
    frontier: list[int]  # ids in `thoughts` that should be expanded next
    depth: int
    max_depth: int
    branching: int
    beam_width: int
    final_answer: str


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class TreeOfThoughts(Architecture):
    """Beam-search over a tree of LLM-generated reasoning steps."""

    name = "tree_of_thoughts"
    description = (
        "Generate K next-thoughts at each step, score them, keep the top "
        "beam_width, and expand for another layer. The best complete leaf wins. "
        "Beam-search over a reasoning tree."
    )
    reference = "https://arxiv.org/abs/2305.10601"

    def __init__(
        self,
        branching: int = 3,
        beam_width: int = 2,
        max_depth: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.branching = branching
        self.beam_width = beam_width
        self.max_depth = max_depth
        self._generator = self.llm.with_structured_output(_ThoughtCandidates)
        self._evaluator: LLMJudge[_ThoughtScore] = LLMJudge(
            schema=_ThoughtScore,
            rubric=(
                "Score this candidate thought on a STRICT 1-5 scale (see schema). "
                "You are running beam search — equal scores destroy the signal. "
                "Be DISCRIMINATING: most thoughts should be 2-4. Reserve 5 for rare "
                "excellence. If a thought is just a paraphrase of a sibling, score it 2."
            ),
            llm=self.llm,
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _path_from_root(thoughts: list[dict[str, Any]], thought_id: int) -> list[dict[str, Any]]:
        """Walk parent pointers up to the root and return ordered path."""
        by_id = {t["id"]: t for t in thoughts}
        path: list[dict[str, Any]] = []
        node = by_id.get(thought_id)
        while node is not None:
            path.append(node)
            parent = node.get("parent_id", -1)
            node = by_id.get(parent) if parent >= 0 else None
        return list(reversed(path))

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _root(self, state: TreeOfThoughtsState) -> dict[str, Any]:
        """Initialise tree with the root task as thought 0."""
        return {
            "thoughts": [
                {
                    "id": 0,
                    "content": f"[ROOT] {state['task']}",
                    "score": 0,
                    "depth": 0,
                    "parent_id": -1,
                }
            ],
            "frontier": [0],
            "depth": 0,
        }

    def _expand_and_score(self, state: TreeOfThoughtsState) -> dict[str, Any]:
        """Expand every frontier node into K children; score each."""
        thoughts = state.get("thoughts", [])
        next_id = max((t["id"] for t in thoughts), default=-1) + 1
        new_thoughts: list[dict[str, Any]] = []

        for parent_id in state["frontier"]:
            path = self._path_from_root(thoughts, parent_id)
            path_text = (
                "\n".join(f"  Step {t['depth']}: {t['content']}" for t in path[1:]) or "(root — no prior steps yet)"
            )

            gen_prompt = (
                f"## Task\n{state['task']}\n\n"
                f"## Reasoning so far (along this branch)\n{path_text}\n\n"
                f"## Generate {self.branching} candidates\n"
                "Produce substantively DIFFERENT next reasoning steps. Hard rules:\n"
                "  - Each candidate must explore a *different angle, framing, or "
                "approach* — not a different wording of the same idea.\n"
                "  - If you cannot find truly different angles, write fewer "
                "candidates rather than fill with paraphrases.\n"
                "  - At least one candidate should be deliberately UNCONVENTIONAL "
                "(a riskier path that might dominate or might flop)."
            )
            cands = self._generator.invoke(gen_prompt)

            for c in cands.candidates[: self.branching]:
                eval_result = self._evaluator.evaluate(
                    candidate=c,
                    context={
                        "task": state["task"],
                        "reasoning_so_far": path_text,
                    },
                )
                new_thoughts.append(
                    {
                        "id": next_id,
                        "content": c,
                        "score": eval_result.score,
                        "depth": path[-1]["depth"] + 1,
                        "parent_id": parent_id,
                        "rationale": eval_result.rationale,
                    }
                )
                next_id += 1

        return {"thoughts": new_thoughts, "depth": state.get("depth", 0) + 1}

    def _prune(self, state: TreeOfThoughtsState) -> dict[str, Any]:
        """Keep the top beam_width thoughts at the current depth as the new frontier."""
        depth = state.get("depth", 0)
        at_depth = [t for t in state["thoughts"] if t["depth"] == depth]
        at_depth.sort(key=lambda t: t["score"], reverse=True)
        kept = at_depth[: self.beam_width]
        return {"frontier": [t["id"] for t in kept]}

    def _finalize(self, state: TreeOfThoughtsState) -> dict[str, Any]:
        """Build the final answer from the best path through the tree."""
        thoughts = state["thoughts"]
        if not thoughts:
            return {"final_answer": "(no thoughts generated)"}

        # Best leaf = highest-scoring at max depth (or deepest if max not reached).
        max_depth_reached = max(t["depth"] for t in thoughts)
        leaves = [t for t in thoughts if t["depth"] == max_depth_reached]
        leaves.sort(key=lambda t: t["score"], reverse=True)
        best = leaves[0]
        path = self._path_from_root(thoughts, best["id"])
        path_md = "\n".join(f"  Step {t['depth']} (score {t['score']}/5): {t['content']}" for t in path[1:])

        prompt = (
            f"## Task\n{state['task']}\n\n"
            f"## Winning reasoning path (depth {max_depth_reached}, top score "
            f"{best['score']}/5)\n{path_md}\n\n"
            "Synthesise a concise final answer that follows from this reasoning. "
            "Do not invent steps beyond what the path supports."
        )
        return {"final_answer": str(self.llm.invoke(prompt).content)}

    # ------------------------------------------------------------------ #
    #  Router                                                             #
    # ------------------------------------------------------------------ #

    def _route(self, state: TreeOfThoughtsState) -> str:
        if state.get("depth", 0) >= self.max_depth:
            return "finalize"
        if not state.get("frontier"):
            return "finalize"
        return "expand"

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(TreeOfThoughtsState)
        g.add_node("root", self._root)
        g.add_node("expand", self._expand_and_score)
        g.add_node("prune", self._prune)
        g.add_node("finalize", self._finalize)

        g.add_edge(START, "root")
        g.add_edge("root", "expand")
        g.add_edge("expand", "prune")
        g.add_conditional_edges("prune", self._route, {"expand": "expand", "finalize": "finalize"})
        g.add_edge("finalize", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        config = {"recursion_limit": 5 * self.max_depth + 10}
        final_state = graph.invoke({"task": task}, config=config)

        thoughts = final_state.get("thoughts", [])
        max_depth_reached = max((t["depth"] for t in thoughts), default=0)
        leaves = [t for t in thoughts if t["depth"] == max_depth_reached]
        leaves.sort(key=lambda t: t["score"], reverse=True)

        trace = [{"type": "thought", **t} for t in thoughts]
        return ArchitectureResult(
            output=final_state.get("final_answer", ""),
            state={
                "task": task,
                "tree_size": len(thoughts),
                "max_depth_reached": max_depth_reached,
            },
            trace=trace,
            metadata={
                "total_thoughts": len(thoughts),
                "max_depth_reached": max_depth_reached,
                "branching": self.branching,
                "beam_width": self.beam_width,
                "best_leaf_score": leaves[0]["score"] if leaves else 0,
                "best_leaf_id": leaves[0]["id"] if leaves else -1,
            },
        )
