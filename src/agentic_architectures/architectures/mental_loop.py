"""Mental Loop (Simulator) — predict outcomes of candidate actions before committing.

Sometimes the agent shouldn't just *do* the obvious next thing. For high-stakes
or irreversible actions, it's better to first *imagine* what would happen for
several candidate actions, score those predictions, and only then commit.

Pattern:
  1. **Generate** K candidate actions from the current situation.
  2. **Simulate** the outcome of each candidate via an LLM "mental model"
     (predicted outcome + risks + benefits + score).
  3. **Decide** by picking the highest-scoring simulation.
  4. **Explain** why this action was chosen, referencing the simulations.

This is the pattern behind robotics-style "world models" and human-style
deliberation. Compared to **Tree of Thoughts** (notebook 09), Mental Loop is
**flat** (1 layer of simulation, not a tree) and **action-centric** (the
candidates are *things to do*, not abstract reasoning steps).

When to reach for it:
  - Irreversible decisions (financial, medical, deployments).
  - Tasks where the consequence of a wrong move is much worse than slow deliberation.
  - Robot / agent planning where simulated outcomes guide selection.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class _CandidateActions(BaseModel):
    """K distinct candidate actions the agent could take."""

    actions: list[str] = Field(
        min_length=2,
        description=(
            "K distinct candidate actions. Each must be a SPECIFIC, actionable "
            "choice (not a vague principle). Each should represent a different "
            "strategy, not different wordings of the same one."
        ),
    )


class _SimulatedOutcome(BaseModel):
    """Mental simulation of one candidate action's outcome.

    The `predicted_metric` field is the key to the deterministic-scoring fix:
    if the caller passes a `scoring_fn` to `MentalLoop`, the LLM-supplied
    `overall_score` is **overridden** by `scoring_fn(predicted_metric)`. This
    sidesteps the LLM-as-Scorer flatness problem (Llama-style models compress
    scores into a narrow band regardless of the rubric).
    """

    predicted_outcome: str = Field(
        description="A concrete 2-3 sentence prediction of what would happen if this action is taken."
    )
    predicted_metric: float | None = Field(
        default=None,
        description=(
            "If the task asks for a specific NUMBER (travel time in minutes, "
            "cost in dollars, error rate %, latency in ms, etc.), extract your "
            "best single point estimate as a float here. If the task doesn't "
            "have a measurable metric, leave null."
        ),
    )
    benefits: list[str] = Field(
        default_factory=list,
        description="Specific upsides if this prediction comes true.",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Specific things that could go wrong with this action.",
    )
    overall_score: int = Field(
        ge=1,
        le=5,
        description=(
            "How good is this action OVERALL — weighing predicted outcome × probability "
            "of that outcome × upside/downside ratio. STRICT 1-5: be discriminating. "
            "1 = clearly bad / high risk. 5 = clearly excellent. Most actions 2-4."
        ),
    )
    rationale: str = Field(description="One sentence explaining the overall score.")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class MentalLoopState(TypedDict, total=False):
    task: str
    candidate_actions: list[str]
    simulations: Annotated[list[dict[str, Any]], operator.add]
    chosen_action: str
    chosen_score: int
    explanation: str


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class MentalLoop(Architecture):
    """Generate → simulate → score → pick best → explain."""

    name = "mental_loop"
    description = (
        "Generate K candidate actions, mentally simulate the outcome of each "
        "(predicted outcome + risks + benefits + score), then pick the highest-"
        "scoring action. The candidates are never executed in the real world — "
        "the agent reports the decision."
    )
    reference = "Robotics 'world models'; classical deliberation in AI"

    def __init__(
        self,
        n_candidates: int = 3,
        scoring_fn: Callable[[float], int] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a Mental Loop architecture.

        Args:
            n_candidates: how many candidate actions to generate.
            scoring_fn: optional deterministic Python function that takes the
                LLM's `predicted_metric` (a float) and returns an int 1-5 score.
                When set, the LLM's own `overall_score` is OVERRIDDEN by this
                function — the canonical fix for LLM-as-Scorer flatness.
                When None (default), the LLM's `overall_score` is used as-is.
        """
        super().__init__(**kwargs)
        self.n_candidates = n_candidates
        self.scoring_fn = scoring_fn
        self._generator = self.llm.with_structured_output(_CandidateActions)
        self._simulator = self.llm.with_structured_output(_SimulatedOutcome)

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _generate(self, state: MentalLoopState) -> dict[str, Any]:
        prompt = (
            f"## Decision task\n{state['task']}\n\n"
            f"## Generate {self.n_candidates} distinct candidate actions\n"
            "Each candidate must be a SPECIFIC course of action (not a vague principle). "
            "Each should represent a different *strategy*, not different wordings of the "
            "same idea. At least one candidate should be UNCONVENTIONAL — a riskier but "
            "potentially-dominant choice."
        )
        result = self._generator.invoke(prompt)
        return {"candidate_actions": list(result.actions[: self.n_candidates])}

    def _simulate(self, state: MentalLoopState) -> dict[str, Any]:
        sims: list[dict[str, Any]] = []
        for action in state["candidate_actions"]:
            metric_instruction = (
                "MUST populate `predicted_metric` with your point-estimate of the "
                "task's measurable outcome (e.g. minutes, dollars, error %)."
                if self.scoring_fn is not None
                else "If the task has a measurable metric, populate predicted_metric."
            )
            sim_prompt = (
                f"## Original task\n{state['task']}\n\n"
                f"## Candidate action being simulated\n{action}\n\n"
                "Mentally simulate what would happen if this specific action is taken. "
                "Be concrete about predicted outcome, benefits, risks. "
                f"{metric_instruction} "
                "Then score the action OVERALL on the strict 1-5 scale."
            )
            outcome = self._simulator.invoke(sim_prompt)
            data = outcome.model_dump()
            data["llm_score"] = data["overall_score"]  # preserve the LLM's original score
            if self.scoring_fn is not None and outcome.predicted_metric is not None:
                data["overall_score"] = self.scoring_fn(outcome.predicted_metric)
                data["score_source"] = "deterministic"
            else:
                data["score_source"] = "llm"
            sims.append({"action": action, **data})
        return {"simulations": sims}

    def _decide(self, state: MentalLoopState) -> dict[str, Any]:
        sims = state.get("simulations", [])
        if not sims:
            return {"chosen_action": "", "chosen_score": 0}
        best = max(sims, key=lambda s: s["overall_score"])
        return {"chosen_action": best["action"], "chosen_score": best["overall_score"]}

    def _explain(self, state: MentalLoopState) -> dict[str, Any]:
        sims = state.get("simulations", [])
        sims_md = "\n".join(
            f"- **{s['action']}** — predicted: {s['predicted_outcome']} (score {s['overall_score']}/5)" for s in sims
        )
        prompt = (
            f"## Decision task\n{state['task']}\n\n"
            f"## Simulations\n{sims_md}\n\n"
            f"## Chosen action\n{state['chosen_action']}  (score {state['chosen_score']}/5)\n\n"
            "Write a 3-5 sentence explanation that:\n"
            "  (a) names the chosen action,\n"
            "  (b) cites the predicted-outcome highlights that justify it,\n"
            "  (c) explicitly mentions what tradeoff was accepted vs the runner-up action."
        )
        return {"explanation": str(self.llm.invoke(prompt).content)}

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(MentalLoopState)
        g.add_node("generate", self._generate)
        g.add_node("simulate", self._simulate)
        g.add_node("decide", self._decide)
        g.add_node("explain", self._explain)
        g.add_edge(START, "generate")
        g.add_edge("generate", "simulate")
        g.add_edge("simulate", "decide")
        g.add_edge("decide", "explain")
        g.add_edge("explain", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke({"task": task})
        sims = final_state.get("simulations", [])
        scores = [s["overall_score"] for s in sims]
        return ArchitectureResult(
            output=final_state.get("explanation", ""),
            state={
                "chosen_action": final_state.get("chosen_action", ""),
                "chosen_score": final_state.get("chosen_score", 0),
            },
            trace=[{"type": "simulation", **s} for s in sims],
            metadata={
                "n_candidates": len(sims),
                "scores": scores,
                "score_spread": max(scores) - min(scores) if scores else 0,
                "chosen_score": final_state.get("chosen_score", 0),
            },
        )
