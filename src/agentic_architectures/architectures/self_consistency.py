"""Self-Consistency — sample N reasoning paths, majority-vote the answer.

The simplest hallucination/error mitigation that actually works at scale:
sample multiple chain-of-thought reasoning paths from the model (with non-zero
temperature for diversity), extract each path's final answer, and majority-vote.

Wang et al.'s insight: complex reasoning problems often admit multiple paths,
and the *most common* terminal answer across paths is usually the right one,
even when individual paths are wrong.

Origin: Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in
Language Models* (Google 2022). https://arxiv.org/abs/2203.11171

**Deterministic-picker pattern applied** (handoff §7):
  - Each sample emits a *structured* `(reasoning, answer)` pair with the
    `answer` field constrained by schema.
  - Python tallies via `collections.Counter` and picks the modal answer.
  - The LLM never sees the votes, so it cannot flatten or game them.
"""

from __future__ import annotations

import operator
from collections import Counter
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class _ReasoningSample(BaseModel):
    """One sampled chain-of-thought."""

    reasoning: str = Field(
        description="The step-by-step reasoning. Be explicit about each arithmetic or logical step. Don't skip steps."
    )
    answer: str = Field(
        description="JUST the final answer in the requested format — no units, no "
        "explanation, no punctuation beyond what the format requires. "
        "If the answer is a number, write it as a bare number (e.g., '179'). "
        "If a single word/name, write only that word."
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class SelfConsistencyState(TypedDict, total=False):
    task: str
    n_samples: int
    samples: Annotated[list[dict[str, str]], operator.add]
    tally: dict[str, int]
    final_answer: str
    history: Annotated[list[dict[str, Any]], operator.add]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class SelfConsistency(Architecture):
    """Sample N CoT paths, majority-vote the modal final answer."""

    name = "self_consistency"
    description = (
        "Sample N independent chain-of-thought reasoning paths with non-zero "
        "temperature, extract each path's final answer, and majority-vote. "
        "Picker is pure Python (collections.Counter) — no LLM-as-Scorer."
    )
    reference = "https://arxiv.org/abs/2203.11171"

    def __init__(
        self,
        n_samples: int = 5,
        sample_temperature: float = 0.8,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            n_samples: how many independent reasoning paths to draw.
            sample_temperature: temperature for the sampling LLM. Higher = more diverse
                paths (and more likely to surface the correct one via majority).
        """
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.sample_temperature = sample_temperature
        # Use the LLM with structured output to enforce the (reasoning, answer) shape.
        # We DON'T rebind temperature here — let the user inject an LLM at desired temp;
        # we override per-call below via .bind() so the sample-LLM is always warm.
        self._sampler_schema = _ReasoningSample

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _sample_all(self, state: SelfConsistencyState) -> dict[str, Any]:
        """Draw n_samples independent reasoning paths.

        We loop here rather than parallel-fanning out to keep the LangGraph
        topology pedagogically simple. For production-scale, swap to
        `asyncio.gather` over `ainvoke` calls.
        """
        # Each invoke gets a fresh model bound to high temperature.
        sampler_llm = self.llm.bind(temperature=self.sample_temperature)
        sampler = sampler_llm.with_structured_output(self._sampler_schema)
        prompt = (
            f"# Task\n{state['task']}\n\n"
            "Reason step-by-step. Be explicit about every arithmetic or logical move. "
            "Then give the final answer in the requested format — JUST the answer, "
            "no units, no explanation in the answer field."
        )
        samples: list[dict[str, str]] = []
        for i in range(state["n_samples"]):
            try:
                s = sampler.invoke(prompt)
                samples.append(
                    {
                        "sample_index": str(i),
                        "reasoning": s.reasoning,
                        "answer": s.answer.strip(),
                    }
                )
            except Exception as e:  # malformed structured output → skip this sample
                samples.append(
                    {
                        "sample_index": str(i),
                        "reasoning": f"(sample failed: {e})",
                        "answer": "",
                    }
                )
        return {
            "samples": samples,
            "history": [{"stage": "sample_all", "n": len(samples)}],
        }

    def _vote(self, state: SelfConsistencyState) -> dict[str, Any]:
        """Python-only majority vote — the deterministic-picker."""
        answers = [s["answer"] for s in state["samples"] if s.get("answer")]
        if not answers:
            return {
                "tally": {},
                "final_answer": "",
                "history": [{"stage": "vote", "tally": {}, "final": ""}],
            }

        # Normalise for tally: lowercase, strip whitespace, drop trailing periods.
        def _norm(s: str) -> str:
            return s.strip().lower().rstrip(".")

        norm_to_raw: dict[str, str] = {}
        for a in answers:
            n = _norm(a)
            # Keep the first raw form we saw for each normalised key.
            if n not in norm_to_raw:
                norm_to_raw[n] = a
        tally = Counter(_norm(a) for a in answers)
        winner_norm, _ = tally.most_common(1)[0]
        winner_raw = norm_to_raw[winner_norm]
        # Convert Counter to dict for state serialization.
        return {
            "tally": dict(tally),
            "final_answer": winner_raw,
            "history": [
                {
                    "stage": "vote",
                    "tally": dict(tally),
                    "winner": winner_raw,
                    "n_total_answers": len(answers),
                }
            ],
        }

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(SelfConsistencyState)
        g.add_node("sample_all", self._sample_all)
        g.add_node("vote", self._vote)
        g.add_edge(START, "sample_all")
        g.add_edge("sample_all", "vote")
        g.add_edge("vote", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke(
            {"task": task, "n_samples": self.n_samples},
            config={"recursion_limit": 25},
        )
        tally = final_state.get("tally", {})
        n_total = sum(tally.values()) if tally else 0
        winner_count = tally.get(final_state.get("final_answer", "").strip().lower().rstrip("."), 0)
        # Robust winner-count lookup (final_answer is raw, tally keys are normalised)
        if tally and winner_count == 0:
            # find the count by max (modal answer's count)
            winner_count = max(tally.values()) if tally else 0
        agreement = winner_count / n_total if n_total else 0.0
        return ArchitectureResult(
            output=final_state.get("final_answer", ""),
            state={
                "n_samples": final_state.get("n_samples", self.n_samples),
                "unique_answers": len(tally),
                "winner_count": winner_count,
                "agreement_fraction": agreement,
            },
            trace=final_state.get("history", []),
            metadata={
                "n_samples": final_state.get("n_samples", self.n_samples),
                "samples": final_state.get("samples", []),
                "tally": tally,
                "winner_count": winner_count,
                "agreement_fraction": agreement,
                "unique_answers": len(tally),
            },
        )
