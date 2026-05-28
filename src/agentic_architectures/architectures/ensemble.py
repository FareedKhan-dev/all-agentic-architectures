"""Ensemble — N independent voters tackle the same problem in parallel; aggregator synthesises.

The simplest "wisdom of crowds" pattern: run K agents on the SAME task, each
with a different perspective / prompt / temperature, then combine.

Three aggregation modes — note the deterministic vs LLM-dependent split:
  - **llm_synth** (default): an Aggregator LLM weaves the K answers into one
    balanced response. Best for open-ended questions where nuance matters.
  - **majority_vote** (DETERMINISTIC picker): each voter emits a
    `categorical_answer` field via structured output; Python tallies the
    answers and returns the mode. Sidesteps LLM-as-Scorer flatness — same
    pattern as Mental Loop's `scoring_fn` (notebook 10).
  - **highest_confidence** (DEPRECATED for argmax — relies on self-reported
    confidence which Llama-style models compress to a flat band of 4/5; kept
    for didactic purposes — see § 11.1 of notebook 13).

Cost: K voter calls + 1 aggregator call (mode `llm_synth`) or 0 (modes
`majority_vote` / `highest_confidence`).
"""

from __future__ import annotations

import operator
import re
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


def _infer_categorical(bottom_line: str) -> str:
    """Fallback for voters who left categorical_answer null — extract YES/NO/UNCERTAIN
    from the bottom_line via keyword heuristic."""
    text = bottom_line.lower()
    # Negation patterns first (otherwise "do not think" matches YES via "think")
    if re.search(r"\b(no|not\s|don't|doesn't|won't|unlikely|doubt|skeptic|disagree)\b", text):
        return "NO"
    if re.search(r"\b(yes|will|likely|probably|agree|support)\b", text):
        return "YES"
    if re.search(r"\b(uncertain|unclear|maybe|depends|possibly|hard to say)\b", text):
        return "UNCERTAIN"
    return ""


# ---------------------------------------------------------------------------
# Default voter perspectives — easily overridden
# ---------------------------------------------------------------------------
DEFAULT_VOTERS: dict[str, str] = {
    "analytical": (
        "You are an ANALYTICAL voter. Approach the question with data, logic, "
        "and structured reasoning. Cite concrete numbers, mechanisms, and "
        "evidence-based arguments."
    ),
    "skeptical": (
        "You are a SKEPTICAL voter. Identify weaknesses, edge cases, missing "
        "context, and assumptions in the question itself. Argue what could go "
        "wrong with the most obvious answer."
    ),
    "pragmatic": (
        "You are a PRAGMATIC voter. Focus on what actually works in practice, "
        "real-world constraints, and shippable recommendations. Skip theoretical "
        "ideals; favour decisions a busy professional would actually make."
    ),
}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class _VoterOpinion(BaseModel):
    """One voter's typed answer to the shared task.

    The `categorical_answer` field is the key to the deterministic-picker fix
    (see module docstring): for yes/no or multiple-choice questions, each
    voter commits to a SHORT discrete string (e.g. "YES" / "NO" / "A" / "B")
    which Python can tally to produce a `majority_vote` argmax — sidestepping
    the unreliable self-reported `confidence` field.
    """

    bottom_line: str = Field(description="A 1-2 sentence direct answer to the question, from this voter's perspective.")
    categorical_answer: str | None = Field(
        default=None,
        description=(
            "For binary or multiple-choice questions, output your DIRECT "
            "categorical answer as a short uppercase string: 'YES' / 'NO' / "
            "'UNCERTAIN' / 'A' / 'B' / 'C'. For open-ended questions where no "
            "single categorical answer fits, leave null."
        ),
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="2-4 supporting points specific to this voter's perspective.",
    )
    confidence: int = Field(
        ge=1,
        le=5,
        description=(
            "Self-reported confidence 1-5. NOTE: instruction-tuned LLMs compress "
            "this to a flat band — do NOT rely on it for argmax selection; use "
            "`majority_vote` mode (which uses `categorical_answer`) instead."
        ),
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class EnsembleState(TypedDict, total=False):
    task: str
    voter_opinions: Annotated[list[dict[str, Any]], operator.add]
    aggregated_answer: str
    aggregator_mode: Literal["llm_synth", "majority_vote", "highest_confidence"]
    vote_tally: dict[str, int]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class Ensemble(Architecture):
    """N voters with diverse prompts run on the same task; aggregator combines."""

    name = "ensemble"
    description = (
        "Run K voter agents (each with its own perspective system prompt) against "
        "the same task in parallel. An aggregator LLM synthesises their answers "
        "into one balanced response that preserves multi-perspective nuance."
    )
    reference = "Wisdom of crowds (Surowiecki 2004); LLM ensembles in modern practice."

    def __init__(
        self,
        voters: dict[str, str] | None = None,
        aggregator_mode: Literal["llm_synth", "majority_vote", "highest_confidence"] = "llm_synth",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.voters = voters or DEFAULT_VOTERS
        self.aggregator_mode = aggregator_mode
        self._voter_llm = self.llm.with_structured_output(_VoterOpinion)

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _vote(self, state: EnsembleState) -> dict[str, Any]:
        """Run all voters sequentially (could be parallelised — see § 11.3)."""
        mode = state.get("aggregator_mode") or self.aggregator_mode
        # When majority_vote mode is active, voters MUST commit to a categorical answer
        # — Python's tally needs it. The instruction here is what drives the schema's
        # otherwise-optional `categorical_answer` field to be populated.
        categorical_instruction = (
            "## CRITICAL\n"
            "The aggregator is using majority_vote mode. You MUST populate "
            "`categorical_answer` with a SHORT UPPERCASE string (e.g. 'YES', 'NO', "
            "'UNCERTAIN', 'A', 'B', 'C') that captures your directional position. "
            "Do NOT leave it null.\n\n"
            if mode == "majority_vote"
            else ""
        )
        opinions: list[dict[str, Any]] = []
        for name, perspective in self.voters.items():
            prompt = (
                f"## Your role / perspective\n{perspective}\n\n"
                f"## Shared question\n{state['task']}\n\n"
                f"{categorical_instruction}"
                "## Your task\n"
                "Answer the question from YOUR perspective. Be true to your role — "
                "don't try to balance views; the aggregator will do that. Output "
                "your bottom-line answer, 2-4 key points, and your confidence."
            )
            op = self._voter_llm.invoke(prompt)
            opinions.append({"voter": name, **op.model_dump()})
        return {"voter_opinions": opinions}

    def _aggregate(self, state: EnsembleState) -> dict[str, Any]:
        opinions = state.get("voter_opinions", [])
        if not opinions:
            return {"aggregated_answer": "(no voter opinions to aggregate)"}

        mode = state.get("aggregator_mode") or self.aggregator_mode

        if mode == "majority_vote":
            # DETERMINISTIC picker — Python tallies `categorical_answer` strings.
            # Falls back to keyword extraction from bottom_line when a voter
            # left categorical_answer null (Llama doesn't always populate optional fields).
            from collections import Counter

            tally: Counter[str] = Counter()
            for o in opinions:
                cat = (o.get("categorical_answer") or "").strip().upper()
                if not cat:
                    cat = _infer_categorical(o.get("bottom_line", ""))
                if cat:
                    o["_inferred_categorical"] = cat
                    tally[cat] += 1
            if not tally:
                return {
                    "aggregated_answer": (
                        "**majority_vote unavailable** — no voter produced a "
                        "categorical answer. Use `llm_synth` mode for "
                        "open-ended questions."
                    ),
                    "vote_tally": {},
                }
            winner_label, winner_count = tally.most_common(1)[0]

            def _label_for(o: dict[str, Any]) -> str:
                return (o.get("categorical_answer") or "").strip().upper() or o.get("_inferred_categorical", "")

            supporting = [o for o in opinions if _label_for(o) == winner_label]
            others = [o for o in opinions if _label_for(o) != winner_label]
            tally_str = ", ".join(f"{k}={v}" for k, v in tally.most_common())
            ans = (
                f"**Majority answer: {winner_label}** "
                f"({winner_count}/{len(opinions)} voters) — tally: {tally_str}.\n\n"
                "Supporting voters' bottom lines:\n"
                + "\n".join(f"- ({o['voter']}) {o['bottom_line']}" for o in supporting)
                + (
                    "\n\nDissenting voter(s):\n"
                    + "\n".join(f"- ({o['voter']} -> {_label_for(o) or '?'}) {o['bottom_line']}" for o in others)
                    if others
                    else ""
                )
            )
            return {"aggregated_answer": ans, "vote_tally": dict(tally)}

        if mode == "highest_confidence":
            winner = max(opinions, key=lambda o: o.get("confidence", 0))
            ans = (
                f"**Winning voter (highest confidence {winner['confidence']}/5): "
                f"{winner['voter']}**\n\n"
                f"{winner['bottom_line']}\n\n"
                "Key points:\n"
                + "\n".join(f"- {p}" for p in winner.get("key_points", []))
                + "\n\n*Note: this mode is unreliable because instruction-tuned "
                "LLMs flat-score confidence at 4/5. Prefer `majority_vote` or "
                "`llm_synth`.*"
            )
            return {"aggregated_answer": ans}

        # llm_synth (default)
        sections = "\n\n".join(
            f"### {o['voter'].upper()} (confidence {o['confidence']}/5)\n"
            f"**Bottom line:** {o['bottom_line']}\n\n"
            "Key points:\n" + "\n".join(f"- {p}" for p in o.get("key_points", []))
            for o in opinions
        )
        prompt = (
            f"You are the Aggregator. Three voters answered the same question with "
            f"different perspectives:\n\n{sections}\n\n"
            f"## Original question\n{state['task']}\n\n"
            "## Your task\n"
            "Synthesise a single balanced ~200-word answer that:\n"
            "  1. States the most likely correct answer in 1-2 sentences.\n"
            "  2. Identifies the POINTS OF AGREEMENT across voters.\n"
            "  3. Identifies the GENUINE DISAGREEMENTS (don't paper over them).\n"
            "  4. Ends with a recommendation hedged by remaining uncertainty.\n"
            "Do NOT invent facts — work only from voters' content."
        )
        return {"aggregated_answer": str(self.llm.invoke(prompt).content)}

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(EnsembleState)
        g.add_node("vote", self._vote)
        g.add_node("aggregate", self._aggregate)
        g.add_edge(START, "vote")
        g.add_edge("vote", "aggregate")
        g.add_edge("aggregate", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke({"task": task, "aggregator_mode": self.aggregator_mode})
        opinions = final_state.get("voter_opinions", [])
        confidences = [o.get("confidence", 0) for o in opinions]
        cat_answers = [(o.get("categorical_answer") or "").strip().upper() or None for o in opinions]
        return ArchitectureResult(
            output=final_state.get("aggregated_answer", ""),
            state={
                "voters_used": [o["voter"] for o in opinions],
                "aggregator_mode": self.aggregator_mode,
                "vote_tally": final_state.get("vote_tally", {}),
            },
            trace=[{"type": "opinion", **o} for o in opinions],
            metadata={
                "n_voters": len(opinions),
                "confidences": confidences,
                "confidence_spread": (max(confidences) - min(confidences)) if confidences else 0,
                "categorical_answers": cat_answers,
                "vote_tally": final_state.get("vote_tally", {}),
                "aggregator_mode": self.aggregator_mode,
            },
        )
