"""Chain-of-Verification (CoVe) — reduce hallucination via self-verification.

Single-pass pipeline (Dhuliawala et al., Meta 2023):
  1. **BASELINE** — produce an initial answer (likely contains hallucinated claims).
  2. **PLAN** — generate verification questions about specific claims in the baseline.
  3. **EXECUTE** — answer each verification question independently (no access to baseline).
  4. **REVISE** — rewrite the baseline keeping only the verified claims; drop or correct the rest.

The key insight: when answering verification questions **without** seeing the
baseline, the model is far less prone to confabulate to maintain consistency
with its earlier (wrong) answer.

Builds on **Reflection** (nb 01): same critique-then-revise spirit, but the
critique here is decomposed into independent atomic checks rather than a
single holistic judgement.

Origin: Dhuliawala et al., *Chain-of-Verification Reduces Hallucination in
Large Language Models* (2023). https://arxiv.org/abs/2309.11495

No LLM-as-Scorer step → no flat-scoring pathology to fix. The REVISE stage
makes categorical keep/drop decisions per claim, not numeric scores.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class _VerificationQuestions(BaseModel):
    """Stage 2 — questions designed to probe specific claims in the baseline."""

    questions: list[str] = Field(
        description=(
            "3-7 verification questions. Each must target ONE specific factual claim "
            "from the baseline. Phrase as standalone questions answerable without "
            "seeing the baseline (e.g., 'Was X born in Y?', not 'Is the claim true?')."
        ),
        min_length=2,
        max_length=10,
    )


class _VerificationAnswer(BaseModel):
    """Stage 3 — independent answer to one verification question."""

    question: str = Field(description="The question, copied verbatim.")
    answer: str = Field(
        description="The answer in 1-2 sentences. If you're not confident, say so explicitly "
        "('I am not certain whether...') rather than guessing."
    )
    confidence: str = Field(
        description="One of: 'high', 'medium', 'low'. Use 'low' if the answer depends on facts you don't actually know."
    )


class _RevisedResponse(BaseModel):
    """Stage 4 — final answer after applying verification."""

    revised_response: str = Field(
        description="The rewritten answer, keeping only claims that the verification "
        "questions confirmed (or didn't disconfirm). Drop or correct any "
        "claim the verification answers contradicted."
    )
    changes_made: list[str] = Field(
        description="One bullet per change made (claim dropped, claim corrected, "
        "claim kept-as-is-because-verified). Empty list = no changes needed."
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class CoVeState(TypedDict, total=False):
    task: str
    baseline_response: str
    verification_questions: list[str]
    verification_answers: list[dict[str, str]]
    revised_response: str
    changes_made: list[str]
    history: Annotated[list[dict[str, Any]], operator.add]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class ChainOfVerification(Architecture):
    """BASELINE → PLAN questions → EXECUTE answers → REVISE."""

    name = "chain_of_verification"
    description = (
        "Generate a baseline answer, plan factual-verification questions, answer "
        "each independently, then revise the baseline keeping only verified claims."
    )
    reference = "https://arxiv.org/abs/2309.11495"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._planner = self.llm.with_structured_output(_VerificationQuestions)
        self._executor = self.llm.with_structured_output(_VerificationAnswer)
        self._reviser = self.llm.with_structured_output(_RevisedResponse)

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _baseline(self, state: CoVeState) -> dict[str, Any]:
        prompt = (
            f"# Task\n{state['task']}\n\n"
            "Produce your best answer. Be specific and concrete. Do not hedge or "
            "include disclaimers about uncertainty — give the direct answer."
        )
        baseline = str(self.llm.invoke(prompt).content)
        return {
            "baseline_response": baseline,
            "history": [{"stage": "baseline", "response": baseline}],
        }

    def _plan(self, state: CoVeState) -> dict[str, Any]:
        prompt = (
            "You will design verification questions to fact-check a candidate answer. "
            "Each question must target ONE specific factual claim. Phrase the "
            "questions so they are answerable WITHOUT seeing the candidate answer "
            "(treat them as fresh standalone questions).\n\n"
            f"## Task that produced the candidate\n{state['task']}\n\n"
            f"## Candidate answer\n{state['baseline_response']}\n\n"
            "Generate 3-7 verification questions probing the most-likely-wrong claims."
        )
        vq = self._planner.invoke(prompt)
        return {
            "verification_questions": list(vq.questions),
            "history": [{"stage": "plan", "questions": list(vq.questions)}],
        }

    def _execute(self, state: CoVeState) -> dict[str, Any]:
        # Answer each verification question independently — critically, without
        # seeing the baseline response. Without that isolation, the LLM tends to
        # rationalise the baseline's claims rather than fact-check them.
        answers: list[dict[str, str]] = []
        for q in state["verification_questions"]:
            prompt = (
                "Answer the following question with what you actually know. "
                "If you are not confident, say so explicitly — do not guess.\n\n"
                f"## Question\n{q}"
            )
            va = self._executor.invoke(prompt)
            answers.append(
                {
                    "question": va.question,
                    "answer": va.answer,
                    "confidence": va.confidence,
                }
            )
        return {
            "verification_answers": answers,
            "history": [{"stage": "execute", "answers": answers}],
        }

    def _revise(self, state: CoVeState) -> dict[str, Any]:
        qa_block = "\n\n".join(
            f"**Q{i + 1}.** {a['question']}\n**A.** {a['answer']}  *(confidence: {a['confidence']})*"
            for i, a in enumerate(state["verification_answers"])
        )
        prompt = (
            "Revise the candidate answer using the verification Q&A below. Keep "
            "only claims the verification supports; drop or correct any claim the "
            "verification contradicts (or that wasn't verified and looks doubtful).\n\n"
            f"## Original task\n{state['task']}\n\n"
            f"## Candidate answer (before verification)\n{state['baseline_response']}\n\n"
            f"## Verification Q&A (answered independently)\n{qa_block}\n\n"
            "Return the revised answer and a bullet-list of changes you made."
        )
        rr = self._reviser.invoke(prompt)
        return {
            "revised_response": rr.revised_response,
            "changes_made": list(rr.changes_made),
            "history": [
                {
                    "stage": "revise",
                    "revised_response": rr.revised_response,
                    "changes_made": list(rr.changes_made),
                }
            ],
        }

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(CoVeState)
        g.add_node("baseline", self._baseline)
        g.add_node("plan", self._plan)
        g.add_node("execute", self._execute)
        g.add_node("revise", self._revise)
        g.add_edge(START, "baseline")
        g.add_edge("baseline", "plan")
        g.add_edge("plan", "execute")
        g.add_edge("execute", "revise")
        g.add_edge("revise", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke({"task": task}, config={"recursion_limit": 25})
        return ArchitectureResult(
            output=final_state.get("revised_response", ""),
            state={
                "verification_question_count": len(final_state.get("verification_questions", [])),
                "changes_made_count": len(final_state.get("changes_made", [])),
            },
            trace=final_state.get("history", []),
            metadata={
                "baseline_response": final_state.get("baseline_response", ""),
                "verification_questions": final_state.get("verification_questions", []),
                "verification_answers": final_state.get("verification_answers", []),
                "changes_made": final_state.get("changes_made", []),
                "low_confidence_count": sum(
                    1 for a in final_state.get("verification_answers", []) if a.get("confidence") == "low"
                ),
                "question_count": len(final_state.get("verification_questions", [])),
            },
        )
