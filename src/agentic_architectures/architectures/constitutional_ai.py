"""Constitutional AI — critique + revise against a written constitution.

Pipeline:
  1. Generate baseline response.
  2. For each constitution rule, the LLM judges pass/fail (categorical).
  3. **Python AND** over all rule-pass booleans → deciding signal (`all_passed: bool`).
  4. If any rule failed, revise the response addressing the failures; loop.

**Deterministic-picker pattern** (handoff §7): per-rule pass/fail is a
`Literal['pass', 'fail']` field; Python composes the overall pass via `all()`.
The LLM never emits a numeric constitution-score.

Origin: Bai et al. (Anthropic), *Constitutional AI* (2022).
https://arxiv.org/abs/2212.08073
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult

DEFAULT_CONSTITUTION: list[str] = [
    "Don't include personal opinions or political stances unless explicitly asked.",
    "Cite or hedge appropriately when making factual claims; don't state guesses as facts.",
    "Be concise — keep responses under 200 words unless the task requires more.",
    "Don't include harmful instructions, even hypothetically.",
]


class _RuleVerdict(BaseModel):
    rule_index: int
    verdict: Literal["pass", "fail"]
    rationale: str = Field(description="ONE sentence — what specifically passes or fails.")


class _CritiqueResult(BaseModel):
    verdicts: list[_RuleVerdict] = Field(min_length=1)
    overall_critique: str = Field(description="One paragraph summarising any failures.")


class ConstitutionalAIState(TypedDict, total=False):
    task: str
    iteration: int
    max_iterations: int
    response: str
    rule_verdicts: list[dict[str, Any]]
    all_passed: bool
    failures: list[str]
    final_answer: str
    history: Annotated[list[dict[str, Any]], operator.add]


class ConstitutionalAI(Architecture):
    """Generate → critique against constitution → revise → loop until all pass."""

    name = "constitutional_ai"
    description = (
        "Generate, critique against a written constitution (per-rule pass/fail), "
        "revise. Loop until all rules pass or max_iterations exhausted."
    )
    reference = "https://arxiv.org/abs/2212.08073"

    def __init__(
        self,
        constitution: list[str] | None = None,
        max_iterations: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.constitution: list[str] = list(constitution) if constitution else list(DEFAULT_CONSTITUTION)
        self.max_iterations = max_iterations
        self._critique = self.llm.with_structured_output(_CritiqueResult)

    def _constitution_block(self) -> str:
        return "\n".join(f"  [rule {i}] {r}" for i, r in enumerate(self.constitution))

    def _generate(self, state: ConstitutionalAIState) -> dict[str, Any]:
        prompt = f"# Task\n{state['task']}\n\nProduce your best response."
        resp = str(self.llm.invoke(prompt).content).strip()
        return {
            "response": resp,
            "iteration": 0,
            "history": [{"stage": "generate", "len": len(resp)}],
        }

    def _critique_node(self, state: ConstitutionalAIState) -> dict[str, Any]:
        try:
            c = self._critique.invoke(
                "Critique the response against EACH rule of the constitution. "
                "For every rule, emit a verdict ('pass' or 'fail') + rationale.\n\n"
                f"## Constitution\n{self._constitution_block()}\n\n"
                f"## Task\n{state['task']}\n\n"
                f"## Response\n{state['response']}"
            )
            verdicts = [v.model_dump() for v in c.verdicts]
            # Python AND on rule passes
            failures = [v for v in verdicts if v["verdict"] == "fail"]
            all_passed = len(failures) == 0
            return {
                "rule_verdicts": verdicts,
                "all_passed": all_passed,
                "failures": [v["rationale"] for v in failures],
                "history": [{
                    "stage": "critique",
                    "n_pass": sum(1 for v in verdicts if v["verdict"] == "pass"),
                    "n_fail": len(failures),
                    "all_passed": all_passed,
                }],
            }
        except Exception as e:
            return {
                "rule_verdicts": [],
                "all_passed": True,  # fail-open
                "failures": [],
                "history": [{"stage": "critique", "error": str(e)}],
            }

    def _revise(self, state: ConstitutionalAIState) -> dict[str, Any]:
        failures_block = "\n".join(f"  - {f}" for f in state.get("failures", []))
        revised = str(self.llm.invoke(
            f"# Task\n{state['task']}\n\n"
            f"# Constitution\n{self._constitution_block()}\n\n"
            f"# Previous response (violates some rules)\n{state['response']}\n\n"
            f"# Failed rule rationales\n{failures_block}\n\n"
            "Rewrite the response addressing every failure while still solving the task."
        ).content).strip()
        return {
            "response": revised,
            "iteration": state.get("iteration", 0) + 1,
            "history": [{"stage": "revise", "iteration": state.get("iteration", 0) + 1}],
        }

    def _finalize(self, state: ConstitutionalAIState) -> dict[str, Any]:
        return {"final_answer": state.get("response", "")}

    def _should_continue(self, state: ConstitutionalAIState) -> str:
        if state.get("all_passed", False):
            return "finalize"
        if state.get("iteration", 0) >= state.get("max_iterations", self.max_iterations):
            return "finalize"
        return "revise"

    def build(self) -> Any:
        g: StateGraph = StateGraph(ConstitutionalAIState)
        g.add_node("generate", self._generate)
        g.add_node("critique", self._critique_node)
        g.add_node("revise", self._revise)
        g.add_node("finalize", self._finalize)
        g.add_edge(START, "generate")
        g.add_edge("generate", "critique")
        g.add_conditional_edges(
            "critique", self._should_continue,
            {"revise": "revise", "finalize": "finalize"},
        )
        g.add_edge("revise", "critique")
        g.add_edge("finalize", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke(
            {"task": task, "max_iterations": self.max_iterations},
            config={"recursion_limit": max(50, self.max_iterations * 6)},
        )
        verdicts = final_state.get("rule_verdicts", [])
        return ArchitectureResult(
            output=final_state.get("final_answer", ""),
            state={
                "iterations": final_state.get("iteration", 0),
                "all_passed": final_state.get("all_passed", False),
            },
            trace=final_state.get("history", []),
            metadata={
                "iterations": final_state.get("iteration", 0),
                "max_iterations": self.max_iterations,
                "all_passed": final_state.get("all_passed", False),
                "rule_verdicts": verdicts,
                "n_pass": sum(1 for v in verdicts if v["verdict"] == "pass"),
                "n_fail": sum(1 for v in verdicts if v["verdict"] == "fail"),
                "n_rules": len(self.constitution),
                "failures": final_state.get("failures", []),
            },
        )
