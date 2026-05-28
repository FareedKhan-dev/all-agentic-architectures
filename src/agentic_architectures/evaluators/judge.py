"""Generic LLM-as-Judge replacing the ~17 inline judge blocks in the old notebooks.

Usage:

    from pydantic import BaseModel, Field
    from agentic_architectures.evaluators import LLMJudge

    class CodeEval(BaseModel):
        correctness: int = Field(ge=1, le=5)
        readability: int = Field(ge=1, le=5)
        justification: str

    judge = LLMJudge(
        schema=CodeEval,
        rubric="Score the candidate code on correctness (1-5) and readability (1-5).",
    )
    result = judge.evaluate(candidate=generated_code, context={"task": original_task})
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

T = TypeVar("T", bound=BaseModel)


class LLMJudge(Generic[T]):
    """Reusable LLM-as-Judge over an arbitrary Pydantic schema."""

    def __init__(
        self,
        schema: type[T],
        rubric: str,
        llm: BaseChatModel | None = None,
        temperature: float = 0.0,
    ) -> None:
        from agentic_architectures.llm.factory import (
            get_llm,
            provider_supports_structured_output,
        )

        self.schema = schema
        self.rubric = rubric
        base = llm if llm is not None else get_llm(temperature=temperature)

        if not provider_supports_structured_output():
            # Fall back gracefully — the with_structured_output method still
            # exists, but may use JSON-mode rather than a tool-call. The
            # judge still works; the downstream user just gets a warning.
            import warnings

            warnings.warn(
                "Current provider does not advertise reliable structured output. "
                "Judge results may be lower-quality JSON parses.",
                stacklevel=2,
            )

        self._judge_llm = base.with_structured_output(schema)

    def evaluate(self, candidate: str, context: dict[str, str] | None = None) -> T:
        """Score the candidate against the rubric and return a populated schema."""
        ctx_block = ""
        if context:
            ctx_block = "\n\n### Context\n" + "\n".join(f"- **{k}**: {v}" for k, v in context.items())

        prompt = (
            f"You are an impartial evaluator.\n\n"
            f"### Rubric\n{self.rubric}\n{ctx_block}\n\n"
            f"### Candidate\n{candidate}\n\n"
            f"Return your evaluation in the requested structured format."
        )
        result = self._judge_llm.invoke(prompt)
        # `with_structured_output` returns the schema instance directly.
        return result  # type: ignore[return-value]
