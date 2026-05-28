"""Adaptive RAG — router picks no-retrieval / single-step / multi-step strategy.

Pre-classifies each query into a *complexity bucket*, then routes:
  - `no_retrieval` — answer from parametric memory.
  - `single_step` — one retrieve + one answer (plain RAG).
  - `multi_step` — agentic loop (multiple retrievals, like nb 23 Agentic RAG).

Sister architecture to **MetaController** (nb 11) — same pre-routing pattern,
specialised to RAG strategies. Sister to **Agentic RAG** (nb 23) — but
Adaptive RAG decides the strategy *up front* via a single classifier call,
not iteratively.

**Deterministic-picker pattern** (handoff §7): the classifier emits a
categorical complexity bucket; Python routes on the categorical. No numeric
score involved.

Origin: Jeong et al., *Adaptive-RAG* (NAACL 2024).
https://arxiv.org/abs/2403.14403
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult
from agentic_architectures.memory.vector import VectorMemory


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class _ComplexityClass(BaseModel):
    """Pre-classification of the query into a RAG complexity bucket."""

    complexity: Literal["no_retrieval", "single_step", "multi_step"] = Field(
        description=(
            "Routing class: "
            "'no_retrieval' = answer from parametric memory (arithmetic, common knowledge); "
            "'single_step' = one retrieval is sufficient (single-fact lookup); "
            "'multi_step' = multi-hop or follow-up retrievals needed."
        )
    )
    rationale: str = Field(description="ONE sentence explaining the classification.")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AdaptiveRAGState(TypedDict, total=False):
    task: str
    complexity: str
    classification_rationale: str
    retrievals: Annotated[list[dict[str, Any]], operator.add]
    final_answer: str
    history: Annotated[list[dict[str, Any]], operator.add]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class AdaptiveRAG(Architecture):
    """Pre-route by complexity → execute the matched RAG strategy."""

    name = "adaptive_rag"
    description = (
        "Classifier pre-routes each query into no_retrieval / single_step / "
        "multi_step buckets; architecture executes the matched strategy. "
        "Categorical router = deterministic-picker."
    )
    reference = "https://arxiv.org/abs/2403.14403"

    def __init__(
        self,
        documents: list[str] | None = None,
        vector_memory: VectorMemory | None = None,
        top_k: int = 3,
        multi_step_max_iterations: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k
        self.multi_step_max_iterations = multi_step_max_iterations
        if vector_memory is not None:
            self.memory = vector_memory
        else:
            self.memory = VectorMemory(collection_name="adaptive_rag_corpus")
            if documents:
                from langchain_core.documents import Document

                self.memory.add([Document(page_content=d) for d in documents])
        self._classifier = self.llm.with_structured_output(_ComplexityClass)

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _classify(self, state: AdaptiveRAGState) -> dict[str, Any]:
        try:
            cls = self._classifier.invoke(
                "Classify this question's RAG complexity:\n"
                "  - 'no_retrieval' for arithmetic, general knowledge, or anything answerable "
                "from a strong LLM's parametric memory.\n"
                "  - 'single_step' for a single-fact lookup (one query suffices).\n"
                "  - 'multi_step' for multi-hop questions requiring follow-up retrievals.\n\n"
                f"# Question\n{state['task']}"
            )
            return {
                "complexity": cls.complexity,
                "classification_rationale": cls.rationale,
                "history": [{"stage": "classify", "complexity": cls.complexity, "rationale": cls.rationale}],
            }
        except Exception as e:
            return {
                "complexity": "single_step",
                "classification_rationale": f"fallback after classifier error: {e}",
                "history": [{"stage": "classify", "fallback": True}],
            }

    def _no_retrieval_answer(self, state: AdaptiveRAGState) -> dict[str, Any]:
        ans = str(
            self.llm.invoke(
                f"Answer from your knowledge. No external sources available.\n\n# Question\n{state['task']}\n\nAnswer:"
            ).content
        ).strip()
        return {
            "final_answer": ans,
            "history": [{"stage": "no_retrieval_answer"}],
        }

    def _single_step_rag(self, state: AdaptiveRAGState) -> dict[str, Any]:
        docs = self.memory.search(state["task"], k=self.top_k)
        ctx = "\n\n".join(f"- {d.page_content[:400]}" for d in docs)
        ans = str(
            self.llm.invoke(
                f"Answer using the context below.\n\n# Context\n{ctx}\n\n# Question\n{state['task']}\n\nAnswer:"
            ).content
        ).strip()
        return {
            "final_answer": ans,
            "retrievals": [{"query": state["task"], "n_docs": len(docs)}],
            "history": [{"stage": "single_step_rag", "n_docs": len(docs)}],
        }

    def _multi_step_rag(self, state: AdaptiveRAGState) -> dict[str, Any]:
        # Simplified multi-step: 2 sequential retrievals, second informed by first.
        first_docs = self.memory.search(state["task"], k=self.top_k)
        first_ctx = "\n\n".join(f"- {d.page_content[:300]}" for d in first_docs)

        # Generate a follow-up query
        followup_prompt = (
            f"You retrieved this context for question '{state['task']}':\n\n{first_ctx}\n\n"
            "What ONE additional search query would best fill the remaining information gap? "
            "Output just the query, no preface."
        )
        followup_query = str(self.llm.invoke(followup_prompt).content).strip().strip('"').strip("'")
        second_docs = self.memory.search(followup_query, k=self.top_k)
        second_ctx = "\n\n".join(f"- {d.page_content[:300]}" for d in second_docs)

        ans = str(
            self.llm.invoke(
                f"Answer the question using BOTH context blocks.\n\n"
                f"# First context (query: {state['task'][:80]})\n{first_ctx}\n\n"
                f"# Second context (follow-up query: {followup_query[:80]})\n{second_ctx}\n\n"
                f"# Question\n{state['task']}\n\nAnswer:"
            ).content
        ).strip()
        return {
            "final_answer": ans,
            "retrievals": [
                {"query": state["task"], "n_docs": len(first_docs)},
                {"query": followup_query, "n_docs": len(second_docs)},
            ],
            "history": [
                {
                    "stage": "multi_step_rag",
                    "followup_query": followup_query,
                    "n_first": len(first_docs),
                    "n_second": len(second_docs),
                }
            ],
        }

    # ------------------------------------------------------------------ #
    #  Router                                                             #
    # ------------------------------------------------------------------ #

    def _route(self, state: AdaptiveRAGState) -> str:
        c = state.get("complexity", "single_step")
        if c == "no_retrieval":
            return "no_retrieval"
        elif c == "multi_step":
            return "multi_step"
        return "single_step"

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(AdaptiveRAGState)
        g.add_node("classify", self._classify)
        g.add_node("no_retrieval", self._no_retrieval_answer)
        g.add_node("single_step", self._single_step_rag)
        g.add_node("multi_step", self._multi_step_rag)
        g.add_edge(START, "classify")
        g.add_conditional_edges(
            "classify",
            self._route,
            {"no_retrieval": "no_retrieval", "single_step": "single_step", "multi_step": "multi_step"},
        )
        g.add_edge("no_retrieval", END)
        g.add_edge("single_step", END)
        g.add_edge("multi_step", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke({"task": task}, config={"recursion_limit": 25})
        retrievals = final_state.get("retrievals", [])
        return ArchitectureResult(
            output=final_state.get("final_answer", ""),
            state={
                "complexity": final_state.get("complexity"),
                "retrieval_count": len(retrievals),
            },
            trace=final_state.get("history", []),
            metadata={
                "complexity": final_state.get("complexity"),
                "classification_rationale": final_state.get("classification_rationale"),
                "retrieval_count": len(retrievals),
                "retrievals": retrievals,
            },
        )
