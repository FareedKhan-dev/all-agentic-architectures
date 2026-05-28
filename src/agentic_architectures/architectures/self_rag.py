"""Self-RAG — retrieve-on-demand with reflection tokens.

Self-RAG adds per-document **reflection tokens** to RAG: after retrieval, the
model emits categorical labels assessing each doc's relevance and support
quality. Python then composes a keep/drop decision per doc.

  1. **Decide retrieval** — `needs_retrieval: bool` (or skip).
  2. **Retrieve** — vector search.
  3. **Reflect per doc** — emit `(is_relevant, is_supported, is_useful)` categoricals.
  4. **Compose** — Python keeps docs that pass the per-doc gate.
  5. **Answer** — generate from kept docs only.

**Deterministic-picker pattern** (handoff §7): every reflection token is a
3-way categorical (e.g., `Literal['fully_relevant','partially_relevant','not_relevant']`),
not a number. Python composes the keep/drop boolean per doc and the final
useful-fraction signal.

Origin: Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique
through Self-Reflection* (2024). https://arxiv.org/abs/2310.11511
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
class _RetrieveDecision(BaseModel):
    needs_retrieval: bool = Field(
        description="True iff answering this requires retrieving from the corpus. "
        "False if the model can answer from its parametric knowledge."
    )
    rationale: str = Field(description="ONE sentence explaining the decision.")


class _ReflectionTokens(BaseModel):
    """Per-document reflection tokens — all categorical (deterministic-picker)."""

    is_relevant: Literal["fully_relevant", "partially_relevant", "not_relevant"] = Field(
        description="Does this document address the question?"
    )
    is_supported: Literal["fully_supported", "partially_supported", "no_support"] = Field(
        description="If you used this document as evidence, how well-grounded would the answer be?"
    )
    is_useful: Literal["very_useful", "somewhat_useful", "not_useful"] = Field(
        description="Overall usefulness for producing a high-quality answer."
    )
    rationale: str = Field(description="ONE sentence — be specific about which signal drove the assessment.")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class SelfRAGState(TypedDict, total=False):
    task: str
    needs_retrieval: bool
    retrieved_docs: list[str]
    reflection_tokens: list[dict[str, str]]
    kept_doc_indices: list[int]
    final_answer: str
    history: Annotated[list[dict[str, Any]], operator.add]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class SelfRAG(Architecture):
    """Retrieve → reflection tokens per doc → Python keeps usable docs → answer."""

    name = "self_rag"
    description = (
        "RAG with per-document reflection tokens. Each retrieved doc gets "
        "categorical (is_relevant, is_supported, is_useful) tokens; Python "
        "composes the keep/drop boolean. Deterministic-picker on the tokens."
    )
    reference = "https://arxiv.org/abs/2310.11511"

    def __init__(
        self,
        documents: list[str] | None = None,
        vector_memory: VectorMemory | None = None,
        top_k: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k
        if vector_memory is not None:
            self.memory = vector_memory
        else:
            self.memory = VectorMemory(collection_name="self_rag_corpus")
            if documents:
                from langchain_core.documents import Document

                self.memory.add([Document(page_content=d) for d in documents])
        self._decider = self.llm.with_structured_output(_RetrieveDecision)
        self._reflector = self.llm.with_structured_output(_ReflectionTokens)

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _decide_retrieval(self, state: SelfRAGState) -> dict[str, Any]:
        try:
            d = self._decider.invoke(
                f"Question: {state['task']}\n\n"
                "Does answering this question require consulting an external corpus, "
                "or can you answer from your parametric knowledge? Set "
                "`needs_retrieval=True` if external retrieval is needed."
            )
            return {
                "needs_retrieval": d.needs_retrieval,
                "history": [
                    {"stage": "decide_retrieval", "needs_retrieval": d.needs_retrieval, "rationale": d.rationale}
                ],
            }
        except Exception:
            return {"needs_retrieval": True, "history": [{"stage": "decide_retrieval", "fallback": True}]}

    def _retrieve(self, state: SelfRAGState) -> dict[str, Any]:
        if not state.get("needs_retrieval", True):
            return {
                "retrieved_docs": [],
                "history": [{"stage": "retrieve", "skipped": True}],
            }
        docs = self.memory.search(state["task"], k=self.top_k)
        texts = [d.page_content for d in docs]
        return {
            "retrieved_docs": texts,
            "history": [{"stage": "retrieve", "n_docs": len(texts)}],
        }

    def _reflect(self, state: SelfRAGState) -> dict[str, Any]:
        docs = state.get("retrieved_docs", [])
        tokens: list[dict[str, str]] = []
        for doc in docs:
            try:
                rt = self._reflector.invoke(
                    f"# Question\n{state['task']}\n\n"
                    f"# Document\n{doc[:500]}\n\n"
                    "Emit reflection tokens for this document."
                )
                tokens.append(
                    {
                        "is_relevant": rt.is_relevant,
                        "is_supported": rt.is_supported,
                        "is_useful": rt.is_useful,
                        "rationale": rt.rationale,
                    }
                )
            except Exception as e:
                tokens.append(
                    {
                        "is_relevant": "not_relevant",
                        "is_supported": "no_support",
                        "is_useful": "not_useful",
                        "rationale": f"(reflection error: {e})",
                    }
                )
        return {
            "reflection_tokens": tokens,
            "history": [
                {
                    "stage": "reflect",
                    "n_tokens": len(tokens),
                    "fully_relevant": sum(1 for t in tokens if t["is_relevant"] == "fully_relevant"),
                    "no_support": sum(1 for t in tokens if t["is_supported"] == "no_support"),
                }
            ],
        }

    def _compose_keep(self, state: SelfRAGState) -> dict[str, Any]:
        """Python decides which docs to keep — pure deterministic-picker."""
        tokens = state.get("reflection_tokens", [])
        kept: list[int] = []
        for i, t in enumerate(tokens):
            # Keep iff relevant AND has support
            if t["is_relevant"] != "not_relevant" and t["is_supported"] != "no_support":
                kept.append(i)
        return {
            "kept_doc_indices": kept,
            "history": [
                {
                    "stage": "compose_keep",
                    "kept": kept,
                    "dropped": [i for i in range(len(tokens)) if i not in kept],
                }
            ],
        }

    def _answer(self, state: SelfRAGState) -> dict[str, Any]:
        docs = state.get("retrieved_docs", [])
        kept_idx = state.get("kept_doc_indices", [])
        kept_docs = [docs[i] for i in kept_idx if i < len(docs)]

        if not kept_docs and docs:
            # All docs failed reflection; admit gap
            return {
                "final_answer": (
                    "I retrieved relevant material but none of it passed the per-doc "
                    "reflection check. I don't have enough verified context to answer."
                ),
                "history": [{"stage": "answer", "kept_count": 0, "all_failed_reflection": True}],
            }
        if not docs:
            # No retrieval was done — answer from parametric memory
            ans = str(
                self.llm.invoke(
                    f"# Question\n{state['task']}\n\n"
                    "Answer from your parametric knowledge (no external context was retrieved)."
                ).content
            ).strip()
            return {
                "final_answer": ans,
                "history": [{"stage": "answer", "kept_count": 0, "no_retrieval": True}],
            }

        ctx_block = "\n\n".join(f"- {d[:400]}" for d in kept_docs)
        ans = str(
            self.llm.invoke(
                f"Answer using ONLY the context below (per-doc reflection-checked).\n\n"
                f"# Context\n{ctx_block}\n\n# Question\n{state['task']}\n\nAnswer:"
            ).content
        ).strip()
        return {
            "final_answer": ans,
            "history": [{"stage": "answer", "kept_count": len(kept_docs)}],
        }

    # ------------------------------------------------------------------ #
    #  Router                                                             #
    # ------------------------------------------------------------------ #

    def _after_decide(self, state: SelfRAGState) -> str:
        return "retrieve" if state.get("needs_retrieval", True) else "answer"

    def _after_retrieve(self, state: SelfRAGState) -> str:
        return "reflect" if state.get("retrieved_docs") else "answer"

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(SelfRAGState)
        g.add_node("decide_retrieval", self._decide_retrieval)
        g.add_node("retrieve", self._retrieve)
        g.add_node("reflect", self._reflect)
        g.add_node("compose_keep", self._compose_keep)
        g.add_node("answer", self._answer)
        g.add_edge(START, "decide_retrieval")
        g.add_conditional_edges(
            "decide_retrieval",
            self._after_decide,
            {"retrieve": "retrieve", "answer": "answer"},
        )
        g.add_conditional_edges(
            "retrieve",
            self._after_retrieve,
            {"reflect": "reflect", "answer": "answer"},
        )
        g.add_edge("reflect", "compose_keep")
        g.add_edge("compose_keep", "answer")
        g.add_edge("answer", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke({"task": task}, config={"recursion_limit": 25})
        tokens = final_state.get("reflection_tokens", [])
        kept = final_state.get("kept_doc_indices", [])
        return ArchitectureResult(
            output=final_state.get("final_answer", ""),
            state={
                "needs_retrieval": final_state.get("needs_retrieval"),
                "n_retrieved": len(final_state.get("retrieved_docs", [])),
                "n_kept": len(kept),
            },
            trace=final_state.get("history", []),
            metadata={
                "needs_retrieval": final_state.get("needs_retrieval"),
                "n_retrieved": len(final_state.get("retrieved_docs", [])),
                "n_kept": len(kept),
                "kept_indices": kept,
                "reflection_tokens": tokens,
                "n_fully_relevant": sum(1 for t in tokens if t["is_relevant"] == "fully_relevant"),
                "n_no_support": sum(1 for t in tokens if t["is_supported"] == "no_support"),
                "n_very_useful": sum(1 for t in tokens if t["is_useful"] == "very_useful"),
            },
        )
