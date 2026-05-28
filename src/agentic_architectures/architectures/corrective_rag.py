"""Corrective RAG (CRAG) — grade retrieved docs, fall back to web if poor.

Plain RAG retrieves and uses the docs as-is. CRAG adds a **grading step**:
each retrieved doc is scored relevant / ambiguous / irrelevant, and the
architecture routes accordingly:

  - majority relevant → answer using retrieved docs
  - majority irrelevant → web-search fallback
  - mixed → use retrieved + web

**Deterministic-picker pattern applied** (handoff §7): the LLM grader commits
to a **categorical** relevance label per doc, not a numeric score. Python then
composes the routing decision from the count of each label — the deciding
signal is `(n_relevant, n_irrelevant)`, never an LLM-emitted number.

Origin: Yan et al., *Corrective Retrieval Augmented Generation* (2024).
https://arxiv.org/abs/2401.15884
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult
from agentic_architectures.memory.vector import VectorMemory


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class _DocGrade(BaseModel):
    """Per-document relevance grade — categorical (deterministic-picker)."""

    relevance: Literal["relevant", "ambiguous", "irrelevant"] = Field(
        description="Categorical assessment of this doc's relevance to the question. "
                    "'relevant' = directly answers part of the question; "
                    "'ambiguous' = related but doesn't directly answer; "
                    "'irrelevant' = off-topic or wrong entity."
    )
    rationale: str = Field(description="ONE sentence justifying the grade.")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class CorrectiveRAGState(TypedDict, total=False):
    task: str
    retrieved_docs: list[str]
    doc_grades: list[dict[str, str]]
    route: Literal["use_retrieved", "use_web", "use_mixed"]
    web_docs: list[str]
    final_answer: str
    history: Annotated[list[dict[str, Any]], operator.add]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class CorrectiveRAG(Architecture):
    """Retrieve → grade per-doc → route → answer."""

    name = "corrective_rag"
    description = (
        "Retrieve from a vector corpus, grade each doc's relevance via a "
        "categorical LLM judgement, route to: (a) answer-from-retrieved, "
        "(b) web-search fallback, or (c) mixed. Deterministic-picker pattern "
        "on the routing decision."
    )
    reference = "https://arxiv.org/abs/2401.15884"

    def __init__(
        self,
        documents: list[str] | None = None,
        vector_memory: VectorMemory | None = None,
        web_search_fn: Callable[[str], list[str]] | None = None,
        top_k: int = 3,
        relevance_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            documents: optional list to seed a fresh VectorMemory.
            vector_memory: pre-built memory; takes precedence over `documents`.
            web_search_fn: callable `(query) -> list[str]` returning web snippets.
                If None, web fallback returns an empty list.
            top_k: how many docs to retrieve.
            relevance_threshold: fraction of docs that must be 'relevant' to
                route directly to use_retrieved (else fall back / mix).
        """
        super().__init__(**kwargs)
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        self.web_search_fn = web_search_fn
        if vector_memory is not None:
            self.memory = vector_memory
        else:
            self.memory = VectorMemory(collection_name="corrective_rag_corpus")
            if documents:
                from langchain_core.documents import Document
                self.memory.add([Document(page_content=d) for d in documents])
        self._grader = self.llm.with_structured_output(_DocGrade)

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _retrieve(self, state: CorrectiveRAGState) -> dict[str, Any]:
        docs = self.memory.search(state["task"], k=self.top_k)
        texts = [d.page_content for d in docs]
        return {
            "retrieved_docs": texts,
            "history": [{"stage": "retrieve", "n_docs": len(texts)}],
        }

    def _grade(self, state: CorrectiveRAGState) -> dict[str, Any]:
        grades: list[dict[str, str]] = []
        for doc in state["retrieved_docs"]:
            try:
                g = self._grader.invoke(
                    f"# Question\n{state['task']}\n\n"
                    f"# Document\n{doc[:500]}\n\n"
                    "Grade this document's relevance to the question. Be strict — "
                    "'relevant' only if the doc directly contains material that helps "
                    "answer the question."
                )
                grades.append({"relevance": g.relevance, "rationale": g.rationale})
            except Exception as e:
                grades.append({"relevance": "irrelevant", "rationale": f"(grade error: {e})"})
        return {
            "doc_grades": grades,
            "history": [{
                "stage": "grade",
                "n_relevant": sum(1 for g in grades if g["relevance"] == "relevant"),
                "n_ambiguous": sum(1 for g in grades if g["relevance"] == "ambiguous"),
                "n_irrelevant": sum(1 for g in grades if g["relevance"] == "irrelevant"),
            }],
        }

    def _route(self, state: CorrectiveRAGState) -> dict[str, Any]:
        """Python-composed route — deterministic-picker."""
        grades = state["doc_grades"]
        n = len(grades)
        if n == 0:
            route = "use_web"
        else:
            relevant = sum(1 for g in grades if g["relevance"] == "relevant")
            irrelevant = sum(1 for g in grades if g["relevance"] == "irrelevant")
            rel_frac = relevant / n
            if rel_frac >= self.relevance_threshold:
                route = "use_retrieved"
            elif irrelevant == n:
                route = "use_web"
            else:
                route = "use_mixed"
        return {
            "route": route,
            "history": [{"stage": "route", "route": route}],
        }

    def _web_search(self, state: CorrectiveRAGState) -> dict[str, Any]:
        if not self.web_search_fn:
            return {
                "web_docs": [],
                "history": [{"stage": "web_search", "skipped": "no web_search_fn"}],
            }
        try:
            docs = self.web_search_fn(state["task"])
        except Exception as e:
            docs = [f"(web search failed: {e})"]
        return {
            "web_docs": docs,
            "history": [{"stage": "web_search", "n_docs": len(docs)}],
        }

    def _answer(self, state: CorrectiveRAGState) -> dict[str, Any]:
        route = state.get("route", "use_retrieved")
        retrieved = state.get("retrieved_docs", [])
        web = state.get("web_docs", [])

        if route == "use_retrieved":
            context_label = "retrieved corpus"
            ctx = retrieved
        elif route == "use_web":
            context_label = "web search"
            ctx = web
        else:  # use_mixed
            context_label = "retrieved corpus + web search"
            ctx = retrieved + web

        ctx_block = "\n\n".join(f"- {d[:400]}" for d in ctx) if ctx else "(no documents available)"
        prompt = (
            f"Answer the question using ONLY the context below (from {context_label}). "
            "If the context doesn't contain the answer, say 'I don't have enough information.'\n\n"
            f"# Context\n{ctx_block}\n\n"
            f"# Question\n{state['task']}\n\nAnswer:"
        )
        ans = str(self.llm.invoke(prompt).content).strip()
        return {
            "final_answer": ans,
            "history": [{"stage": "answer", "route": route, "context_source": context_label}],
        }

    # ------------------------------------------------------------------ #
    #  Router                                                             #
    # ------------------------------------------------------------------ #

    def _next_after_route(self, state: CorrectiveRAGState) -> str:
        route = state.get("route", "use_retrieved")
        if route in ("use_web", "use_mixed"):
            return "web_search"
        return "answer"

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(CorrectiveRAGState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade", self._grade)
        g.add_node("route", self._route)
        g.add_node("web_search", self._web_search)
        g.add_node("answer", self._answer)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_edge("grade", "route")
        g.add_conditional_edges(
            "route", self._next_after_route,
            {"web_search": "web_search", "answer": "answer"},
        )
        g.add_edge("web_search", "answer")
        g.add_edge("answer", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke({"task": task}, config={"recursion_limit": 25})
        grades = final_state.get("doc_grades", [])
        n_rel = sum(1 for g in grades if g["relevance"] == "relevant")
        n_amb = sum(1 for g in grades if g["relevance"] == "ambiguous")
        n_irr = sum(1 for g in grades if g["relevance"] == "irrelevant")
        return ArchitectureResult(
            output=final_state.get("final_answer", ""),
            state={
                "route": final_state.get("route"),
                "n_retrieved": len(final_state.get("retrieved_docs", [])),
                "n_web": len(final_state.get("web_docs", [])),
            },
            trace=final_state.get("history", []),
            metadata={
                "route": final_state.get("route"),
                "n_retrieved": len(final_state.get("retrieved_docs", [])),
                "n_relevant": n_rel,
                "n_ambiguous": n_amb,
                "n_irrelevant": n_irr,
                "relevance_fraction": n_rel / len(grades) if grades else 0.0,
                "n_web": len(final_state.get("web_docs", [])),
                "doc_grades": grades,
                "threshold": self.relevance_threshold,
            },
        )
