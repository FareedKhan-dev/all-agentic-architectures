"""Agentic RAG — the agent decides WHEN and WHAT to retrieve.

Plain RAG always retrieves first, then answers. Agentic RAG treats retrieval
as a tool the agent can decide to call (or not), with what query, possibly
multiple times. It's a small ReAct loop over a single retrieve-tool.

State machine
-------------
    [decide] →─ action=retrieve ─→ [retrieve] ──┐
        ↑                                       │
        └───────────────────────────────────────┘
        action=answer
              └→ [answer] → END

Builds on **Tool Use** (notebook 02): same agent-decides-which-tool pattern,
but specialised to retrieval. Each turn the agent commits to either calling
the retriever (with a specific query) or producing the final answer.

No LLM-as-Scorer step → no flat-scoring pathology. The deciding signal is a
*categorical action* (retrieve vs answer), not a numeric judgement.
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
class _AgentDecision(BaseModel):
    """One step of the agentic-RAG loop."""

    action: Literal["retrieve", "answer"] = Field(
        description="Either 'retrieve' (need more context) or 'answer' (have enough to answer)."
    )
    query: str = Field(
        default="",
        description="If action='retrieve', the search query — focused, specific. Empty string if action='answer'.",
    )
    answer: str = Field(
        default="",
        description="If action='answer', the final answer using retrieved context. Empty string if action='retrieve'.",
    )
    rationale: str = Field(description="ONE sentence: why this action right now.")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AgenticRAGState(TypedDict, total=False):
    task: str
    iteration: int
    max_iterations: int
    retrievals: Annotated[list[dict[str, Any]], operator.add]
    last_decision: dict[str, Any]
    final_answer: str
    history: Annotated[list[dict[str, Any]], operator.add]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class AgenticRAG(Architecture):
    """ReAct-style agent with a single retrieve-tool."""

    name = "agentic_rag"
    description = (
        "Agent loop: each step the LLM decides whether to retrieve (with what "
        "query) or to answer from what it already has. The deciding signal is "
        "a categorical action — no LLM-as-Scorer."
    )
    reference = "LangGraph reference / Adaptive-RAG (Jeong et al. 2024)"

    def __init__(
        self,
        documents: list[str] | None = None,
        vector_memory: VectorMemory | None = None,
        max_iterations: int = 4,
        top_k: int = 3,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            documents: optional list of source documents to seed the corpus.
            vector_memory: a pre-built VectorMemory; takes precedence over `documents`.
            max_iterations: hard cap on decide → retrieve cycles.
            top_k: number of documents to return per retrieve call.
        """
        super().__init__(**kwargs)
        self.max_iterations = max_iterations
        self.top_k = top_k
        if vector_memory is not None:
            self.memory = vector_memory
        else:
            self.memory = VectorMemory(collection_name="agentic_rag_corpus")
            if documents:
                from langchain_core.documents import Document

                self.memory.add([Document(page_content=d) for d in documents])
        self._decider = self.llm.with_structured_output(_AgentDecision)

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _format_retrievals(self, retrievals: list[dict[str, Any]]) -> str:
        if not retrievals:
            return "(no retrievals yet)"
        blocks: list[str] = []
        for r in retrievals:
            doc_block = "\n".join(f"  - {d[:240]}" for d in r["docs"])
            blocks.append(f"### Query: {r['query']}\n{doc_block}")
        return "\n\n".join(blocks)

    def _decide(self, state: AgenticRAGState) -> dict[str, Any]:
        iter_count = state.get("iteration", 0)
        retrievals_block = self._format_retrievals(state.get("retrievals", []))
        prompt = (
            "You are an agent with access to a single tool: `retrieve(query)`, which "
            "returns the top-K most-similar documents from a corpus. Each step you "
            "decide whether to call `retrieve` (with a query) or to commit to a "
            "final `answer` based on what you have.\n\n"
            f"## Task\n{state['task']}\n\n"
            f"## Iteration {iter_count + 1}/{state.get('max_iterations', self.max_iterations)}\n\n"
            f"## Retrievals so far\n{retrievals_block}\n\n"
            "Rules:\n"
            "  - Call `retrieve` ONLY when you genuinely lack a fact needed for the answer. "
            "Do NOT retrieve redundantly.\n"
            "  - When you have enough information, set action='answer' and provide the "
            "final answer that explicitly uses the retrieved facts.\n"
            "  - At the last iteration, you MUST answer (no more retrievals allowed)."
        )
        # Force answer on the last iteration to avoid running over budget.
        if iter_count + 1 >= state.get("max_iterations", self.max_iterations):
            prompt += "\n\nNOTE: This is the FINAL iteration. You MUST set action='answer'."
        try:
            decision = self._decider.invoke(prompt)
            dec_dict = decision.model_dump()
        except Exception as e:
            dec_dict = {
                "action": "answer",
                "query": "",
                "answer": f"(decider failed: {e})",
                "rationale": "fallback after structured-output failure",
            }
        return {
            "last_decision": dec_dict,
            "iteration": iter_count + 1,
            "history": [{"stage": "decide", "iteration": iter_count + 1, "decision": dec_dict}],
        }

    def _retrieve(self, state: AgenticRAGState) -> dict[str, Any]:
        query = state["last_decision"].get("query", "").strip()
        if not query:
            # Degenerate case: decider said retrieve but gave no query
            return {
                "retrievals": [{"query": "(empty)", "docs": []}],
                "history": [{"stage": "retrieve", "query": "(empty)", "n_docs": 0}],
            }
        docs = self.memory.search(query, k=self.top_k)
        doc_texts = [d.page_content for d in docs]
        return {
            "retrievals": [{"query": query, "docs": doc_texts}],
            "history": [{"stage": "retrieve", "query": query, "n_docs": len(doc_texts)}],
        }

    def _answer(self, state: AgenticRAGState) -> dict[str, Any]:
        final = state["last_decision"].get("answer", "").strip()
        return {
            "final_answer": final or "(decider produced no answer)",
            "history": [{"stage": "answer", "final": final}],
        }

    # ------------------------------------------------------------------ #
    #  Router                                                             #
    # ------------------------------------------------------------------ #

    def _route_decision(self, state: AgenticRAGState) -> str:
        action = state["last_decision"].get("action", "answer")
        if action == "retrieve" and state.get("iteration", 0) < state.get("max_iterations", self.max_iterations):
            return "retrieve"
        return "answer"

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(AgenticRAGState)
        g.add_node("decide", self._decide)
        g.add_node("retrieve", self._retrieve)
        g.add_node("answer", self._answer)
        g.add_edge(START, "decide")
        g.add_conditional_edges(
            "decide",
            self._route_decision,
            {"retrieve": "retrieve", "answer": "answer"},
        )
        g.add_edge("retrieve", "decide")
        g.add_edge("answer", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke(
            {"task": task, "max_iterations": self.max_iterations},
            config={"recursion_limit": max(50, self.max_iterations * 4)},
        )
        retrievals = final_state.get("retrievals", [])
        return ArchitectureResult(
            output=final_state.get("final_answer", ""),
            state={
                "retrieval_count": len(retrievals),
                "iterations_used": final_state.get("iteration", 0),
            },
            trace=final_state.get("history", []),
            metadata={
                "retrieval_count": len(retrievals),
                "iterations_used": final_state.get("iteration", 0),
                "max_iterations": self.max_iterations,
                "retrievals": retrievals,
                "final_decision": final_state.get("last_decision", {}),
            },
        )
