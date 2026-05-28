"""GraphRAG — knowledge graph + community summaries for global questions.

Extends [Graph Memory (nb 12)](../graph_memory.py) with two additions:
  1. **Community detection** on the extracted entity graph (via NetworkX
     greedy-modularity).
  2. **Community summaries** — one LLM-summary per community, precomputed
     once at ingestion time.

For *global* questions ("what are the main themes?"), GraphRAG composes the
answer from community summaries. For *local* questions ("what does the graph
say about X?"), it falls back to entity-anchored traversal (same as nb 12).

A simplified faithful-to-paper version of Microsoft's GraphRAG. The original
recurses community summaries hierarchically; we do a single flat level for
notebook-scale clarity.

Origin: Edge et al., *From Local to Global: A GraphRAG Approach* (Microsoft 2024).
https://arxiv.org/abs/2404.16130
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

import networkx as nx
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult
from agentic_architectures.memory.semantic import SemanticMemory


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class _IngestionTriple(BaseModel):
    subject: str = Field(description="Specific named entity (capitalised).")
    predicate: str = Field(description="Short relation verb (snake_case).")
    object: str = Field(description="Specific named entity or short literal.")


class _IngestionResult(BaseModel):
    triples: list[_IngestionTriple] = Field(default_factory=list)


class _QuestionScope(BaseModel):
    """Pre-classify question as local (entity-anchored) or global (about themes)."""

    scope: Literal["local", "global"] = Field(
        description="'local' for entity-specific questions ('who founded X?'); "
                    "'global' for theme/summary questions ('what are the main themes?')."
    )
    target_entities: list[str] = Field(
        default_factory=list,
        description="If scope='local', the entities the question is about (1-3 names). Empty for global.",
    )
    rationale: str = Field(description="ONE sentence.")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class GraphRAGState(TypedDict, total=False):
    task: str
    scope: str
    target_entities: list[str]
    context_block: str
    final_answer: str
    history: Annotated[list[dict[str, Any]], operator.add]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class GraphRAG(Architecture):
    """Build knowledge graph + community summaries; route queries local vs global."""

    name = "graph_rag"
    description = (
        "Extracts entity/relation triples from a corpus into a knowledge graph, "
        "detects communities, summarises each. Local questions traverse entity "
        "neighbourhoods; global questions consult community summaries."
    )
    reference = "https://arxiv.org/abs/2404.16130"

    def __init__(
        self,
        documents: list[str] | None = None,
        semantic_memory: SemanticMemory | None = None,
        max_communities: int = 6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_communities = max_communities
        self.semantic = semantic_memory if semantic_memory is not None else SemanticMemory()
        self._extractor = self.llm.with_structured_output(_IngestionResult)
        self._classifier = self.llm.with_structured_output(_QuestionScope)
        self.documents = documents or []
        self.communities: list[list[str]] = []
        self.community_summaries: list[str] = []
        if self.documents:
            self._ingest(self.documents)
            self._build_communities()

    # ------------------------------------------------------------------ #
    #  Build phase — done once at construction                            #
    # ------------------------------------------------------------------ #

    def _ingest(self, docs: list[str]) -> None:
        for d in docs:
            try:
                result = self._extractor.invoke(
                    "Extract atomic (subject, predicate, object) triples about "
                    "specific named entities from the text. Skip generic claims.\n\n"
                    f"Text: {d}"
                )
                for t in result.triples:
                    self.semantic.add_fact(t.subject, t.predicate, t.object)
            except Exception:
                continue

    def _build_communities(self) -> None:
        """Detect entity communities via NetworkX greedy modularity, then summarise."""
        backend = self.semantic.backend
        # Pull the underlying NetworkX graph if backend supports it
        nx_graph = (
            getattr(backend, "_g", None)
            or getattr(backend, "_graph", None)
            or getattr(backend, "graph", None)
        )
        if nx_graph is None or len(nx_graph.nodes) == 0:
            return
        # Convert to undirected for community detection
        ug = nx_graph.to_undirected() if nx_graph.is_directed() else nx_graph
        try:
            communities = list(nx.community.greedy_modularity_communities(ug))
        except Exception:
            communities = [set(ug.nodes())]
        # Keep top-N largest
        communities = sorted(communities, key=len, reverse=True)[: self.max_communities]
        self.communities = [list(c) for c in communities]
        # Summarise each community via the LLM
        for entities in self.communities:
            facts_lines: list[str] = []
            for ent in entities[:15]:  # cap to avoid huge prompts
                facts = self.semantic.facts_about(ent, depth=1)
                for f in facts[:5]:
                    facts_lines.append(f"{f.get('subject')} --[{f.get('predicate')}]--> {f.get('object')}")
            prompt = (
                "Summarise this community of related facts in 2-3 sentences, naming "
                "the key entities and what links them.\n\n"
                + "\n".join(facts_lines[:30])
            )
            try:
                summary = str(self.llm.invoke(prompt).content).strip()
            except Exception as e:
                summary = f"(summary failed: {e})"
            self.community_summaries.append(summary)

    # ------------------------------------------------------------------ #
    #  Query nodes                                                        #
    # ------------------------------------------------------------------ #

    def _classify(self, state: GraphRAGState) -> dict[str, Any]:
        try:
            cls = self._classifier.invoke(
                "Classify this question:\n"
                "  - 'local' = asks about specific named entities or facts\n"
                "  - 'global' = asks about themes, topics, organisational structure\n\n"
                f"# Question\n{state['task']}"
            )
            return {
                "scope": cls.scope,
                "target_entities": list(cls.target_entities),
                "history": [{"stage": "classify", "scope": cls.scope, "entities": list(cls.target_entities)}],
            }
        except Exception:
            return {"scope": "local", "target_entities": [], "history": [{"stage": "classify", "fallback": True}]}

    def _build_context(self, state: GraphRAGState) -> dict[str, Any]:
        scope = state.get("scope", "local")
        if scope == "global":
            # Use community summaries
            block_lines = [
                f"### Community {i+1} ({len(self.communities[i]) if i < len(self.communities) else '?'} entities)\n{s}"
                for i, s in enumerate(self.community_summaries)
            ]
            ctx = "\n\n".join(block_lines) or "(no community summaries built)"
        else:
            # Local: entity neighbourhood
            entities = state.get("target_entities", [])
            if not entities:
                # No entities identified — fall back to all community summaries
                ctx = "\n\n".join(self.community_summaries[:3]) or "(no graph context)"
            else:
                lines: list[str] = []
                for ent in entities:
                    facts = self.semantic.facts_about(ent, depth=2)
                    for f in facts[:15]:
                        lines.append(f"{f.get('subject')} --[{f.get('predicate')}]--> {f.get('object')}")
                ctx = "\n".join(lines) or f"(no facts found about: {entities})"
        return {
            "context_block": ctx,
            "history": [{"stage": "build_context", "scope": scope, "context_chars": len(ctx)}],
        }

    def _answer(self, state: GraphRAGState) -> dict[str, Any]:
        ctx = state.get("context_block", "")
        ans = str(self.llm.invoke(
            f"Use the graph context below to answer.\n\n# Context\n{ctx}\n\n# Question\n{state['task']}\n\nAnswer:"
        ).content).strip()
        return {
            "final_answer": ans,
            "history": [{"stage": "answer"}],
        }

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(GraphRAGState)
        g.add_node("classify", self._classify)
        g.add_node("build_context", self._build_context)
        g.add_node("answer", self._answer)
        g.add_edge(START, "classify")
        g.add_edge("classify", "build_context")
        g.add_edge("build_context", "answer")
        g.add_edge("answer", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke({"task": task}, config={"recursion_limit": 25})
        return ArchitectureResult(
            output=final_state.get("final_answer", ""),
            state={
                "scope": final_state.get("scope"),
                "n_target_entities": len(final_state.get("target_entities", [])),
            },
            trace=final_state.get("history", []),
            metadata={
                "scope": final_state.get("scope"),
                "target_entities": final_state.get("target_entities", []),
                "context_chars": len(final_state.get("context_block", "")),
                "n_communities": len(self.communities),
                "community_sizes": [len(c) for c in self.communities],
            },
        )
