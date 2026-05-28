"""Graph Memory — knowledge-graph world model with traversal-based Q&A.

Builds a knowledge graph of (subject, predicate, object) triples extracted from
text, then answers questions by *traversing* the graph rather than re-reading
the source. The graph **is** the agent's world model.

Compared to Episodic + Semantic Memory (notebook 08):
  - Same semantic-graph machinery (`SemanticMemory` from the library).
  - Different *purpose*: nb 08 stores facts about ONE user across conversation
    turns; nb 12 ingests a corpus and uses the resulting graph for Q&A.

Compared to GraphRAG (notebook 27):
  - GraphRAG adds community-detection summaries over the graph for global
    questions ("what are the main themes of the corpus?").
  - Graph Memory (this one) supports only entity-anchored questions
    ("what does the graph say about X?").

Default backend: **NetworkX** (in-process, zero setup). Set GRAPH_BACKEND=neo4j
in .env to swap to AuraDB Free or self-hosted Neo4j.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult
from agentic_architectures.memory import SemanticMemory


# ---------------------------------------------------------------------------
# Extraction schema
# ---------------------------------------------------------------------------
class _IngestionTriple(BaseModel):
    subject: str = Field(description="A specific named entity (capitalised).")
    predicate: str = Field(
        description="A short relation verb in snake_case (e.g. founded_by, headquartered_in, ceo_is)."
    )
    object: str = Field(description="A specific named entity OR a short literal value.")


class _IngestionResult(BaseModel):
    """Triples extracted from one ingest call."""

    triples: list[_IngestionTriple] = Field(
        default_factory=list,
        description=(
            "All atomic (subject, predicate, object) triples that can be extracted "
            "from the text. Be exhaustive — capture every entity and relation. "
            "Use specific named entities; skip generic statements."
        ),
    )


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class GraphMemoryAgent(Architecture):
    """Knowledge-graph world model + traversal-based Q&A."""

    name = "graph_memory"
    description = (
        "Ingest text into a knowledge graph (entity-relation triples), then answer "
        "questions by graph traversal. The graph IS the agent's world model — "
        "answers come from the graph, not by re-reading the source text."
    )
    reference = "https://en.wikipedia.org/wiki/Knowledge_graph"

    def __init__(
        self,
        semantic: SemanticMemory | None = None,
        traversal_depth: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.semantic = semantic if semantic is not None else SemanticMemory()
        self.traversal_depth = traversal_depth
        self._extractor = self.llm.with_structured_output(_IngestionResult)

    # ------------------------------------------------------------------ #
    #  Ingestion (call between runs to populate the graph)                #
    # ------------------------------------------------------------------ #

    def ingest(self, text: str) -> list[_IngestionTriple]:
        """Extract triples from text and add them to the graph."""
        prompt = (
            "Extract a comprehensive set of (subject, predicate, object) triples "
            "from the text below. Each triple must use SPECIFIC named entities and "
            "a SHORT snake_case relation verb. Capture every concrete fact, "
            "skip generic statements.\n\n"
            f"--- TEXT ---\n{text}"
        )
        result = self._extractor.invoke(prompt)
        for t in result.triples:
            self.semantic.add_fact(t.subject, t.predicate, t.object)
        return result.triples

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _entities_in_graph(self) -> list[str]:
        from agentic_architectures.memory.graph import NetworkXGraphMemory

        backend = self.semantic.backend
        if isinstance(backend, NetworkXGraphMemory):
            return list(backend._g.nodes)
        # Neo4j fallback
        try:
            rows = backend.query("MATCH (n:Entity) RETURN n.name AS name LIMIT 200")
            return [r["name"] for r in rows]
        except Exception:
            return []

    def _entities_in_query(self, query: str) -> list[str]:
        all_entities = self._entities_in_graph()
        q_lower = query.lower()
        return [e for e in all_entities if e.lower() in q_lower]

    def _facts_block(self, entities: list[str]) -> tuple[str, list[dict[str, str]]]:
        """Pull facts about each named entity (depth = self.traversal_depth)."""
        seen: set[tuple[str, str, str]] = set()
        facts: list[dict[str, str]] = []
        for e in entities:
            for f in self.semantic.facts_about(e, depth=self.traversal_depth):
                key = (f.get("subject", ""), f.get("predicate", ""), f.get("object", ""))
                if key not in seen:
                    seen.add(key)
                    facts.append(f)
        if not facts:
            return "(no relevant facts in graph)", []
        block = "\n".join(f"  - ({f.get('subject')}, {f.get('predicate')}, {f.get('object')})" for f in facts)
        return block, facts

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        def _passthrough(state: dict) -> dict:
            return state

        g: StateGraph = StateGraph(dict)
        g.add_node("query", _passthrough)
        g.add_edge(START, "query")
        g.add_edge("query", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        """Answer a question by traversing the current graph."""
        query = task
        entities = self._entities_in_query(query)
        # Fall back to all entities for short / vague queries
        if not entities and len(query.split()) <= 12:
            entities = self._entities_in_graph()[:20]

        facts_block, facts = self._facts_block(entities)
        prompt = (
            "You are answering a question using ONLY the facts in the knowledge "
            "graph. Do NOT use parametric knowledge — if the graph doesn't have "
            "the answer, say so explicitly.\n\n"
            f"## Question\n{query}\n\n"
            f"## Relevant graph facts (from {len(entities)} matched entit(y/ies))\n"
            f"{facts_block}\n\n"
            "Answer concisely. Cite the specific (s, p, o) triples you used."
        )
        answer = str(self.llm.invoke(prompt).content)

        return ArchitectureResult(
            output=answer,
            state={
                "matched_entities": entities,
                "facts_used": len(facts),
                "total_entities_in_graph": len(self._entities_in_graph()),
            },
            trace=[
                {"type": "matched_entities", "items": entities},
                {"type": "facts_retrieved", "items": facts[:30]},
                {"type": "answer", "content": answer},
            ],
            metadata={
                "matched_entities": len(entities),
                "facts_retrieved": len(facts),
                "total_entities_in_graph": len(self._entities_in_graph()),
                "graph_backend": type(self.semantic.backend).__name__,
            },
        )
