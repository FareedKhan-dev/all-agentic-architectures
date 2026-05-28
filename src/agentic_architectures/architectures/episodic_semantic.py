"""Episodic + Semantic Memory — dual-memory agent.

An agent that maintains **two** distinct memory systems that persist across
interactions on the same architecture instance:

  - **Episodic memory** (vector store, FAISS by default) — remembers past
    conversations / events by similarity. Used to recall: *"have I had a
    similar conversation before? what did I say?"*
  - **Semantic memory** (graph store, NetworkX by default) — remembers
    structured facts as (subject, predicate, object) triples. Used to recall:
    *"what do I know about entity X?"*

The dual-memory design is loosely inspired by Tulving's psychology distinction
(1972): episodic memory is *autobiographical* ("I had pizza on Tuesday");
semantic memory is *abstract* ("pizza is Italian"). LLM agents benefit from
both — episodes ground the assistant in conversation continuity, semantic
facts answer "what do you know about me?" queries reliably.

Single-call architecture flow per `run(query)`:
  1. **Retrieve** — recall relevant episodes (vector similarity) + facts
     (entity match) given the query.
  2. **Answer** — compose response using the retrieved context.
  3. **Extract** — pull new (s, p, o) triples from this interaction; save
     to semantic.
  4. **Record** — save the full Q&A as a new episode in episodic.

Used by:
  - 08 (this notebook) — long-term personal assistant.
  - 18 Reflexion (extends to episode-based self-reflection).
  - 31 MemGPT (extends to OS-style paging between memory tiers).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult
from agentic_architectures.memory import EpisodicMemory, SemanticMemory
from agentic_architectures.memory.episodic import Episode


# ---------------------------------------------------------------------------
# Fact-extraction schema
# ---------------------------------------------------------------------------
class _Triple(BaseModel):
    subject: str = Field(description="A specific named entity (person, place, thing).")
    predicate: str = Field(description="A short relation verb (e.g. 'works_at', 'lives_in', 'likes').")
    object: str = Field(description="A specific named entity OR a short literal value.")


class _ExtractedFacts(BaseModel):
    """Triples extracted from a single interaction. Skip generic, vague statements."""

    facts: list[_Triple] = Field(
        default_factory=list,
        description=(
            "Atomic (subject, predicate, object) triples explicitly stated or strongly "
            "implied in the conversation. Use specific named entities only. "
            "Skip generic claims (e.g. 'people like food'). Return an empty list "
            "if no concrete facts can be extracted."
        ),
    )


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class EpisodicSemanticAgent(Architecture):
    """Dual-memory agent: FAISS episodes + NetworkX/Neo4j facts; persistent across run() calls."""

    name = "episodic_semantic"
    description = (
        "Agent with two complementary memory systems: a vector store for past "
        "conversations (episodic) and a graph store for structured facts (semantic). "
        "Memory persists across run() calls on the same instance."
    )
    reference = "Tulving (1972) Episodic vs Semantic memory; MemGPT (2023)"

    def __init__(
        self,
        episodic: EpisodicMemory | None = None,
        semantic: SemanticMemory | None = None,
        recall_k_episodes: int = 3,
        recall_facts_depth: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.episodic = episodic if episodic is not None else EpisodicMemory()
        self.semantic = semantic if semantic is not None else SemanticMemory()
        self.recall_k_episodes = recall_k_episodes
        self.recall_facts_depth = recall_facts_depth
        self._fact_extractor = self.llm.with_structured_output(_ExtractedFacts)

    # ------------------------------------------------------------------ #
    #  Sub-steps — exposed as separate methods for transparency           #
    # ------------------------------------------------------------------ #

    def _retrieve(self, query: str) -> dict[str, Any]:
        # Episodic: vector recall over past episodes
        episodes = (
            self.episodic.recall(query, k=self.recall_k_episodes)
            if self.episodic.episodes
            else []
        )
        # Semantic: brute-force entity-match on stored nodes (small graphs only)
        all_entities = []
        try:
            backend = self.semantic.backend
            from agentic_architectures.memory.graph import NetworkXGraphMemory

            if isinstance(backend, NetworkXGraphMemory):
                all_entities = list(backend._g.nodes)
        except Exception:
            pass

        query_lower = query.lower()
        relevant_entities = [
            e for e in all_entities if e.lower() in query_lower or query_lower in e.lower()
        ]
        # Also include all entities if query is short / generic (e.g. "tell me about me")
        if not relevant_entities and len(query.split()) <= 10:
            relevant_entities = all_entities[:10]

        recalled_facts: list[dict[str, Any]] = []
        seen = set()
        for entity in relevant_entities:
            for fact in self.semantic.facts_about(entity, depth=self.recall_facts_depth):
                key = (fact.get("subject"), fact.get("predicate"), fact.get("object"))
                if key not in seen:
                    seen.add(key)
                    recalled_facts.append(fact)

        return {"episodes": episodes, "facts": recalled_facts}

    def _format_context(self, episodes: list[Episode], facts: list[dict[str, Any]]) -> str:
        lines = []
        if facts:
            lines.append("## Known facts (semantic memory)")
            for f in facts[:20]:
                lines.append(f"  - ({f.get('subject')}, {f.get('predicate')}, {f.get('object')})")
        if episodes:
            lines.append("\n## Past episodes (recent / similar)")
            for ep in episodes:
                snippet = ep.content[:200].replace("\n", " ")
                lines.append(f"  - [{ep.role}] {snippet}{'…' if len(ep.content) > 200 else ''}")
        return "\n".join(lines) if lines else "(no relevant memory)"

    def _answer(self, query: str, context: str) -> str:
        prompt = (
            "You are a personal assistant with persistent memory.\n\n"
            f"## Your memory of this conversation so far\n{context}\n\n"
            f"## Current user message\n{query}\n\n"
            "Reply concisely. If the user asks what you know about them, "
            "answer ONLY from the memory above — do not fabricate."
        )
        return str(self.llm.invoke(prompt).content)

    def _extract_and_save_facts(self, query: str, answer: str) -> list[_Triple]:
        prompt = (
            "Extract atomic (subject, predicate, object) triples from this "
            "interaction. Use specific named entities — skip generic claims. "
            "Return an empty list if no concrete facts.\n\n"
            f"User: {query}\nAssistant: {answer}"
        )
        result = self._fact_extractor.invoke(prompt)
        for t in result.facts:
            self.semantic.add_fact(t.subject, t.predicate, t.object)
        return result.facts

    def _save_episode(self, query: str, answer: str) -> None:
        self.episodic.record(
            Episode(
                content=f"User: {query}\nAssistant: {answer}",
                role="conversation",
                metadata={},
            )
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        # We don't strictly need LangGraph here — the flow is linear per call.
        # Return a minimal one-node graph for compatibility with diagram().
        from langgraph.graph import END, START, StateGraph

        class _S(dict):
            pass

        def _passthrough(state: _S) -> _S:
            return state

        g: StateGraph = StateGraph(dict)
        g.add_node("interact", _passthrough)
        g.add_edge(START, "interact")
        g.add_edge("interact", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        query = task

        retrieved = self._retrieve(query)
        episodes_recalled = retrieved["episodes"]
        facts_recalled = retrieved["facts"]
        context = self._format_context(episodes_recalled, facts_recalled)

        answer = self._answer(query, context)
        new_facts = self._extract_and_save_facts(query, answer)
        self._save_episode(query, answer)

        return ArchitectureResult(
            output=answer,
            state={
                "context_used": context,
                "total_episodes_stored": len(self.episodic.episodes),
                "total_entities_stored": self._count_entities(),
            },
            trace=[
                {"type": "retrieved_episodes", "count": len(episodes_recalled),
                 "items": [{"role": e.role, "content": e.content[:200]} for e in episodes_recalled]},
                {"type": "retrieved_facts", "count": len(facts_recalled),
                 "items": facts_recalled[:10]},
                {"type": "answer", "content": answer},
                {"type": "extracted_facts", "count": len(new_facts),
                 "items": [t.model_dump() for t in new_facts]},
            ],
            metadata={
                "episodes_recalled": len(episodes_recalled),
                "facts_recalled": len(facts_recalled),
                "new_facts_extracted": len(new_facts),
                "total_episodes_stored": len(self.episodic.episodes),
                "total_entities_stored": self._count_entities(),
            },
        )

    def _count_entities(self) -> int:
        from agentic_architectures.memory.graph import NetworkXGraphMemory

        backend = self.semantic.backend
        if isinstance(backend, NetworkXGraphMemory):
            return len(backend._g.nodes)
        # Neo4j fallback
        try:
            rows = backend.query("MATCH (n:Entity) RETURN count(n) AS c")
            return int(rows[0]["c"]) if rows else 0
        except Exception:
            return 0
