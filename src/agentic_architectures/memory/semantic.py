"""Semantic memory — structured (subject, predicate, object) facts.

Backend-agnostic: defaults to NetworkX in-memory, switches to Neo4j when
GRAPH_BACKEND=neo4j is set in .env. Same API either way.

Used by:
  - 08_episodic_semantic_memory (paired with episodic for full memory system)
  - 12_graph_memory (world-model)
  - 27_graphrag (knowledge graph + community summaries)
"""

from __future__ import annotations

from agentic_architectures.config import GraphBackend
from agentic_architectures.memory.graph import BaseGraphMemory, get_graph_memory


class SemanticMemory:
    """Triple-store wrapper on top of whichever graph backend is configured."""

    def __init__(
        self,
        graph: BaseGraphMemory | None = None,
        backend: GraphBackend | None = None,
    ) -> None:
        self._memory: BaseGraphMemory = graph if graph is not None else get_graph_memory(backend)

    def add_fact(self, subject: str, predicate: str, obj: str) -> None:
        """Record a (subject, predicate, object) triple."""
        self._memory.add_triple(subject, predicate, obj)

    def add_facts(self, facts: list[tuple[str, str, str]]) -> None:
        """Bulk-add triples."""
        for s, p, o in facts:
            self._memory.add_triple(s, p, o)

    def facts_about(self, entity: str, depth: int = 1) -> list[dict[str, str]]:
        """Return facts within `depth` hops of the given entity."""
        cypher = (
            "MATCH path = (e:Entity {name: $name})-[r:RELATES*1.." + str(depth) + "]-(other) "
            "UNWIND relationships(path) AS rel "
            "RETURN startNode(rel).name AS subject, rel.predicate AS predicate, "
            "endNode(rel).name AS object LIMIT 100"
        )
        return self._memory.query(cypher, params={"name": entity})

    def neighbors(self, entity: str) -> list[str]:
        return self._memory.neighbors(entity)

    def reset(self) -> None:
        self._memory.reset()

    @property
    def backend(self) -> BaseGraphMemory:
        return self._memory
