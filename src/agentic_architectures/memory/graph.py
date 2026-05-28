"""Graph-memory abstraction with two backends:

  - **networkx** (default): pure-Python, in-memory. Zero install, runs anywhere.
    Use this for notebooks, tests, and quick demos.
  - **neo4j**: connects to a Neo4j instance (local Desktop, AuraDB Free, or self-hosted).
    Use this for persistence + Cypher queries + production workloads.

Switch via `GRAPH_BACKEND` in .env. Both backends implement the same `GraphMemory`
API so the architectures don't care which one is running underneath.

Used by:
  - 08_episodic_semantic_memory (semantic half of dual memory system)
  - 12_graph_memory (world-model)
  - 27_graphrag (knowledge graph + community summaries)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from agentic_architectures.config import GraphBackend, settings

if TYPE_CHECKING:
    import networkx as nx
    from langchain_neo4j import Neo4jGraph


# ============================================================================
#  Abstract base — every backend implements this
# ============================================================================
class BaseGraphMemory(ABC):
    @abstractmethod
    def add_triple(self, subject: str, predicate: str, obj: str) -> None: ...
    @abstractmethod
    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...
    @abstractmethod
    def neighbors(self, entity: str) -> list[str]: ...
    @abstractmethod
    def reset(self) -> None: ...
    @abstractmethod
    def to_cytoscape(self) -> dict[str, Any]:
        """Export the graph for visualisation (used by docs/notebook diagrams)."""


# ============================================================================
#  NetworkX backend — pure Python, zero install
# ============================================================================
class NetworkXGraphMemory(BaseGraphMemory):
    """In-memory graph backend. Subset of Cypher supported via a tiny translator."""

    def __init__(self) -> None:
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "NetworkX backend requires `pip install networkx`. "
                "It's a tiny pure-Python dep; install it or set GRAPH_BACKEND=neo4j."
            ) from e
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()

    def add_triple(self, subject: str, predicate: str, obj: str) -> None:
        self._g.add_node(subject, label="Entity")
        self._g.add_node(obj, label="Entity")
        self._g.add_edge(subject, obj, predicate=predicate)

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Tiny Cypher subset: enough to back the notebooks.

        Supports two patterns the architectures actually use:
          - `MATCH (n:Entity {name:$name})-[r]-(o) RETURN o.name AS name`
          - `MATCH (s)-[r]->(o) RETURN s.name AS subject, r.predicate AS predicate, o.name AS object`
        Anything else: raises so the user knows to switch backends.
        """
        params = params or {}
        c = " ".join(cypher.split()).lower()

        if "match (n:entity {name:" in c and "return distinct o.name" in c:
            name = params.get("name", "")
            results = []
            for src, dst, _ in self._g.edges(data=True):
                if src == name:
                    results.append({"name": dst})
                elif dst == name:
                    results.append({"name": src})
            # de-dup
            seen: set[str] = set()
            return [r for r in results if not (r["name"] in seen or seen.add(r["name"]))]

        if "match path =" in c or "relates*" in c:
            # Multi-hop traversal used by SemanticMemory.facts_about
            name = params.get("name", "")
            depth = 1
            for tok in cypher.split():
                if tok.startswith("1..") or "RELATES*1.." in tok:
                    try:
                        depth = int(tok.split("..")[-1].rstrip("]"))
                    except ValueError:
                        depth = 1
            visited: set[str] = {name}
            frontier: set[str] = {name}
            triples: list[dict[str, str]] = []
            for _ in range(depth):
                next_frontier: set[str] = set()
                for n in frontier:
                    for u, v, data in list(self._g.in_edges(n, data=True)) + list(self._g.out_edges(n, data=True)):
                        if u == n:
                            triples.append({"subject": u, "predicate": data.get("predicate", ""), "object": v})
                            if v not in visited:
                                next_frontier.add(v)
                        else:
                            triples.append({"subject": u, "predicate": data.get("predicate", ""), "object": v})
                            if u not in visited:
                                next_frontier.add(u)
                visited |= next_frontier
                frontier = next_frontier
            # de-dup
            seen_keys: set[tuple[str, str, str]] = set()
            out: list[dict[str, Any]] = []
            for t in triples:
                key = (t["subject"], t["predicate"], t["object"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    out.append(t)
            return out[:100]

        if "match (s)-[r]->(o)" in c and "return" in c:
            return [
                {"subject": s, "predicate": data.get("predicate", ""), "object": o}
                for s, o, data in self._g.edges(data=True)
            ]

        raise NotImplementedError(
            f"NetworkX backend doesn't support this Cypher pattern yet:\n  {cypher}\n"
            f"Switch to GRAPH_BACKEND=neo4j for full Cypher support."
        )

    def neighbors(self, entity: str) -> list[str]:
        if entity not in self._g:
            return []
        return sorted({n for n in self._g.predecessors(entity)} | {n for n in self._g.successors(entity)})

    def reset(self) -> None:
        self._g.clear()

    def to_cytoscape(self) -> dict[str, Any]:
        nodes = [{"data": {"id": n}} for n in self._g.nodes]
        edges = [
            {"data": {"source": s, "target": t, "label": d.get("predicate", "")}}
            for s, t, d in self._g.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}


# ============================================================================
#  Neo4j backend — production-grade, supports full Cypher
# ============================================================================
class Neo4jGraphMemory(BaseGraphMemory):
    """Thin wrapper around langchain-neo4j's Neo4jGraph."""

    def __init__(self, graph: Neo4jGraph | None = None) -> None:
        if graph is None:
            graph = _connect_neo4j()
        self._graph = graph

    @property
    def graph(self) -> Neo4jGraph:
        return self._graph

    def add_triple(self, subject: str, predicate: str, obj: str) -> None:
        self._graph.query(
            "MERGE (s:Entity {name: $s}) MERGE (o:Entity {name: $o}) MERGE (s)-[r:RELATES {predicate: $p}]->(o)",
            params={"s": subject, "p": predicate, "o": obj},
        )

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self._graph.query(cypher, params=params or {})

    def neighbors(self, entity: str) -> list[str]:
        rows = self._graph.query(
            "MATCH (e:Entity {name: $name})-[r:RELATES]-(o:Entity) RETURN DISTINCT o.name AS name",
            params={"name": entity},
        )
        return [r["name"] for r in rows]

    def reset(self) -> None:
        self._graph.query("MATCH (n) DETACH DELETE n")

    def to_cytoscape(self) -> dict[str, Any]:
        nodes_q = self._graph.query("MATCH (n) RETURN DISTINCT n.name AS name")
        edges_q = self._graph.query(
            "MATCH (s)-[r]->(o) RETURN s.name AS s, type(r) AS type, "
            "coalesce(r.predicate, type(r)) AS predicate, o.name AS o"
        )
        nodes = [{"data": {"id": r["name"]}} for r in nodes_q if r.get("name")]
        edges = [{"data": {"source": r["s"], "target": r["o"], "label": r["predicate"]}} for r in edges_q]
        return {"nodes": nodes, "edges": edges}


def _connect_neo4j() -> Neo4jGraph:
    """Build a Neo4jGraph using settings."""
    try:
        from langchain_neo4j import Neo4jGraph
    except ImportError as e:
        raise ImportError("Neo4j backend requires `pip install agentic-architectures[neo4j]`.") from e

    if settings.neo4j_password is None or not settings.neo4j_password.get_secret_value():
        raise RuntimeError(
            "GRAPH_BACKEND=neo4j requires NEO4J_PASSWORD in .env. "
            "Either set it, or set GRAPH_BACKEND=networkx to use the in-memory backend."
        )

    return Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password.get_secret_value(),
        refresh_schema=False,
    )


# ============================================================================
#  Factory + public alias
# ============================================================================
def get_graph_memory(backend: GraphBackend | None = None) -> BaseGraphMemory:
    """Return a graph-memory instance of the configured backend."""
    backend = backend or settings.graph_backend
    if backend == "networkx":
        return NetworkXGraphMemory()
    if backend == "neo4j":
        return Neo4jGraphMemory()
    raise ValueError(f"Unknown graph backend: {backend!r}")


#: Public alias so notebooks can write ``GraphMemory()`` regardless of backend.
def GraphMemory(backend: GraphBackend | None = None) -> BaseGraphMemory:  # noqa: N802
    """Factory disguised as a class for ergonomic ``GraphMemory()`` calls."""
    return get_graph_memory(backend)
