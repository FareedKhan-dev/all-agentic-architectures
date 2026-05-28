"""Episodic memory — store + retrieve past 'episodes' (conversations, traces).

Used by:
  - 08_episodic_semantic_memory (long-term personal assistant)
  - 15_rlhf_self_improvement (archive of high-quality outputs)
  - 18_reflexion (verbal-reflection log across episodes)
  - 31_memgpt (long-term memory tier)
  - 35_agent_workflow_memory (workflow extraction from past traces)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from agentic_architectures.memory.vector import VectorMemory

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


@dataclass
class Episode:
    """A single recorded episode."""

    content: str
    role: str = "user"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    """Vector-backed episodic store with timestamped retrieval."""

    def __init__(
        self,
        embeddings: Embeddings | None = None,
        collection_name: str = "episodic_memory",
    ) -> None:
        self._vector = VectorMemory(
            embeddings=embeddings, collection_name=collection_name
        )
        self._episodes: list[Episode] = []

    def record(self, episode: Episode | str, **metadata: Any) -> None:
        """Add an episode to memory (string is converted to a default Episode)."""
        from langchain_core.documents import Document

        ep = episode if isinstance(episode, Episode) else Episode(content=episode, metadata=metadata)
        self._episodes.append(ep)
        self._vector.add(
            [Document(page_content=ep.content, metadata={**asdict(ep), **ep.metadata})]
        )

    def recall(self, query: str, k: int = 5) -> list[Episode]:
        """Retrieve the k most semantically similar past episodes."""
        docs = self._vector.search(query, k=k)
        return [
            Episode(
                content=d.page_content,
                role=d.metadata.get("role", "user"),
                timestamp=d.metadata.get("timestamp", ""),
                metadata={k: v for k, v in d.metadata.items() if k not in {"role", "timestamp", "content"}},
            )
            for d in docs
        ]

    @property
    def episodes(self) -> list[Episode]:
        return list(self._episodes)
