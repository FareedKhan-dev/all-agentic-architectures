"""Abstract base class every architecture implements.

Defining a common contract gives us three free things:
  1. Uniform notebooks — every notebook calls `arch.build()`, `arch.run()`, `arch.diagram()`.
  2. Composability — meta-controllers and ensembles can treat any architecture as a black-box callable.
  3. Auto-generated docs + mermaid — the docs site renders `arch.diagram()` and `arch.explain()`
     so every architecture's docs page is consistent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


@dataclass
class ArchitectureResult:
    """Standardized return type so notebooks can render results uniformly."""

    output: str
    state: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Architecture(ABC):
    """Base class for every agentic pattern in this library.

    Subclasses MUST implement `build()` and `run()`. They MAY override `diagram()`
    and `explain()` — both have sensible defaults.
    """

    #: short identifier (snake_case), used as the registry key + docs slug
    name: str = "unnamed"
    #: one-line description shown in the README/docs tables
    description: str = ""
    #: link to the originating paper or blog (for citation)
    reference: str = ""

    def __init__(self, llm: BaseChatModel | None = None, **kwargs: Any) -> None:
        from agentic_architectures.llm.factory import get_llm

        self.llm = llm if llm is not None else get_llm()
        self.config = kwargs

    # ------------------------------------------------------------------ #
    #  Required hooks                                                     #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def build(self) -> CompiledStateGraph:
        """Compile and return the LangGraph for this architecture."""

    @abstractmethod
    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        """Execute the architecture on a task and return a standardized result."""

    # ------------------------------------------------------------------ #
    #  Optional helpers                                                   #
    # ------------------------------------------------------------------ #

    def diagram(self) -> str:
        """Return a Mermaid diagram string for the compiled graph."""
        from agentic_architectures.ui.diagram import graph_to_mermaid

        return graph_to_mermaid(self.build())

    def explain(self) -> str:
        """Return the theory section (markdown) for this architecture."""
        return self.description or self.__doc__ or "(no description provided)"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(llm={self.llm.__class__.__name__})"
