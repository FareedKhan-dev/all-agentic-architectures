"""Render a compiled LangGraph as a Mermaid diagram string.

Notebooks call this once to show the architecture diagram inline; the same
function is used by the docs site so notebook + docs diagrams stay byte-identical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def graph_to_mermaid(graph: CompiledStateGraph) -> str:
    """Return the Mermaid syntax for a compiled LangGraph state graph.

    LangGraph exposes `get_graph().draw_mermaid()` which we delegate to —
    centralizing it here means notebooks don't need to remember the call.
    """
    return graph.get_graph().draw_mermaid()


def render_mermaid_in_notebook(graph: CompiledStateGraph) -> None:
    """Display the mermaid diagram inline in a Jupyter notebook (PNG via mermaid.ink)."""
    try:
        from IPython.display import Image, display
    except ImportError as e:
        raise ImportError("render_mermaid_in_notebook requires ipython") from e

    png = graph.get_graph().draw_mermaid_png()
    display(Image(png))
