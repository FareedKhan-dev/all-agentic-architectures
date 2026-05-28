"""Registry sweep — every architecture in the public API must import + build cleanly.

This is a single test that touches all 35 architectures via the package's
public `__all__`. If any architecture's import is broken, this fails fast.
"""

from __future__ import annotations

import inspect

import pytest

import agentic_architectures.architectures as A
from agentic_architectures.architectures.base import Architecture

# Architectures that need extra constructor args beyond `llm=` — provide minimal defaults.
EXTRA_KWARGS: dict[str, dict[str, object]] = {
    "AgenticRAG": {"documents": ["fact one", "fact two"]},
    "CorrectiveRAG": {"documents": ["fact one", "fact two"]},
    "SelfRAG": {"documents": ["fact one", "fact two"]},
    "AdaptiveRAG": {"documents": ["fact one", "fact two"]},
    "GraphRAG": {"documents": []},  # skip KG build by passing empty docs
    "SWEAgent": {"working_dir": "."},  # current dir is fine — we don't .run()
    "ComputerUse": {"initial_screen": {"url": "about:blank", "elements": []}},
    "BrowserAgent": {"headless": True},
    "CellularAutomata": {
        "rule_prompt": "Update the cell based on its neighbors.",
        "allowed_states": ["empty", "filled"],
    },
}


def _all_architecture_classes() -> list[type[Architecture]]:
    classes: list[type[Architecture]] = []
    for name in A.__all__:
        obj = getattr(A, name)
        if inspect.isclass(obj) and issubclass(obj, Architecture) and obj is not Architecture:
            classes.append(obj)
    return classes


CLASSES = _all_architecture_classes()


def test_at_least_thirty_five_architectures() -> None:
    """Phase 2 (17) + Phase 3 (18) = 35 architectures expected."""
    assert len(CLASSES) >= 35, f"Expected ≥35 architectures, got {len(CLASSES)}: {[c.__name__ for c in CLASSES]}"


@pytest.mark.parametrize("cls", CLASSES, ids=[c.__name__ for c in CLASSES])
def test_class_metadata_present(cls: type[Architecture]) -> None:
    """Every architecture must declare name/description/reference (non-default)."""
    assert cls.name != "unnamed", f"{cls.__name__} did not override `name`"
    assert cls.description, f"{cls.__name__} did not set `description`"
    assert cls.reference, f"{cls.__name__} did not set `reference`"


@pytest.mark.parametrize("cls", CLASSES, ids=[c.__name__ for c in CLASSES])
def test_can_instantiate_with_mock_llm(cls: type[Architecture], mock_llm) -> None:
    """Every architecture must instantiate with a MockLLM + minimal kwargs."""
    kwargs = EXTRA_KWARGS.get(cls.__name__, {})
    arch = cls(llm=mock_llm, **kwargs)
    assert arch.llm is mock_llm


@pytest.mark.parametrize("cls", CLASSES, ids=[c.__name__ for c in CLASSES])
def test_can_build_graph(cls: type[Architecture], mock_llm) -> None:
    """Every architecture's build() must return a compiled LangGraph
    (or at least an object with `get_graph()` method)."""
    kwargs = EXTRA_KWARGS.get(cls.__name__, {})
    arch = cls(llm=mock_llm, **kwargs)
    graph = arch.build()
    assert hasattr(graph, "get_graph") or hasattr(graph, "invoke"), (
        f"{cls.__name__}.build() returned a non-graph object: {type(graph)}"
    )
    if hasattr(arch, "close"):
        arch.close()
