"""Tests for the Architecture base class and ArchitectureResult dataclass."""

from __future__ import annotations

import pytest

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


def test_architecture_result_defaults() -> None:
    r = ArchitectureResult(output="hi")
    assert r.output == "hi"
    assert r.state == {}
    assert r.trace == []
    assert r.metadata == {}


def test_architecture_result_full() -> None:
    r = ArchitectureResult(output="x", state={"a": 1}, trace=[{"e": 1}], metadata={"k": 2})
    assert r.state == {"a": 1}
    assert r.trace == [{"e": 1}]
    assert r.metadata == {"k": 2}


def test_architecture_is_abstract() -> None:
    """Architecture cannot be instantiated directly — it's ABC."""
    with pytest.raises(TypeError):
        Architecture()  # type: ignore[abstract]


def test_architecture_subclass_must_implement_build_and_run(mock_llm) -> None:
    """Subclass missing build() OR run() is still abstract."""

    class IncompleteA(Architecture):
        name = "incomplete_a"

        def build(self) -> object:
            return None

        # missing run()

    with pytest.raises(TypeError):
        IncompleteA(llm=mock_llm)  # type: ignore[abstract]


def test_architecture_complete_subclass_works(mock_llm) -> None:
    """A subclass implementing both abstract methods is instantiable."""

    class Toy(Architecture):
        name = "toy"

        def build(self) -> str:
            return "graph_stub"

        def run(self, task: str, **_: object) -> ArchitectureResult:
            return ArchitectureResult(output=f"answer-to:{task}", metadata={"task_len": len(task)})

    arch = Toy(llm=mock_llm)
    r = arch.run("ping")
    assert r.output == "answer-to:ping"
    assert r.metadata == {"task_len": 4}
    assert arch.build() == "graph_stub"


def test_architecture_explain_falls_back_to_description(mock_llm) -> None:
    class WithDesc(Architecture):
        name = "with_desc"
        description = "I explain things."

        def build(self) -> None:
            return None

        def run(self, task: str, **_: object) -> ArchitectureResult:
            return ArchitectureResult(output="")

    arch = WithDesc(llm=mock_llm)
    assert "explain" in arch.explain()
