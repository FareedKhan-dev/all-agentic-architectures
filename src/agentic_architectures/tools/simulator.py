"""Generic simulator tool used by 10_mental_loop and 14_dry_run.

The architecture passes a function `simulate(action) -> outcome` at construction
time; the tool wraps it so an agent can call it as a normal LangChain tool.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class _SimulateInput(BaseModel):
    action: str = Field(description="The action to simulate before executing for real.")


class SimulatorTool:
    """Wrap a domain-specific simulator function in a LangChain tool."""

    def __init__(
        self,
        simulate: Callable[[str], str],
        name: str = "simulate_action",
        description: str = (
            "Simulate an action and return the predicted outcome WITHOUT executing it "
            "in the real world. Use this to weigh options before committing."
        ),
    ) -> None:
        self._simulate = simulate
        self.name = name
        self.description = description

    def as_tool(self) -> StructuredTool:
        def _run(action: str) -> str:
            return self._simulate(action)

        return StructuredTool.from_function(
            func=_run,
            name=self.name,
            description=self.description,
            args_schema=_SimulateInput,
        )

    def __call__(self, action: str) -> str:
        return self._simulate(action)


def make_deterministic_simulator(transitions: dict[str, str]) -> SimulatorTool:
    """Build a simulator from a static lookup table — handy for unit tests."""

    def _simulate(action: str) -> str:
        return transitions.get(action, f"UNKNOWN_ACTION: {action}")

    return SimulatorTool(simulate=_simulate)


def chain_tools(*tools: Any) -> list[Any]:
    """Convenience helper used in some notebooks to bind multiple tools at once."""
    return list(tools)
