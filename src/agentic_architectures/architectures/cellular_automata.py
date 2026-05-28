"""Cellular Automata — grid of LLM-driven cells with local update rules; emergent global behaviour.

Many simple agents on a grid. Each cell at each time-step:
  1. Reads its OWN state + the state of its 4-neighbours (von Neumann neighbourhood).
  2. Calls an LLM with a tiny prompt asking what to do next.
  3. Updates its state synchronously with the rest of the grid.

Why this is an *agentic* pattern, not just "Conway's Life with LLMs":
  - Each cell's "update rule" is a *prompt*, not hard-coded transition logic.
  - Cells can encode richer state than `{alive, dead}` — strings, sentiments,
    typed Pydantic objects.
  - Global behaviour emerges from local LLM decisions — useful for spatial
    reasoning, logistics, opinion dynamics, agent-based simulations.

Cost: GRID_SIZE^2 × LLM_CALLS_PER_CELL × NUM_STEPS. Cap aggressively — a 5×5
grid × 3 steps = 75 LLM calls per simulation. We use a tiny **4×4 grid × 3
steps** in the demo for educational tractability.

Origin: Cellular automata (von Neumann, Conway 1970); LLM-augmented variants
explored in agent-based modeling literature.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


# ---------------------------------------------------------------------------
# Update schema
# ---------------------------------------------------------------------------
class _CellUpdate(BaseModel):
    """The next state a cell should transition to, given its neighbours."""

    next_state: str = Field(
        description=(
            "The cell's state for the NEXT time step. MUST be one of the allowed "
            "state labels listed in the prompt. Keep state labels SHORT (1-2 words)."
        ),
    )
    reason: str = Field(description="One short sentence explaining the transition.")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class CellularAutomataState(TypedDict, total=False):
    grid: list[list[str]]              # current grid state (HxW strings)
    history: list[list[list[str]]]     # past grids (for visualisation)
    step: int
    max_steps: int


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class CellularAutomata(Architecture):
    """Grid of cells, each updated by an LLM per step."""

    name = "cellular_automata"
    description = (
        "Each cell on an HxW grid is an LLM agent. At each step, every cell reads "
        "its own state + 4 neighbours' states and decides its next state via a "
        "structured-output call. Emergent global behaviour."
    )
    reference = "Cellular automata (von Neumann, Conway); LLM-as-rule extension."

    def __init__(
        self,
        rule_prompt: str,
        allowed_states: list[str],
        height: int = 4,
        width: int = 4,
        max_steps: int = 3,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            rule_prompt: A short description of the LOCAL update rule the LLM should follow.
            allowed_states: List of valid state labels each cell can hold (e.g. ['fire', 'tree', 'empty']).
            height, width: Grid dimensions. Default 4x4 = 16 cells; cap aggressively.
            max_steps: How many time steps to simulate.
        """
        super().__init__(**kwargs)
        self.rule_prompt = rule_prompt
        self.allowed_states = allowed_states
        self.height = height
        self.width = width
        self.max_steps = max_steps
        self._updater = self.llm.with_structured_output(_CellUpdate)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _neighbours(grid: list[list[str]], r: int, c: int) -> dict[str, str]:
        """Von Neumann neighbourhood: N, E, S, W. Edges = 'edge'."""
        h, w = len(grid), len(grid[0])
        def _g(rr: int, cc: int) -> str:
            return grid[rr][cc] if 0 <= rr < h and 0 <= cc < w else "edge"
        return {
            "N": _g(r - 1, c),
            "E": _g(r, c + 1),
            "S": _g(r + 1, c),
            "W": _g(r, c - 1),
        }

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _step(self, state: CellularAutomataState) -> dict[str, Any]:
        grid = state["grid"]
        new_grid: list[list[str]] = [row.copy() for row in grid]
        states_list = ", ".join(self.allowed_states)

        for r in range(self.height):
            for c in range(self.width):
                neighbours = self._neighbours(grid, r, c)
                prompt = (
                    f"## Local update rule\n{self.rule_prompt}\n\n"
                    f"## Allowed next states\n{states_list}\n\n"
                    f"## This cell's current state\n{grid[r][c]}\n\n"
                    f"## Neighbours\n"
                    f"  North: {neighbours['N']}\n"
                    f"  East:  {neighbours['E']}\n"
                    f"  South: {neighbours['S']}\n"
                    f"  West:  {neighbours['W']}\n\n"
                    "Determine this cell's next state given the rule and its neighbours. "
                    "Return one of the allowed states only."
                )
                update = self._updater.invoke(prompt)
                # Deterministic clamp: if LLM returns something outside allowed_states, keep current.
                next_state = (
                    update.next_state
                    if update.next_state in self.allowed_states
                    else grid[r][c]
                )
                new_grid[r][c] = next_state

        history = state.get("history", []) + [grid]
        return {
            "grid": new_grid,
            "history": history,
            "step": state.get("step", 0) + 1,
        }

    # ------------------------------------------------------------------ #
    #  Router                                                             #
    # ------------------------------------------------------------------ #

    def _should_continue(self, state: CellularAutomataState) -> str:
        return "step" if state.get("step", 0) < self.max_steps else "end"

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(CellularAutomataState)
        g.add_node("step", self._step)
        g.add_edge(START, "step")
        g.add_conditional_edges("step", self._should_continue, {"step": "step", "end": END})
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        """Run the simulation. `task` is the INITIAL GRID as a multi-line string,
        with cells separated by `|` and rows by `\\n`.
        Example: 'tree|tree|fire|empty\\ntree|tree|tree|empty'
        """
        # Parse initial grid
        rows = [row.split("|") for row in task.strip().split("\n")]
        # Validate dimensions
        if len(rows) != self.height:
            raise ValueError(f"Initial grid has {len(rows)} rows, expected {self.height}")
        if any(len(row) != self.width for row in rows):
            raise ValueError(f"Initial grid rows must have {self.width} columns")
        initial_grid: list[list[str]] = [[c.strip() for c in row] for row in rows]

        graph = self.build()
        config = {"recursion_limit": self.max_steps + 5}
        final_state = graph.invoke(
            {"grid": initial_grid, "step": 0, "history": [], "max_steps": self.max_steps},
            config=config,
        )

        history = final_state.get("history", []) + [final_state["grid"]]

        # Count state-label changes across history
        from collections import Counter
        per_step_counts = []
        for step_grid in history:
            counter: Counter[str] = Counter()
            for row in step_grid:
                for cell in row:
                    counter[cell] += 1
            per_step_counts.append(dict(counter))

        return ArchitectureResult(
            output="\n\n".join(
                f"Step {i}:\n" + "\n".join("|".join(row) for row in g)
                for i, g in enumerate(history)
            ),
            state={
                "final_grid": final_state["grid"],
                "history": history,
                "per_step_counts": per_step_counts,
            },
            trace=[{"type": "grid", "step": i, "grid": g} for i, g in enumerate(history)],
            metadata={
                "steps_run": len(history) - 1,
                "grid_size": f"{self.height}x{self.width}",
                "per_step_counts": per_step_counts,
                "allowed_states": self.allowed_states,
            },
        )
