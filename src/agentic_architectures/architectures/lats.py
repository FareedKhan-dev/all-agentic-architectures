"""LATS — Language Agent Tree Search (MCTS-flavoured reasoning over a thought tree).

Extends **Tree of Thoughts** (nb 09) with three MCTS-style additions:
  1. A real tree (parent pointers + visit counts) rather than a flat beam.
  2. **UCB1 selection** — balance exploitation (high-value nodes) vs exploration
     (rarely-visited nodes), instead of greedy best-first.
  3. **Value backup** — when a leaf is evaluated, its value propagates up its
     ancestors so siblings of high-value subtrees become more attractive.

Reward at each leaf comes from an LLM-evaluator. **The deterministic-picker
pattern (handoff §7) is applied here**: the LLM commits to independent
booleans (`makes_progress`, `is_complete`, `avoids_loops`) plus a categorical
`confidence`; Python composes the numeric `value` ∈ [0, 10] deterministically.
This is the same fix as Mental Loop (nb 10) and RLHF (nb 15) and is essential
for LATS to have non-flat reward signal across leaves.

Origin: Zhou et al., *Language Agent Tree Search Unifies Reasoning, Acting,
and Planning in Language Models* (2024). https://arxiv.org/abs/2310.04406

Demo task is Game of 24 (same as nb 09 ToT) so the contrast with ToT is
clean: ToT is greedy-beam, LATS is tree-with-backup.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class _ThoughtCandidates(BaseModel):
    """K candidate next-thoughts to expand under a tree node."""

    candidates: list[str] = Field(
        description="K substantively-different next reasoning steps. Each must be a "
                    "single concrete move (e.g., 'apply 8-6=2; remaining [2,4,12]') — "
                    "not multi-step or vague.",
        min_length=2,
    )


class _LeafEvaluation(BaseModel):
    """Deterministic-picker reward — LLM commits to objective features only."""

    makes_progress: bool = Field(
        description="True iff this leaf advances toward the goal vs its parent."
    )
    is_complete: bool = Field(
        description="True iff this leaf represents a COMPLETE solution to the original task."
    )
    avoids_loops: bool = Field(
        description="True iff this leaf does NOT repeat a state already seen in its ancestor chain."
    )
    confidence: str = Field(
        description="One of: 'high', 'medium', 'low'. How confident are you in the above features?"
    )
    rationale: str = Field(description="ONE sentence explaining the assessment.")


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------
@dataclass
class _Node:
    id: int
    thought: str                  # the partial-solution text at this node
    parent_id: int | None = None
    children_ids: list[int] = field(default_factory=list)
    value: float = 0.0            # average reward across visits
    visits: int = 0
    is_terminal: bool = False
    features: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class LATSState(TypedDict, total=False):
    task: str
    iteration: int
    max_iterations: int
    branching: int
    nodes: dict[int, _Node]       # node_id -> _Node
    next_id: int
    root_id: int
    best_leaf_id: int | None
    final_answer: str
    history: Annotated[list[dict[str, Any]], operator.add]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class LATS(Architecture):
    """MCTS-style tree search over LLM-generated thoughts with deterministic-picker reward."""

    name = "lats"
    description = (
        "Tree search over reasoning thoughts: UCB1 select → expand K children → "
        "LLM-evaluate (deterministic-picker reward) → backup. Returns the best "
        "complete path."
    )
    reference = "https://arxiv.org/abs/2310.04406"

    def __init__(
        self,
        max_iterations: int = 6,
        branching: int = 3,
        ucb_c: float = 1.4,
        max_depth: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_iterations = max_iterations
        self.branching = branching
        self.ucb_c = ucb_c
        self.max_depth = max_depth
        self._expander = self.llm.with_structured_output(_ThoughtCandidates)
        self._evaluator = self.llm.with_structured_output(_LeafEvaluation)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _composite_value(features: dict[str, Any]) -> float:
        """Python-composed reward. Same deterministic-picker pattern as RLHF nb 15."""
        v = 0.0
        if features.get("is_complete", False):
            v += 5.0
        if features.get("makes_progress", False):
            v += 2.0
        if features.get("avoids_loops", False):
            v += 1.0
        conf = features.get("confidence", "low")
        v += {"high": 2.0, "medium": 1.0, "low": 0.0}.get(conf, 0.0)
        return min(v, 10.0)

    def _ucb1(self, node: _Node, parent_visits: int) -> float:
        if node.visits == 0:
            return float("inf")     # unvisited nodes are infinitely attractive
        exploit = node.value
        explore = self.ucb_c * math.sqrt(math.log(max(parent_visits, 1)) / node.visits)
        return exploit + explore

    def _depth(self, nodes: dict[int, _Node], node_id: int) -> int:
        d = 0
        cur = node_id
        while nodes[cur].parent_id is not None:
            cur = nodes[cur].parent_id  # type: ignore[assignment]
            d += 1
        return d

    def _path_to_root(self, nodes: dict[int, _Node], node_id: int) -> list[_Node]:
        path = []
        cur: int | None = node_id
        while cur is not None:
            path.append(nodes[cur])
            cur = nodes[cur].parent_id
        return list(reversed(path))

    def _select_leaf(self, nodes: dict[int, _Node], root_id: int) -> int:
        """UCB1 descent from root to a leaf."""
        cur = root_id
        while nodes[cur].children_ids:
            parent_visits = nodes[cur].visits
            children = [nodes[cid] for cid in nodes[cur].children_ids]
            cur = max(children, key=lambda n: self._ucb1(n, parent_visits)).id
        return cur

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _init(self, state: LATSState) -> dict[str, Any]:
        root = _Node(id=0, thought=f"START. Task: {state['task']}", parent_id=None)
        return {
            "iteration": 0,
            "nodes": {0: root},
            "next_id": 1,
            "root_id": 0,
            "max_iterations": self.max_iterations,
            "branching": self.branching,
            "history": [{"stage": "init", "root_thought": root.thought[:120]}],
        }

    def _iterate(self, state: LATSState) -> dict[str, Any]:
        """One MCTS-style iteration: select → expand → evaluate → backup.

        Implemented as a single LangGraph node (instead of one node per phase)
        because the four phases are tightly coupled and operate on the same
        mutable tree dict — splitting them would force LangGraph to copy state
        between every phase, which would be wasteful and obscure the algorithm.
        """
        nodes = state["nodes"]
        root_id = state["root_id"]
        next_id = state["next_id"]
        iteration = state.get("iteration", 0) + 1

        # 1. SELECT leaf via UCB1
        leaf_id = self._select_leaf(nodes, root_id)
        leaf = nodes[leaf_id]

        # 2. EXPAND — generate branching candidates if not terminal & under depth
        new_child_ids: list[int] = []
        if not leaf.is_terminal and self._depth(nodes, leaf_id) < self.max_depth:
            path = self._path_to_root(nodes, leaf_id)
            path_str = "\n".join(f"  {i}. {n.thought}" for i, n in enumerate(path))
            expand_prompt = (
                f"# Task\n{state['task']}\n\n"
                f"# Reasoning trajectory so far\n{path_str}\n\n"
                f"Propose {self.branching} substantively-DIFFERENT next reasoning "
                "moves to extend this trajectory. Each move should be a single "
                "concrete step. If the trajectory already solves the task, repeat "
                "the final answer as the only candidate."
            )
            try:
                cands = self._expander.invoke(expand_prompt)
                for c in cands.candidates[: self.branching]:
                    child = _Node(id=next_id, thought=c, parent_id=leaf_id)
                    nodes[next_id] = child
                    leaf.children_ids.append(next_id)
                    new_child_ids.append(next_id)
                    next_id += 1
            except Exception as e:
                nodes[leaf_id].features["expand_error"] = str(e)

        # 3. EVALUATE every newly-added child (the simulation step)
        for cid in new_child_ids:
            child = nodes[cid]
            anc_path = self._path_to_root(nodes, cid)
            anc_str = "\n".join(f"  {i}. {n.thought}" for i, n in enumerate(anc_path))
            try:
                verdict = self._evaluator.invoke(
                    f"# Task\n{state['task']}\n\n"
                    f"# Trajectory ending at this leaf\n{anc_str}\n\n"
                    "Evaluate this leaf using the objective features."
                )
                feats = verdict.model_dump()
                v = self._composite_value(feats)
                child.value = v
                child.visits = 1
                child.features = feats
                child.is_terminal = bool(feats.get("is_complete", False))
            except Exception as e:
                child.features = {"eval_error": str(e)}
                child.value = 0.0
                child.visits = 1

        # If the selected leaf had no expansion (depth cap / terminal), do a 1-visit re-eval
        # so its visit count rises and UCB rotates to siblings.
        if not new_child_ids and leaf_id != root_id:
            leaf.visits += 1

        # 4. BACKUP — propagate average reward up the tree
        if new_child_ids:
            best_child_value = max(nodes[cid].value for cid in new_child_ids)
            cur = leaf_id
            while cur is not None:
                node = nodes[cur]
                # Running mean over visits
                node.value = ((node.value * node.visits) + best_child_value) / (node.visits + 1)
                node.visits += 1
                cur = node.parent_id

        # Track best leaf overall (highest value among leaves)
        leaves = [n for n in nodes.values() if not n.children_ids]
        best_leaf = max(leaves, key=lambda n: n.value)
        return {
            "nodes": nodes,
            "next_id": next_id,
            "iteration": iteration,
            "best_leaf_id": best_leaf.id,
            "history": [{
                "stage": "iterate",
                "iteration": iteration,
                "selected_leaf_id": leaf_id,
                "expanded_to": new_child_ids,
                "best_leaf_value": best_leaf.value,
                "tree_size": len(nodes),
            }],
        }

    def _finalize(self, state: LATSState) -> dict[str, Any]:
        nodes = state["nodes"]
        best_id = state.get("best_leaf_id", state["root_id"])
        path = self._path_to_root(nodes, best_id)
        # The final answer = the leaf's thought (often the actual solution)
        return {"final_answer": path[-1].thought}

    # ------------------------------------------------------------------ #
    #  Router                                                             #
    # ------------------------------------------------------------------ #

    def _should_continue(self, state: LATSState) -> str:
        if state.get("iteration", 0) >= self.max_iterations:
            return "finalize"
        # Stop early if we found a complete solution
        nodes = state["nodes"]
        for n in nodes.values():
            if n.is_terminal and n.value >= 8.0:
                return "finalize"
        return "iterate"

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(LATSState)
        g.add_node("init", self._init)
        g.add_node("iterate", self._iterate)
        g.add_node("finalize", self._finalize)
        g.add_edge(START, "init")
        g.add_edge("init", "iterate")
        g.add_conditional_edges(
            "iterate", self._should_continue,
            {"iterate": "iterate", "finalize": "finalize"},
        )
        g.add_edge("finalize", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke(
            {"task": task},
            config={"recursion_limit": max(50, self.max_iterations * 4)},
        )
        nodes = final_state["nodes"]
        leaves = [n for n in nodes.values() if not n.children_ids]
        values = [n.value for n in leaves]
        best_id = final_state.get("best_leaf_id", final_state["root_id"])
        best_path = self._path_to_root(nodes, best_id)
        return ArchitectureResult(
            output=final_state.get("final_answer", ""),
            state={
                "tree_size": len(nodes),
                "leaf_count": len(leaves),
                "best_leaf_value": nodes[best_id].value,
            },
            trace=final_state.get("history", []),
            metadata={
                "tree_size": len(nodes),
                "leaf_count": len(leaves),
                "iterations_used": final_state.get("iteration", 0),
                "best_leaf_value": nodes[best_id].value,
                "best_path_thoughts": [n.thought for n in best_path],
                "best_path_features": nodes[best_id].features,
                "leaf_values": sorted(values, reverse=True),
                "leaf_values_spread": (max(values) - min(values)) if values else 0,
                "max_iterations": self.max_iterations,
                "branching": self.branching,
            },
        )
