"""MemGPT — OS-style virtual memory tiers for LLM context.

Two memory tiers:
  - **Context** (analog of RAM): bounded list of recent facts; evicts FIFO when full.
  - **Archival** (analog of disk): vector-backed unbounded store.

Each `run()` call: agent decides action (`write_to_archival`, `search_archival`,
or `answer`) and loops until ready to answer.

Origin: Packer et al., *MemGPT* (2023). https://arxiv.org/abs/2310.08560

Simplified faithful-to-paper version — we model the paging behaviour without
the full tool-use plumbing of the original.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult
from agentic_architectures.memory.vector import VectorMemory


class _MemAction(BaseModel):
    action: Literal["write_to_archival", "search_archival", "answer"] = Field(
        description="Choose: write_to_archival (save a new fact), "
                    "search_archival (look up info from disk), or answer (commit final response)."
    )
    payload: str = Field(
        default="",
        description="For write_to_archival: the fact text. "
                    "For search_archival: the search query. "
                    "For answer: the final answer text.",
    )
    rationale: str = Field(description="ONE sentence.")


class MemGPTState(TypedDict, total=False):
    task: str
    iteration: int
    max_iterations: int
    actions_taken: Annotated[list[dict[str, Any]], operator.add]
    last_action: dict[str, Any]
    final_answer: str
    history: Annotated[list[dict[str, Any]], operator.add]


class MemGPT(Architecture):
    """OS-style tiered memory: context (RAM) + archival (disk)."""

    name = "memgpt"
    description = (
        "Tiered memory architecture. Agent loops over decide-action: "
        "write_to_archival, search_archival, or answer. Context tier evicts FIFO."
    )
    reference = "https://arxiv.org/abs/2310.08560"

    def __init__(
        self,
        context_limit: int = 4,
        max_iterations: int = 5,
        archival: VectorMemory | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.context_limit = context_limit
        self.max_iterations = max_iterations
        # Persistent across run() calls
        self.context_tier: list[str] = []
        self.archival: VectorMemory = archival or VectorMemory(collection_name="memgpt_archival")
        self.archival_count = 0
        self._decider = self.llm.with_structured_output(_MemAction)

    def _push_context(self, fact: str) -> None:
        """Add to context tier; FIFO-evict if over capacity."""
        self.context_tier.append(fact)
        while len(self.context_tier) > self.context_limit:
            evicted = self.context_tier.pop(0)
            # Auto-archive evicted items so info isn't lost
            self.archival.add([Document(page_content=evicted)])
            self.archival_count += 1

    def _decide(self, state: MemGPTState) -> dict[str, Any]:
        ctx_block = "\n".join(f"  - {c}" for c in self.context_tier) or "(empty)"
        actions_so_far = "\n".join(
            f"  [{i}] action={a['action']} payload={a['payload'][:80]}"
            for i, a in enumerate(state.get("actions_taken", []), 1)
        ) or "(none)"
        iter_count = state.get("iteration", 0) + 1
        # Force answer on last iteration
        force_answer = iter_count >= state.get("max_iterations", self.max_iterations)
        prompt = (
            "You are an agent with tiered memory.\n"
            f"## Context tier ({len(self.context_tier)}/{self.context_limit} slots used)\n{ctx_block}\n\n"
            f"## Archival entries available: {self.archival_count}\n\n"
            f"## Task\n{state['task']}\n\n"
            f"## Actions this run\n{actions_so_far}\n\n"
            "Pick ONE action: write_to_archival (save a fact), search_archival (retrieve from "
            "disk), or answer (commit final). Don't repeat the same action twice."
        )
        if force_answer:
            prompt += "\n\nNOTE: Final iteration — you MUST set action='answer'."
        try:
            d = self._decider.invoke(prompt)
            return {
                "iteration": iter_count,
                "last_action": d.model_dump(),
                "actions_taken": [d.model_dump()],
                "history": [{"stage": "decide", "iteration": iter_count, "action": d.action, "payload": d.payload[:80]}],
            }
        except Exception as e:
            return {
                "iteration": iter_count,
                "last_action": {"action": "answer", "payload": f"(decider error: {e})", "rationale": ""},
                "actions_taken": [{"action": "answer", "payload": "", "rationale": "fallback"}],
                "history": [{"stage": "decide", "error": str(e)}],
            }

    def _execute_action(self, state: MemGPTState) -> dict[str, Any]:
        action = state["last_action"]
        kind = action["action"]
        payload = action["payload"]
        if kind == "write_to_archival":
            self.archival.add([Document(page_content=payload)])
            self.archival_count += 1
            # Also write to context tier (the agent saw this fact)
            self._push_context(payload)
            return {"history": [{"stage": "execute", "kind": "write", "archival_size": self.archival_count}]}
        elif kind == "search_archival":
            try:
                docs = self.archival.search(payload, k=3)
                texts = [d.page_content for d in docs]
                # Hits go into context tier
                for t in texts:
                    self._push_context(f"[recalled] {t[:200]}")
                return {"history": [{"stage": "execute", "kind": "search", "n_hits": len(texts)}]}
            except Exception:
                return {"history": [{"stage": "execute", "kind": "search", "n_hits": 0}]}
        elif kind == "answer":
            return {
                "final_answer": payload,
                "history": [{"stage": "execute", "kind": "answer"}],
            }
        return {"history": [{"stage": "execute", "kind": "noop"}]}

    def _route(self, state: MemGPTState) -> str:
        if state.get("last_action", {}).get("action") == "answer":
            return "end"
        if state.get("iteration", 0) >= state.get("max_iterations", self.max_iterations):
            return "end"
        return "decide"

    def build(self) -> Any:
        g: StateGraph = StateGraph(MemGPTState)
        g.add_node("decide", self._decide)
        g.add_node("execute", self._execute_action)
        g.add_edge(START, "decide")
        g.add_edge("decide", "execute")
        g.add_conditional_edges(
            "execute", self._route,
            {"decide": "decide", "end": END},
        )
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        archival_before = self.archival_count
        context_before = len(self.context_tier)
        graph = self.build()
        final_state = graph.invoke(
            {"task": task, "max_iterations": self.max_iterations},
            config={"recursion_limit": max(50, self.max_iterations * 4)},
        )
        actions = final_state.get("actions_taken", [])
        return ArchitectureResult(
            output=final_state.get("final_answer", ""),
            state={
                "iterations": final_state.get("iteration", 0),
                "context_tier_after": len(self.context_tier),
                "archival_after": self.archival_count,
            },
            trace=final_state.get("history", []),
            metadata={
                "iterations": final_state.get("iteration", 0),
                "actions_taken": [a["action"] for a in actions],
                "context_tier_before": context_before,
                "context_tier_after": len(self.context_tier),
                "archival_before": archival_before,
                "archival_after": self.archival_count,
                "archival_grew": self.archival_count - archival_before,
                "context_limit": self.context_limit,
            },
        )
