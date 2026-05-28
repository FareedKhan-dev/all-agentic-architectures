"""Computer-Use — screenshot + click/type loop with hard safety gate.

A faithful-to-shape version of Anthropic's Computer-Use agent, with TWO
crucial differences for safety:

1. **Mock screen environment** — no actual screenshots, no actual clicks.
   The architecture maintains an in-memory `dict` representing the screen
   state; the agent's actions mutate that dict.
2. **Hard Python safety gate** — every action is screened against a list of
   blocked patterns (sensitive data, dangerous URLs) BEFORE execution.
   Categorical `allowed: bool` per action (deterministic-picker).

For real Computer-Use deployment you'd compose this with sandboxing (VM /
container) and human-in-the-loop confirmation. See handoff §11 for the
production safety recipe.

Origin: Anthropic, *Computer-Use* (2024). Architecture inspired by the
Anthropic Computer-Use public release; this is a simplified pedagogical
version that demonstrates the safety-gate pattern.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


DEFAULT_SENSITIVE_PATTERNS: list[str] = [
    "password",
    "passwd",
    "ssn",
    "social security",
    "credit card",
    "cvv",
    "api_key",
    "secret_key",
]


class _ComputerAction(BaseModel):
    action: Literal["screenshot", "click", "type", "navigate", "submit", "answer"] = Field(
        description="Choose ONE action. screenshot = read current screen state; "
                    "click = click a named element; type = enter text into focused field; "
                    "navigate = go to a URL; submit = submit the active form; answer = done."
    )
    target: str = Field(
        default="",
        description="Element name (for click/submit), URL (for navigate), or empty."
    )
    value: str = Field(
        default="",
        description="Text to type (for type). For answer: the final answer."
    )
    rationale: str = Field(description="ONE sentence.")


class ComputerUseState(TypedDict, total=False):
    task: str
    iteration: int
    max_iterations: int
    screen: dict[str, Any]
    last_action: dict[str, Any]
    actions_attempted: Annotated[list[dict[str, Any]], operator.add]
    blocked_count: int
    final_answer: str
    history: Annotated[list[dict[str, Any]], operator.add]


class ComputerUse(Architecture):
    """Mock-environment Computer-Use agent with hard safety gate."""

    name = "computer_use"
    description = (
        "Agent loop over a mock screen (no real clicks/screenshots). Every action "
        "passes a Python safety gate that blocks sensitive data and dangerous URLs "
        "before execution. Demonstrates the safety pattern without OS risk."
    )
    reference = "Anthropic Computer-Use (2024) — simplified pedagogical version"

    def __init__(
        self,
        initial_screen: dict[str, Any] | None = None,
        sensitive_patterns: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        max_iterations: int = 6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.initial_screen = dict(initial_screen) if initial_screen else {
            "url": "about:blank",
            "elements": [],
            "fields": {},
            "submitted": False,
        }
        self.sensitive_patterns = [p.lower() for p in (sensitive_patterns or DEFAULT_SENSITIVE_PATTERNS)]
        self.blocked_domains = blocked_domains or []
        self.max_iterations = max_iterations
        self._decider = self.llm.with_structured_output(_ComputerAction)

    def _check_safety(self, action: dict[str, str]) -> tuple[bool, str]:
        """Python-only safety verdict. Categorical bool — no LLM in the loop."""
        kind = action["action"]
        value = (action.get("value") or "").lower()
        target = (action.get("target") or "").lower()
        # Block typing sensitive content
        if kind == "type":
            for pat in self.sensitive_patterns:
                if pat in value:
                    return False, f"contains blocked pattern '{pat}'"
        # Block dangerous navigation
        if kind == "navigate":
            for dom in self.blocked_domains:
                if dom.lower() in target:
                    return False, f"navigation to blocked domain '{dom}'"
        return True, "ok"

    def _decide(self, state: ComputerUseState) -> dict[str, Any]:
        iter_count = state.get("iteration", 0) + 1
        screen = state.get("screen", self.initial_screen)
        history_actions = state.get("actions_attempted", [])
        history_block = "\n".join(
            f"  [{i}] action={a['action']} target={a.get('target', '')[:40]} allowed={a.get('allowed')}"
            for i, a in enumerate(history_actions[-6:])
        ) or "(none)"
        force = iter_count >= state.get("max_iterations", self.max_iterations)
        prompt = (
            "You control a web browser via 6 actions: screenshot, click, type, navigate, submit, answer.\n\n"
            f"# Task\n{state['task']}\n\n"
            f"# Current screen state\n{screen}\n\n"
            f"# Recent actions\n{history_block}\n\n"
            f"# Iteration {iter_count}/{state.get('max_iterations', self.max_iterations)}\n"
            "Pick ONE action. **Never type sensitive data** (passwords, SSNs, credit cards). "
            "**Never navigate to blocked domains.** The architecture will block unsafe actions automatically."
        )
        if force:
            prompt += "\n\nNOTE: Final iteration — set action='answer'."
        try:
            d = self._decider.invoke(prompt)
            return {
                "iteration": iter_count,
                "last_action": d.model_dump(),
                "history": [{"stage": "decide", "iter": iter_count, "action": d.action, "target": d.target}],
            }
        except Exception as e:
            return {
                "iteration": iter_count,
                "last_action": {"action": "answer", "value": f"(decider error: {e})", "rationale": ""},
                "history": [{"stage": "decide", "error": str(e)}],
            }

    def _safety_gate(self, state: ComputerUseState) -> dict[str, Any]:
        action = state["last_action"]
        allowed, reason = self._check_safety(action)
        record = {**action, "allowed": allowed, "block_reason": "" if allowed else reason}
        blocked = state.get("blocked_count", 0) + (0 if allowed else 1)
        return {
            "actions_attempted": [record],
            "blocked_count": blocked,
            "history": [{
                "stage": "safety_gate",
                "action": action["action"],
                "allowed": allowed,
                "reason": reason,
            }],
        }

    def _execute(self, state: ComputerUseState) -> dict[str, Any]:
        action = state["last_action"]
        last_record = state.get("actions_attempted", [{}])[-1]
        if not last_record.get("allowed", True):
            # Action was blocked; advise the agent on next iteration
            return {
                "history": [{"stage": "execute", "skipped_blocked": True}],
            }
        screen = dict(state.get("screen", self.initial_screen))
        kind = action["action"]
        if kind == "screenshot":
            # No-op; the screen is already in state
            pass
        elif kind == "click":
            screen["focused"] = action.get("target", "")
        elif kind == "type":
            focused = screen.get("focused", "")
            if focused:
                fields = dict(screen.get("fields", {}))
                fields[focused] = action.get("value", "")
                screen["fields"] = fields
        elif kind == "navigate":
            screen["url"] = action.get("target", "")
            screen["fields"] = {}
            screen["submitted"] = False
        elif kind == "submit":
            screen["submitted"] = True
        elif kind == "answer":
            return {
                "final_answer": action.get("value", ""),
                "history": [{"stage": "execute", "kind": "answer"}],
            }
        return {
            "screen": screen,
            "history": [{"stage": "execute", "kind": kind, "screen_url": screen.get("url")}],
        }

    def _route(self, state: ComputerUseState) -> str:
        if state.get("last_action", {}).get("action") == "answer":
            return "end"
        if state.get("iteration", 0) >= state.get("max_iterations", self.max_iterations):
            return "end"
        return "decide"

    def build(self) -> Any:
        g: StateGraph = StateGraph(ComputerUseState)
        g.add_node("decide", self._decide)
        g.add_node("safety_gate", self._safety_gate)
        g.add_node("execute", self._execute)
        g.add_edge(START, "decide")
        g.add_edge("decide", "safety_gate")
        g.add_edge("safety_gate", "execute")
        g.add_conditional_edges(
            "execute", self._route,
            {"decide": "decide", "end": END},
        )
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke(
            {
                "task": task,
                "screen": dict(self.initial_screen),
                "max_iterations": self.max_iterations,
            },
            config={"recursion_limit": max(50, self.max_iterations * 4)},
        )
        actions = final_state.get("actions_attempted", [])
        return ArchitectureResult(
            output=final_state.get("final_answer", ""),
            state={
                "iterations": final_state.get("iteration", 0),
                "blocked_count": final_state.get("blocked_count", 0),
                "final_screen": final_state.get("screen"),
            },
            trace=final_state.get("history", []),
            metadata={
                "iterations": final_state.get("iteration", 0),
                "n_actions_attempted": len(actions),
                "n_blocked": final_state.get("blocked_count", 0),
                "action_log": actions,
                "final_screen": final_state.get("screen"),
            },
        )
