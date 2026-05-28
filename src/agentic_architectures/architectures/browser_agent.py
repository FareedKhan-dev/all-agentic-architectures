"""BrowserAgent — REAL browser control via Playwright (no mocking).

Sister to [ComputerUse (nb 34)](./computer_use.py): same safety-gate pattern,
but actions execute on a real headless Chromium browser instead of mutating a
dict.

Tools (all routed through the same Python safety gate before execution):
  - `navigate(url)` — go to a URL.
  - `extract_text()` — return the page's visible text (truncated).
  - `click(text)` — click an element by its visible text content.
  - `answer(value)` — commit final answer.

We use Playwright's **sync API** (`sync_playwright().start()`) because it
composes cleanly with LangGraph nodes; the async API would require rewriting
the architecture as async-throughout.

Requires:
  - `pip install playwright`
  - `playwright install chromium`

Lifecycle: the Playwright browser is opened lazily on first `run()` call and
closed via the architecture's `close()` method. Users should call `arch.close()`
when finished, or use the architecture as a context manager.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult

DEFAULT_BLOCKED_DOMAINS: list[str] = [
    "evil-phishing.com",
    "malware-site.test",
]

DEFAULT_SENSITIVE_PATTERNS: list[str] = [
    "password", "passwd", "ssn", "social security",
    "credit card", "cvv", "api_key", "secret_key",
]


class _BrowserAction(BaseModel):
    action: Literal["navigate", "extract_text", "click", "answer"] = Field(
        description="navigate(target=url) | extract_text() | click(target=visible_text) | answer(value=final_answer)"
    )
    target: str = Field(default="", description="URL for navigate, visible text for click. Empty for extract_text/answer.")
    value: str = Field(default="", description="Final answer text for action='answer'. Empty otherwise.")
    rationale: str = Field(description="ONE sentence: why this action.")


class BrowserAgentState(TypedDict, total=False):
    task: str
    iteration: int
    max_iterations: int
    current_url: str
    last_action: dict[str, Any]
    actions_log: Annotated[list[dict[str, Any]], operator.add]
    page_text: str
    final_answer: str
    history: Annotated[list[dict[str, Any]], operator.add]


class BrowserAgent(Architecture):
    """Real headless-Chromium agent via Playwright, with Python safety gate."""

    name = "browser_agent"
    description = (
        "Real browser control via Playwright. Tools: navigate / extract_text / "
        "click / answer. Every action passes a Python safety gate before "
        "execution (no LLM judgement on safety)."
    )
    reference = "Playwright + safety-gate pattern (composes with nb 14 Dry-Run)"

    def __init__(
        self,
        max_iterations: int = 6,
        headless: bool = True,
        page_text_chars: int = 2500,
        sensitive_patterns: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_iterations = max_iterations
        self.headless = headless
        self.page_text_chars = page_text_chars
        self.sensitive_patterns = [p.lower() for p in (sensitive_patterns or DEFAULT_SENSITIVE_PATTERNS)]
        self.blocked_domains = blocked_domains or list(DEFAULT_BLOCKED_DOMAINS)
        self._decider = self.llm.with_structured_output(_BrowserAction)
        # Playwright is lazily opened on first navigate
        self._pw = None
        self._browser = None
        self._page = None

    # ------------------------------------------------------------------ #
    #  Playwright lifecycle                                               #
    # ------------------------------------------------------------------ #

    def _ensure_browser(self) -> None:
        if self._page is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise ImportError(
                "BrowserAgent requires Playwright. Install with:\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            ) from e
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._page = self._browser.new_page()

    def close(self) -> None:
        """Shut down the Playwright browser. Always call after `run()` is done."""
        if self._page is not None:
            try:
                self._page.close()
            except Exception:
                pass
            self._page = None
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._pw is not None:
            try:
                self._pw.stop()
            except Exception:
                pass
            self._pw = None

    def __enter__(self) -> "BrowserAgent":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    #  Safety gate (pure Python)                                          #
    # ------------------------------------------------------------------ #

    def _check_safety(self, action: dict[str, str]) -> tuple[bool, str]:
        kind = action["action"]
        target = (action.get("target") or "").lower()
        value = (action.get("value") or "").lower()
        if kind == "navigate":
            for dom in self.blocked_domains:
                if dom.lower() in target:
                    return False, f"navigation to blocked domain '{dom}'"
            if not (target.startswith("http://") or target.startswith("https://")):
                return False, f"navigation target must start with http(s)://, got {target[:50]!r}"
        if kind == "answer":
            for pat in self.sensitive_patterns:
                if pat in value:
                    return False, f"answer contains blocked pattern '{pat}'"
        return True, "ok"

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _decide(self, state: BrowserAgentState) -> dict[str, Any]:
        iter_count = state.get("iteration", 0) + 1
        force = iter_count >= state.get("max_iterations", self.max_iterations)
        ctx = (
            f"Current URL: {state.get('current_url', '(no navigation yet)')}\n"
            f"Last page-text snippet (first 600 chars): {(state.get('page_text', '') or '(none)')[:600]}\n"
            f"Recent actions: {[a['action'] for a in (state.get('actions_log', []) or [])[-5:]]}"
        )
        prompt = (
            "You control a headless web browser with 4 actions: navigate, extract_text, click, answer.\n\n"
            f"# Task\n{state['task']}\n\n"
            f"# Browser state\n{ctx}\n\n"
            f"# Iteration {iter_count}/{state.get('max_iterations', self.max_iterations)}\n"
            "Pick ONE action. Use navigate to load a page, extract_text to read it, click to follow links, answer when done.\n"
            "RULES:\n"
            "  - Do NOT call extract_text more than once on the same URL — the prior snippet is shown above; use it.\n"
            "  - If the page_text snippet already contains enough to answer the task, set action='answer' immediately with the final answer in `value`.\n"
            "  - Don't repeat the previous action."
        )
        if force:
            prompt += "\n\nFINAL ITERATION — you MUST set action='answer' now. Compose the best answer from the page_text shown above."
        try:
            d = self._decider.invoke(prompt)
            return {
                "iteration": iter_count,
                "last_action": d.model_dump(),
                "history": [{"stage": "decide", "iter": iter_count, "action": d.action, "target": d.target[:60]}],
            }
        except Exception as e:
            return {
                "iteration": iter_count,
                "last_action": {"action": "answer", "value": f"(decider error: {e})", "rationale": ""},
                "history": [{"stage": "decide", "error": str(e)}],
            }

    def _execute(self, state: BrowserAgentState) -> dict[str, Any]:
        action = state["last_action"]
        kind = action["action"]
        allowed, reason = self._check_safety(action)
        log_entry = {**action, "allowed": allowed, "block_reason": "" if allowed else reason}

        if not allowed:
            return {
                "actions_log": [log_entry],
                "history": [{"stage": "execute", "kind": kind, "blocked": reason}],
            }

        if kind == "answer":
            return {
                "actions_log": [log_entry],
                "final_answer": action.get("value", ""),
                "history": [{"stage": "execute", "kind": "answer"}],
            }

        # All non-answer actions need the browser
        try:
            self._ensure_browser()
        except Exception as e:
            return {
                "actions_log": [{**log_entry, "error": str(e)}],
                "history": [{"stage": "execute", "kind": kind, "browser_error": str(e)}],
            }

        update: dict[str, Any] = {"actions_log": [log_entry]}
        try:
            if kind == "navigate":
                url = action["target"]
                self._page.goto(url, wait_until="domcontentloaded", timeout=15000)  # type: ignore[union-attr]
                update["current_url"] = self._page.url  # type: ignore[union-attr]
                update["history"] = [{"stage": "execute", "kind": "navigate", "url": self._page.url}]  # type: ignore[union-attr]
            elif kind == "extract_text":
                text = self._page.locator("body").inner_text(timeout=5000)  # type: ignore[union-attr]
                snippet = text[: self.page_text_chars]
                update["page_text"] = snippet
                update["history"] = [{"stage": "execute", "kind": "extract_text", "chars": len(snippet)}]
            elif kind == "click":
                target = action["target"]
                # Click by visible text (Playwright's get_by_text is fuzzy-tolerant)
                self._page.get_by_text(target, exact=False).first.click(timeout=5000)  # type: ignore[union-attr]
                self._page.wait_for_load_state("domcontentloaded", timeout=5000)  # type: ignore[union-attr]
                update["current_url"] = self._page.url  # type: ignore[union-attr]
                update["history"] = [{"stage": "execute", "kind": "click", "target": target[:60]}]
        except Exception as e:
            update["history"] = [{"stage": "execute", "kind": kind, "error": str(e)[:200]}]
            update["actions_log"] = [{**log_entry, "error": str(e)[:200]}]
        return update

    def _route(self, state: BrowserAgentState) -> str:
        if state.get("last_action", {}).get("action") == "answer":
            return "end"
        if state.get("iteration", 0) >= state.get("max_iterations", self.max_iterations):
            return "end"
        return "decide"

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(BrowserAgentState)
        g.add_node("decide", self._decide)
        g.add_node("execute", self._execute)
        g.add_edge(START, "decide")
        g.add_edge("decide", "execute")
        g.add_conditional_edges(
            "execute", self._route,
            {"decide": "decide", "end": END},
        )
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke(
            {"task": task, "max_iterations": self.max_iterations},
            config={"recursion_limit": max(50, self.max_iterations * 4)},
        )
        log = final_state.get("actions_log", [])
        blocked = [a for a in log if not a.get("allowed", True)]
        kinds = [a["action"] for a in log]
        return ArchitectureResult(
            output=final_state.get("final_answer", ""),
            state={
                "iterations": final_state.get("iteration", 0),
                "current_url": final_state.get("current_url", ""),
                "n_blocked": len(blocked),
            },
            trace=final_state.get("history", []),
            metadata={
                "iterations": final_state.get("iteration", 0),
                "current_url": final_state.get("current_url", ""),
                "page_text_chars": len(final_state.get("page_text", "")),
                "action_sequence": kinds,
                "n_actions": len(log),
                "n_blocked": len(blocked),
                "n_navigate": kinds.count("navigate"),
                "n_extract": kinds.count("extract_text"),
                "n_click": kinds.count("click"),
                "actions_log": log,
            },
        )
