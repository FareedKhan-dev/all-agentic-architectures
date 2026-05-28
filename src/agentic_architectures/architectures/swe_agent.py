"""SWE-Agent — code-repo agent with sandboxed file-system tools.

Agent loop over four tools:
  - `list_files()` — list files in the working dir.
  - `read_file(path)` — return file contents.
  - `write_file(path, content)` — write/overwrite.
  - `run_check(path)` — exec the file in a subprocess; return stdout/stderr/returncode.

All paths are constrained to a `working_dir` sandbox; absolute paths and `..`
escapes are rejected.

Origin: Yang et al., *SWE-Agent: Agent-Computer Interface Enables Software
Engineering Language Models* (Princeton 2024). https://arxiv.org/abs/2405.15793

Simplified faithful-to-paper version — we model the LM-tool loop without
the full ACI plumbing.
"""

from __future__ import annotations

import operator
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


class _SWEAction(BaseModel):
    action: Literal["list_files", "read_file", "write_file", "run_check", "answer"] = Field(
        description="Choose ONE: list_files, read_file, write_file, run_check, or answer (commit final response)."
    )
    path: str = Field(
        default="",
        description="Relative path within the sandbox. Required for read_file/write_file/run_check."
    )
    content: str = Field(
        default="",
        description="File content. Required for write_file (the full new contents)."
    )
    answer: str = Field(
        default="",
        description="Final answer text. Required for action='answer'."
    )
    rationale: str = Field(description="ONE sentence: why this action.")


class SWEAgentState(TypedDict, total=False):
    task: str
    iteration: int
    max_iterations: int
    actions: Annotated[list[dict[str, Any]], operator.add]
    last_action: dict[str, Any]
    observations: Annotated[list[str], operator.add]
    final_answer: str
    history: Annotated[list[dict[str, Any]], operator.add]


class SWEAgent(Architecture):
    """Code-repo agent with sandboxed FS tools (list / read / write / run / answer)."""

    name = "swe_agent"
    description = (
        "Agent loop with 4 file-system tools constrained to a working_dir. "
        "Decide → execute → loop until answer."
    )
    reference = "https://arxiv.org/abs/2405.15793"

    def __init__(
        self,
        working_dir: str | Path,
        max_iterations: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.working_dir = Path(working_dir).resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.max_iterations = max_iterations
        self._decider = self.llm.with_structured_output(_SWEAction)

    def _safe_path(self, p: str) -> Path:
        """Resolve and verify the path stays inside working_dir."""
        full = (self.working_dir / p).resolve()
        try:
            full.relative_to(self.working_dir)
        except ValueError as e:
            raise PermissionError(f"path '{p}' escapes sandbox") from e
        return full

    def _decide(self, state: SWEAgentState) -> dict[str, Any]:
        obs_block = "\n".join(f"  [{i}] {o[:300]}" for i, o in enumerate(state.get("observations", []))) or "(no observations yet)"
        iter_count = state.get("iteration", 0) + 1
        force = iter_count >= state.get("max_iterations", self.max_iterations)
        prompt = (
            f"You are a software-engineering agent in a sandbox directory '{self.working_dir.name}'. "
            "You have 5 tools: list_files, read_file, write_file, run_check, answer.\n\n"
            f"# Task\n{state['task']}\n\n"
            f"# Observations so far\n{obs_block}\n\n"
            f"# Iteration {iter_count}/{state.get('max_iterations', self.max_iterations)}\n"
            "Pick ONE action to make progress. Don't repeat the same read on a file you already saw."
        )
        if force:
            prompt += "\n\nNOTE: Final iteration — set action='answer'."
        try:
            d = self._decider.invoke(prompt)
            return {
                "iteration": iter_count,
                "last_action": d.model_dump(),
                "actions": [d.model_dump()],
                "history": [{"stage": "decide", "iter": iter_count, "action": d.action, "path": d.path}],
            }
        except Exception as e:
            return {
                "iteration": iter_count,
                "last_action": {"action": "answer", "path": "", "content": "", "answer": f"(decider error: {e})", "rationale": ""},
                "actions": [{"action": "answer"}],
                "history": [{"stage": "decide", "error": str(e)}],
            }

    def _execute(self, state: SWEAgentState) -> dict[str, Any]:
        a = state["last_action"]
        kind = a["action"]
        obs = ""
        try:
            if kind == "list_files":
                files = sorted(p.name for p in self.working_dir.iterdir())
                obs = f"[list_files] {files}"
            elif kind == "read_file":
                full = self._safe_path(a["path"])
                obs = f"[read_file {a['path']}]\n{full.read_text(encoding='utf-8')[:1500]}"
            elif kind == "write_file":
                full = self._safe_path(a["path"])
                full.parent.mkdir(parents=True, exist_ok=True)
                full.write_text(a["content"], encoding="utf-8")
                obs = f"[write_file {a['path']}] wrote {len(a['content'])} chars"
            elif kind == "run_check":
                full = self._safe_path(a["path"])
                proc = subprocess.run(
                    [sys.executable, str(full)],
                    cwd=self.working_dir,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                obs = f"[run_check {a['path']}] rc={proc.returncode}\nstdout: {proc.stdout[:400]}\nstderr: {proc.stderr[:400]}"
            elif kind == "answer":
                return {
                    "final_answer": a["answer"],
                    "history": [{"stage": "execute", "kind": "answer"}],
                }
        except Exception as e:
            obs = f"[{kind} ERROR] {e}"
        return {
            "observations": [obs],
            "history": [{"stage": "execute", "kind": kind, "obs_chars": len(obs)}],
        }

    def _route(self, state: SWEAgentState) -> str:
        if state.get("last_action", {}).get("action") == "answer":
            return "end"
        if state.get("iteration", 0) >= state.get("max_iterations", self.max_iterations):
            return "end"
        return "decide"

    def build(self) -> Any:
        g: StateGraph = StateGraph(SWEAgentState)
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
        actions = final_state.get("actions", [])
        kinds = [a["action"] for a in actions]
        return ArchitectureResult(
            output=final_state.get("final_answer", ""),
            state={
                "iterations": final_state.get("iteration", 0),
                "n_actions": len(actions),
            },
            trace=final_state.get("history", []),
            metadata={
                "iterations": final_state.get("iteration", 0),
                "action_sequence": kinds,
                "n_list_files": kinds.count("list_files"),
                "n_read_file": kinds.count("read_file"),
                "n_write_file": kinds.count("write_file"),
                "n_run_check": kinds.count("run_check"),
                "observations": final_state.get("observations", []),
                "working_dir": str(self.working_dir),
            },
        )
