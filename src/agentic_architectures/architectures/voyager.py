"""Voyager — persistent skill library.

Each completed task can leave behind a *skill* (a named, documented Python
snippet). Future tasks first retrieve relevant skills from the library; if
one matches, the agent reuses it. Otherwise the agent writes a new skill and
adds it to the library.

The skill library is in-process state (`arch.skills`) persisting across
`run()` calls, indexed in a vector store for retrieval by description.

Origin: Wang et al., *Voyager: An Open-Ended Embodied Agent with Large
Language Models* (2023). https://arxiv.org/abs/2305.16291

We follow the **structure** of Voyager (write-skill / retrieve-skill /
reuse-skill loop) without the Minecraft-environment specifics — skills here
are Python functions for small computational tasks.
"""

from __future__ import annotations

import operator
import subprocess
import sys
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult
from agentic_architectures.memory.vector import VectorMemory


def _exec_skill(code: str, invocation: str, timeout: int = 5) -> tuple[str, bool, str]:
    """Run skill code + invocation in a fresh Python subprocess.

    Returns (stdout, ok, error). The subprocess gets its own interpreter so
    a bad skill can't corrupt the host process; timeout caps runaway code.
    """
    script = f"{code}\n\n_result = {invocation}\nprint(_result)\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ("", False, f"timeout after {timeout}s")
    except Exception as e:
        return ("", False, f"subprocess error: {e}")
    if proc.returncode != 0:
        return (proc.stdout.strip(), False, proc.stderr.strip()[:300])
    return (proc.stdout.strip(), True, "")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class _SkillDecision(BaseModel):
    """Per-task decision: reuse existing or write new?"""

    action: Literal["reuse", "write_new"] = Field(
        description="'reuse' if the retrieved skill genuinely solves the task; 'write_new' otherwise."
    )
    rationale: str = Field(description="ONE sentence.")


class _NewSkillSpec(BaseModel):
    """Definition of a newly-written skill (one Python function)."""

    function_name: str = Field(
        description="The Python identifier of the function you're defining "
        "(snake_case, e.g. 'factorial', 'fibonacci'). NOT a schema or class name."
    )
    description: str = Field(
        description="ONE sentence describing what the skill does. This is what gets "
        "embedded for future retrieval — be specific."
    )
    code: str = Field(
        description="Complete Python source of the function, including `def name(...):` "
        "and a docstring. No imports outside stdlib. No side effects."
    )
    example_invocation: str = Field(
        description="Example call site (e.g., 'factorial(5)') showing how to use the skill."
    )


class _ApplySkill(BaseModel):
    """The agent's application of a (reused or newly-written) skill to the current task.

    The `invocation` is what we actually run via subprocess; the LLM's
    `predicted_result` is kept for trace comparison vs the real subprocess output.
    """

    invocation: str = Field(
        description="The exact Python expression that, if executed, solves the task "
        "(e.g., 'factorial(7)'). Must call the skill's function."
    )
    predicted_result: str = Field(
        description="What you EXPECT the invocation to produce. Just the value, "
        "no preface. (We will actually run the code and check.)"
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class VoyagerState(TypedDict, total=False):
    task: str
    candidate_skill: dict[str, str] | None
    decision: str
    skill_used: dict[str, str]
    invocation: str
    final_answer: str
    history: Annotated[list[dict[str, Any]], operator.add]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class Voyager(Architecture):
    """Persistent skill library — retrieve, reuse, or write new skills."""

    name = "voyager"
    description = (
        "Skill library that grows across run() calls. Each task: retrieve top skill, "
        "decide reuse/write_new, apply. Library is vector-indexed by skill description."
    )
    reference = "https://arxiv.org/abs/2305.16291"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.skills: list[dict[str, str]] = []
        self._index = VectorMemory(collection_name="voyager_skills")
        self._decider = self.llm.with_structured_output(_SkillDecision)
        self._writer = self.llm.with_structured_output(_NewSkillSpec)
        self._applier = self.llm.with_structured_output(_ApplySkill)

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _retrieve_candidate(self, state: VoyagerState) -> dict[str, Any]:
        if not self.skills:
            return {
                "candidate_skill": None,
                "history": [{"stage": "retrieve_candidate", "library_size": 0}],
            }
        # Vector search on skill description
        try:
            results = self._index.search(state["task"], k=1)
        except Exception:
            results = []
        if not results:
            return {
                "candidate_skill": None,
                "history": [{"stage": "retrieve_candidate", "n_matches": 0}],
            }
        # Find the full skill record matching the retrieved doc's name metadata
        top_name = results[0].metadata.get("name", "")
        match = next((s for s in self.skills if s["name"] == top_name), None)
        return {
            "candidate_skill": match,
            "history": [{"stage": "retrieve_candidate", "candidate_name": top_name}],
        }

    def _decide_reuse(self, state: VoyagerState) -> dict[str, Any]:
        candidate = state.get("candidate_skill")
        if not candidate:
            return {
                "decision": "write_new",
                "history": [{"stage": "decide", "decision": "write_new", "reason": "no candidate"}],
            }
        prompt = (
            f"# Task\n{state['task']}\n\n"
            f"# Candidate existing skill (top match from library)\n"
            f"name: {candidate['name']}\n"
            f"description: {candidate['description']}\n"
            f"code:\n{candidate['code']}\n\n"
            "Should we REUSE this skill to solve the task, or WRITE_NEW because the "
            "candidate doesn't actually fit?"
        )
        try:
            d = self._decider.invoke(prompt)
            return {
                "decision": d.action,
                "history": [{"stage": "decide", "decision": d.action, "rationale": d.rationale}],
            }
        except Exception:
            return {
                "decision": "write_new",
                "history": [{"stage": "decide", "fallback": True}],
            }

    def _write_new_skill(self, state: VoyagerState) -> dict[str, Any]:
        try:
            new = self._writer.invoke(
                f"# Task\n{state['task']}\n\n"
                "Write a SINGLE reusable Python function (no imports outside stdlib, "
                "no side effects) that solves this task class. Then provide an example "
                "invocation that solves THIS specific task."
            )
            skill = {
                "name": new.function_name,
                "description": new.description,
                "code": new.code,
                "example_invocation": new.example_invocation,
            }
            self.skills.append(skill)
            self._index.add([Document(page_content=skill["description"], metadata={"name": skill["name"]})])
            return {
                "skill_used": skill,
                "invocation": new.example_invocation,
                "history": [
                    {"stage": "write_new", "skill_name": skill["name"], "library_size_after": len(self.skills)}
                ],
            }
        except Exception as e:
            return {
                "skill_used": {"name": "_failed", "description": "", "code": "", "example_invocation": ""},
                "invocation": "",
                "history": [{"stage": "write_new", "error": str(e)}],
            }

    def _apply_existing(self, state: VoyagerState) -> dict[str, Any]:
        skill = state["candidate_skill"]  # type: ignore[index]
        prompt = (
            f"# Task\n{state['task']}\n\n"
            f"# Skill to apply (from your library)\n"
            f"name: {skill['name']}\n"
            f"description: {skill['description']}\n"
            f"code:\n{skill['code']}\n\n"
            "Compose the exact Python invocation that solves this task using this skill, "
            "and predict the expected result. We will actually execute the invocation."
        )
        try:
            ap = self._applier.invoke(prompt)
            stdout, ok, err = _exec_skill(skill["code"], ap.invocation)
            final = stdout if ok else f"(execution failed: {err}; LLM predicted: {ap.predicted_result})"
            return {
                "skill_used": skill,
                "invocation": ap.invocation,
                "final_answer": final,
                "history": [
                    {
                        "stage": "apply_existing",
                        "skill_name": skill["name"],
                        "invocation": ap.invocation,
                        "predicted": ap.predicted_result.strip(),
                        "executed_stdout": stdout,
                        "execution_ok": ok,
                        "execution_error": err,
                    }
                ],
            }
        except Exception as e:
            return {
                "skill_used": skill,
                "invocation": "(apply failed)",
                "final_answer": f"(apply error: {e})",
                "history": [{"stage": "apply_existing", "error": str(e)}],
            }

    def _apply_new(self, state: VoyagerState) -> dict[str, Any]:
        skill = state.get("skill_used", {})
        invocation = state.get("invocation", "")
        if not invocation or not skill.get("code"):
            return {
                "final_answer": "(no skill or invocation to apply)",
                "history": [{"stage": "apply_new", "skipped": True}],
            }
        stdout, ok, err = _exec_skill(skill["code"], invocation)
        final = stdout if ok else f"(execution failed: {err})"
        return {
            "final_answer": final,
            "history": [
                {
                    "stage": "apply_new",
                    "invocation": invocation,
                    "executed_stdout": stdout,
                    "execution_ok": ok,
                    "execution_error": err,
                }
            ],
        }

    # ------------------------------------------------------------------ #
    #  Router                                                             #
    # ------------------------------------------------------------------ #

    def _route_decision(self, state: VoyagerState) -> str:
        return "apply_existing" if state.get("decision") == "reuse" else "write_new"

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(VoyagerState)
        g.add_node("retrieve_candidate", self._retrieve_candidate)
        g.add_node("decide_reuse", self._decide_reuse)
        g.add_node("write_new", self._write_new_skill)
        g.add_node("apply_existing", self._apply_existing)
        g.add_node("apply_new", self._apply_new)
        g.add_edge(START, "retrieve_candidate")
        g.add_edge("retrieve_candidate", "decide_reuse")
        g.add_conditional_edges(
            "decide_reuse",
            self._route_decision,
            {"apply_existing": "apply_existing", "write_new": "write_new"},
        )
        g.add_edge("write_new", "apply_new")
        g.add_edge("apply_existing", END)
        g.add_edge("apply_new", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        library_before = len(self.skills)
        graph = self.build()
        final_state = graph.invoke({"task": task}, config={"recursion_limit": 25})
        # Find the apply-stage event to surface execution truth
        apply_evt = next(
            (e for e in reversed(final_state.get("history", [])) if e.get("stage", "").startswith("apply_")),
            {},
        )
        return ArchitectureResult(
            output=final_state.get("final_answer", ""),
            state={
                "library_size": len(self.skills),
                "library_grew": len(self.skills) > library_before,
                "decision": final_state.get("decision"),
            },
            trace=final_state.get("history", []),
            metadata={
                "library_size_before": library_before,
                "library_size_after": len(self.skills),
                "library_grew": len(self.skills) > library_before,
                "decision": final_state.get("decision"),
                "skill_used_name": (final_state.get("skill_used") or {}).get("name", ""),
                "invocation": final_state.get("invocation", ""),
                "candidate_was_offered": final_state.get("candidate_skill") is not None,
                "execution_ok": apply_evt.get("execution_ok", False),
                "executed_stdout": apply_evt.get("executed_stdout", ""),
                "execution_error": apply_evt.get("execution_error", ""),
                "llm_predicted": apply_evt.get("predicted", ""),
            },
        )
