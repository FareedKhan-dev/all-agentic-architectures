"""Self-Discover — agent composes its own reasoning structure from atomic modules.

Stages (run once per task):
  1. **SELECT** — pick a subset of atomic reasoning modules relevant to the task.
  2. **ADAPT** — rephrase each selected module in task-specific language.
  3. **IMPLEMENT** — turn the adapted modules into a concrete step-by-step plan.
  4. **SOLVE** — execute the plan and produce the final answer.

Stages 1-3 are *structure discovery* — the agent designs its own reasoning recipe.
Stage 4 is *structure execution* — following that recipe to answer.

Builds on **Tree of Thoughts** (notebook 09): both explicitly enumerate reasoning
moves rather than relying on chain-of-thought to wander into the right shape.
ToT enumerates *candidate thoughts at each step* and scores them; Self-Discover
enumerates *reasoning modules upfront* and composes them into a plan once.

Origin: Zhou et al., *Self-Discover: Large Language Models Self-Compose
Reasoning Structures* (Google DeepMind, 2024).
https://arxiv.org/abs/2402.03620

There is **no LLM-as-Scorer step anywhere** in this architecture — SELECT is a
categorical multi-pick, ADAPT and IMPLEMENT produce text, SOLVE produces an
answer. No numeric judgement → no flat-scoring pathology to fix.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


# ---------------------------------------------------------------------------
# Module library — curated subset of the paper's 39 atomic reasoning modules
# ---------------------------------------------------------------------------
MODULE_LIBRARY: list[str] = [
    "Critical thinking — question the question's assumptions and surface biases",
    "Break the problem into sub-problems and solve each",
    "Identify the type of problem (logic, math, classification, planning, ...)",
    "Identify the goal — what specifically must the final answer contain",
    "List the relevant facts, constraints, and unknowns",
    "Consider analogies — what problem is this structurally similar to?",
    "Brainstorm alternative interpretations of the question",
    "Step-by-step reasoning from premises to conclusion",
    "Consider counterexamples that would falsify a candidate answer",
    "Make a table, list, or diagram to organize the information",
    "Reverse-engineer from a candidate answer back to the premises",
    "Evaluate trade-offs between competing options explicitly",
    "Devise an algorithm or procedure that always finds the answer",
    "Synthesize multiple perspectives or stakeholder views",
    "Check consistency — does the candidate answer satisfy EVERY constraint?",
    "Self-verify — solve a second way and compare the two answers",
]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class _SelectedModules(BaseModel):
    """Stage 1 output."""

    selected_ids: list[int] = Field(
        description=(
            "Indices (0-based) into the MODULE_LIBRARY of the atomic reasoning "
            "modules most relevant to this task type. Pick 3-6 — enough to "
            "structure the reasoning, few enough to keep the plan focused."
        ),
        min_length=2,
        max_length=8,
    )
    rationale: str = Field(
        description="ONE SENTENCE: why these specific modules suit this task type."
    )


class _AdaptedModule(BaseModel):
    original: str = Field(description="The original module text, copied verbatim.")
    adapted: str = Field(
        description="The module rephrased in task-specific language. "
                    "Replace generic words with concrete nouns from the task."
    )


class _AdaptedModules(BaseModel):
    """Stage 2 output."""

    items: list[_AdaptedModule] = Field(
        description="One adapted-module entry per selected module, in the same order."
    )


class _PlanStep(BaseModel):
    step_number: int = Field(ge=1, description="1-indexed step number.")
    description: str = Field(
        description="What the solver does in this step. Reference which adapted "
                    "module(s) this step draws on."
    )
    expected_output: str = Field(
        description="Format / type of what this step should produce "
                    "(e.g., 'a Python list of pairs', 'an ordered chain of names')."
    )


class _ReasoningPlan(BaseModel):
    """Stage 3 output."""

    steps: list[_PlanStep] = Field(min_length=2, max_length=8)
    final_answer_format: str = Field(
        description="Exact format the final answer should take "
                    "(e.g., 'a single comma-separated list of names from tallest to shortest')."
    )


class _Solution(BaseModel):
    """Stage 4 output."""

    step_outputs: list[str] = Field(
        description="One concise string per plan step recording what was produced "
                    "for that step. Same length as the plan's steps."
    )
    final_answer: str = Field(
        description="JUST the final answer in the requested format — no preface, "
                    "no explanation, no markdown."
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class SelfDiscoverState(TypedDict, total=False):
    task: str

    # Stage 1
    selected_ids: list[int]
    selected_modules: list[str]
    selection_rationale: str

    # Stage 2
    adapted_modules: list[dict[str, str]]

    # Stage 3
    plan_steps: list[dict[str, Any]]
    final_answer_format: str

    # Stage 4
    step_outputs: list[str]
    final_answer: str

    history: Annotated[list[dict[str, Any]], operator.add]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class SelfDiscover(Architecture):
    """SELECT → ADAPT → IMPLEMENT → SOLVE — agent composes its reasoning recipe."""

    name = "self_discover"
    description = (
        "Compose a custom reasoning structure for each task from a library of "
        "atomic reasoning modules, then execute that structure to solve. "
        "Four-stage pipeline: SELECT, ADAPT, IMPLEMENT, SOLVE."
    )
    reference = "https://arxiv.org/abs/2402.03620"

    def __init__(self, modules: list[str] | None = None, **kwargs: Any) -> None:
        """
        Args:
            modules: optional custom module library (replaces the default 16).
        """
        super().__init__(**kwargs)
        self.modules: list[str] = list(modules) if modules else list(MODULE_LIBRARY)
        # Pre-bind structured-output wrappers once per architecture instance.
        self._selector = self.llm.with_structured_output(_SelectedModules)
        self._adapter = self.llm.with_structured_output(_AdaptedModules)
        self._implementer = self.llm.with_structured_output(_ReasoningPlan)
        self._solver = self.llm.with_structured_output(_Solution)

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _module_menu(self) -> str:
        return "\n".join(f"  [{i}] {m}" for i, m in enumerate(self.modules))

    def _select(self, state: SelfDiscoverState) -> dict[str, Any]:
        prompt = (
            "You will be given a task and a library of atomic reasoning modules. "
            "Pick the modules whose application would most help structure the "
            "reasoning. Do NOT solve the task yet — only select modules.\n\n"
            f"## Module library\n{self._module_menu()}\n\n"
            f"## Task\n{state['task']}\n\n"
            "Return the indices of the 3-6 most useful modules and a one-sentence "
            "rationale for the selection."
        )
        verdict = self._selector.invoke(prompt)
        # Clamp indices into range so a hallucinated id can't blow up downstream.
        ids = [i for i in verdict.selected_ids if 0 <= i < len(self.modules)]
        selected = [self.modules[i] for i in ids]
        return {
            "selected_ids": ids,
            "selected_modules": selected,
            "selection_rationale": verdict.rationale,
            "history": [{
                "stage": "select",
                "selected_ids": ids,
                "selected_modules": selected,
                "rationale": verdict.rationale,
            }],
        }

    def _adapt(self, state: SelfDiscoverState) -> dict[str, Any]:
        modules_block = "\n".join(f"- {m}" for m in state["selected_modules"])
        prompt = (
            "Rephrase each of the following reasoning modules in language SPECIFIC "
            "to this task — replace generic words with concrete nouns and verbs "
            "drawn from the task itself. Keep one adapted entry per module, in "
            "the same order. Do NOT solve the task yet.\n\n"
            f"## Task\n{state['task']}\n\n"
            f"## Modules to adapt\n{modules_block}"
        )
        adapted = self._adapter.invoke(prompt)
        items = [{"original": a.original, "adapted": a.adapted} for a in adapted.items]
        return {
            "adapted_modules": items,
            "history": [{"stage": "adapt", "adapted_modules": items}],
        }

    def _implement(self, state: SelfDiscoverState) -> dict[str, Any]:
        adapted_block = "\n".join(
            f"{i+1}. {a['adapted']}" for i, a in enumerate(state["adapted_modules"])
        )
        prompt = (
            "Translate the task-adapted reasoning modules below into a concrete "
            "step-by-step plan. Each plan step references one or more of the "
            "adapted modules and specifies what output it must produce. "
            "The final step's output should be the answer in the format you "
            "designate as `final_answer_format`. Do NOT actually solve yet — "
            "produce the plan only.\n\n"
            f"## Task\n{state['task']}\n\n"
            f"## Adapted modules\n{adapted_block}"
        )
        plan = self._implementer.invoke(prompt)
        steps = [
            {
                "step_number": s.step_number,
                "description": s.description,
                "expected_output": s.expected_output,
            }
            for s in plan.steps
        ]
        return {
            "plan_steps": steps,
            "final_answer_format": plan.final_answer_format,
            "history": [{
                "stage": "implement",
                "plan_steps": steps,
                "final_answer_format": plan.final_answer_format,
            }],
        }

    def _solve(self, state: SelfDiscoverState) -> dict[str, Any]:
        plan_block = "\n".join(
            f"Step {s['step_number']}: {s['description']}\n  → produces: {s['expected_output']}"
            for s in state["plan_steps"]
        )
        prompt = (
            "Execute the plan below to solve the task. For each plan step, write "
            "one concise string capturing what you produced for that step. Then "
            "give the final answer in EXACTLY the format the plan specified.\n\n"
            f"## Task\n{state['task']}\n\n"
            f"## Plan\n{plan_block}\n\n"
            f"## Final answer format\n{state['final_answer_format']}\n\n"
            "Return `step_outputs` (one entry per plan step) and `final_answer` "
            "(in the exact required format — no preface, no explanation)."
        )
        sol = self._solver.invoke(prompt)
        return {
            "step_outputs": list(sol.step_outputs),
            "final_answer": sol.final_answer,
            "history": [{
                "stage": "solve",
                "step_outputs": list(sol.step_outputs),
                "final_answer": sol.final_answer,
            }],
        }

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(SelfDiscoverState)
        g.add_node("select", self._select)
        g.add_node("adapt", self._adapt)
        g.add_node("implement", self._implement)
        g.add_node("solve", self._solve)
        g.add_edge(START, "select")
        g.add_edge("select", "adapt")
        g.add_edge("adapt", "implement")
        g.add_edge("implement", "solve")
        g.add_edge("solve", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke(
            {"task": task},
            config={"recursion_limit": 25},
        )
        return ArchitectureResult(
            output=final_state.get("final_answer", ""),
            state={
                "selected_module_count": len(final_state.get("selected_modules", [])),
                "plan_step_count": len(final_state.get("plan_steps", [])),
            },
            trace=final_state.get("history", []),
            metadata={
                "selected_ids": final_state.get("selected_ids", []),
                "selected_modules": final_state.get("selected_modules", []),
                "plan_step_count": len(final_state.get("plan_steps", [])),
                "plan_steps": final_state.get("plan_steps", []),
                "adapted_modules": final_state.get("adapted_modules", []),
                "step_outputs": final_state.get("step_outputs", []),
                "final_answer_format": final_state.get("final_answer_format", ""),
                "modules_total": len(self.modules),
            },
        )
