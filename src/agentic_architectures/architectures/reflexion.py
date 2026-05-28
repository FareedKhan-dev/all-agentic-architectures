"""Reflexion — try, evaluate, verbally reflect, store the lesson, retry.

Builds on **Reflection** (notebook 01): same generate → evaluate → retry loop
topology, but adds a *verbal-reflection* node that writes a self-addressed
lesson about why the last attempt failed, and stores that lesson in
:class:`agentic_architectures.memory.EpisodicMemory` so subsequent calls on
structurally-similar tasks can recall it and avoid the same mistake.

Builds on **RLHF Self-Improvement** (notebook 15): same archive-across-run()
pattern, but RLHF stores *positive examples* (final accepted outputs); Reflexion
stores *negative-experience lessons* phrased as transferable corrections.

Origin: Shinn, Cassano, Berman, Gopinath, Narasimhan, Yao,
*Reflexion: Language Agents with Verbal Reinforcement Learning* (2023).
https://arxiv.org/abs/2303.11366

The deterministic-picker pattern (see notebook 10 Mental Loop) applies to the
EVALUATOR: by default a pure-Python checker computes pass/fail from objective
features of the candidate output, sidestepping the LLM-as-Scorer flatness
pathology entirely. If a hybrid LLM-as-Judge is plugged in via the `evaluator`
constructor argument, the :class:`_ReflexionEvaluation` schema below also
follows the deterministic-picker pattern (independent booleans, Python composes
the decision).

State machine
-------------
    [attempt] → [evaluate] ─┐
        ↑                   │  passed OR trial>=max? → [finalize]
        └─ [reflect] ←──────┘  else
"""

from __future__ import annotations

import operator
import re
from typing import TYPE_CHECKING, Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult
from agentic_architectures.memory.episodic import EpisodicMemory

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


# ---------------------------------------------------------------------------
# Self-reflection schema — what the reflection LLM commits to
# ---------------------------------------------------------------------------
class _SelfReflection(BaseModel):
    """The verbal-reflection schema. Three fields so the recalled lesson is
    *actionable* rather than a vague paragraph."""

    root_cause: str = Field(
        description="ONE SENTENCE: precisely why the previous attempt failed. "
                    "Reference the specific feature that broke (e.g., 'line 2 had "
                    "8 syllables instead of 7' — not 'the haiku was off')."
    )
    correction: str = Field(
        description="ONE SENTENCE: the concrete change to make on the next attempt. "
                    "Phrase as an imperative (e.g., 'Count syllables of line 2 out loud "
                    "before submitting and trim if over 7')."
    )
    reflection: str = Field(
        description="2-4 sentences, second person ('You under-counted...'). Combine "
                    "the root_cause + correction + any generalisable lesson for "
                    "similar future tasks. THIS text is stored verbatim in episodic "
                    "memory and recalled when a future task is similar."
    )


# ---------------------------------------------------------------------------
# Optional hybrid evaluator schema — deterministic-picker style
# ---------------------------------------------------------------------------
class _ReflexionEvaluation(BaseModel):
    """Optional secondary LLM-as-Judge for use only with `evaluator='hybrid'`.

    Multiple independent booleans → Python composes the deciding score, same
    pattern as :class:`_EditorCritique` in :mod:`.rlhf`. Not used by the
    default haiku demo because the deterministic Python checker already covers
    pass/fail.
    """

    addresses_constraints: bool = Field(
        description="True iff the candidate satisfies EVERY explicit constraint in the task."
    )
    is_natural_language: bool = Field(
        description="True iff the output reads as fluent natural English, not stilted or padded."
    )
    avoids_meta_commentary: bool = Field(
        description="True iff the output is JUST the answer (no 'Here is...' preface)."
    )
    quality_notes: str = Field(description="ONE SENTENCE of concrete observation.")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ReflexionState(TypedDict, total=False):
    """LangGraph state for one Reflexion run (one task, multiple trials).

    `history` uses `operator.add` so every node appends rather than overwrites —
    one entry per trial from `evaluate`, plus an entry per reflection from
    `reflect` (only when the trial failed and the loop continues).
    """

    task: str
    max_trials: int

    # Per-trial mutables (overwritten each iteration)
    trial: int
    attempt_text: str
    recalled_reflections: list[str]
    evaluator_features: dict[str, Any]
    success: bool
    reflection_text: str
    root_cause: str
    correction: str

    # Accumulators
    history: Annotated[list[dict[str, Any]], operator.add]
    final_output: str


# ---------------------------------------------------------------------------
# Default deterministic checker — constrained haiku
# ---------------------------------------------------------------------------
_VOWEL_GROUP = re.compile(r"[aeiouy]+", re.I)


def _count_syllables_word(word: str) -> int:
    """Naive English heuristic. Vowel-group count, drop trailing silent 'e',
    minimum 1. Good enough for haiku validation, not for real NLP."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = _VOWEL_GROUP.findall(w)
    n = len(groups)
    if w.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def _line_syllables(line: str) -> int:
    return sum(_count_syllables_word(w) for w in line.split())


def default_haiku_checker(candidate: str, task_spec: str) -> dict[str, Any]:
    """Pure-Python checker for the demo scenario. No LLM involved.

    `task_spec` is a tiny ``key=value; key=value`` string parsed out of the
    task text — keeps the checker decoupled from the Reflexion class so users
    can plug their own.

    Example: ``'topic=glacier; required_words=silence,centuries'``

    Returns objective features the reflect node feeds verbatim into the
    reflection LLM prompt, plus the deciding ``passed`` boolean.
    """
    # The spec is the `key=val; key=val` suffix appended after the literal
    # marker `spec=` at the end of the task text. Everything before that marker
    # is human-language and ignored here.
    spec_str = task_spec.rsplit("spec=", 1)[1] if "spec=" in task_spec else ""
    spec: dict[str, str] = {}
    for part in spec_str.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            spec[k.strip()] = v.strip()
    required = [w.strip() for w in spec.get("required_words", "").split(",") if w.strip()]
    topic = spec.get("topic", "").strip()

    lines = [ln.strip() for ln in candidate.strip().splitlines() if ln.strip()]
    # Drop obvious prose preambles ("Here is a haiku...", "Title:", etc.)
    skip_prefixes = ("here", "note", "this", "haiku", "title", "sure")
    haiku_lines = [ln for ln in lines if not ln.lower().startswith(skip_prefixes)]
    haiku_lines = haiku_lines[:3]

    syllable_counts = [_line_syllables(ln) for ln in haiku_lines]
    meets_5_7_5 = syllable_counts == [5, 7, 5]
    has_three_lines = len(haiku_lines) == 3

    text_lower = candidate.lower()
    words_present = {w: (w.lower() in text_lower) for w in required}
    required_words_present = all(words_present.values()) if required else True
    topic_present = (topic.lower() in text_lower) if topic else True

    passed = bool(has_three_lines and meets_5_7_5 and required_words_present)

    return {
        "lines": haiku_lines,
        "syllable_counts": syllable_counts,
        "has_three_lines": has_three_lines,
        "meets_5_7_5": meets_5_7_5,
        "required_words_present": required_words_present,
        "words_present_detail": words_present,
        "topic_present": topic_present,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class Reflexion(Architecture):
    """Try → evaluate → verbal self-reflection → store → retry, with memory
    that persists across :meth:`run` calls on the same instance."""

    name = "reflexion"
    description = (
        "Try → evaluate → verbal self-reflection → store reflection → retry. "
        "Reflections accumulate in EpisodicMemory across run() calls so the "
        "agent learns transferable lessons from past failures."
    )
    reference = "https://arxiv.org/abs/2303.11366"

    def __init__(
        self,
        max_trials: int = 3,
        reflections_to_recall: int = 3,
        evaluator: Callable[[str, str], dict[str, Any]] | None = None,
        reflection_llm: BaseChatModel | None = None,
        episodic: EpisodicMemory | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            max_trials: hard cap on attempt → evaluate iterations per task.
            reflections_to_recall: top-k past reflections to prepend on each
                new attempt's prompt.
            evaluator: ``(candidate, task) -> features dict``. The dict MUST
                contain a ``passed: bool`` key — that's the deciding signal.
                Defaults to :func:`default_haiku_checker`.
            reflection_llm: optionally use a stronger model for the reflection
                step (defaults to the same LLM as the agent).
            episodic: optionally inject a pre-existing :class:`EpisodicMemory`
                (for sharing lessons across multiple Reflexion instances).
        """
        super().__init__(**kwargs)
        self.max_trials = max_trials
        self.reflections_to_recall = reflections_to_recall
        self.evaluator = evaluator if evaluator is not None else default_haiku_checker
        # Persistent across run() calls — the whole point of Reflexion.
        self.episodic = episodic if episodic is not None else EpisodicMemory(
            collection_name="reflexion_lessons",
        )
        self._reflector = (reflection_llm or self.llm).with_structured_output(_SelfReflection)

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _attempt(self, state: ReflexionState) -> dict[str, Any]:
        task = state["task"]
        recalled: list[str] = []
        # Guard: an empty FAISS store can raise on .search; only query when
        # we know we have at least one episode.
        if self.episodic.episodes:
            try:
                episodes = self.episodic.recall(task, k=self.reflections_to_recall)
                recalled = [ep.content for ep in episodes]
            except Exception:
                recalled = []

        reflections_block = ""
        if recalled:
            reflections_block = (
                "\n## Lessons from past attempts (recalled from your episodic memory)\n"
                "Pay close attention — these are mistakes YOU made on similar tasks "
                "and the corrections YOU committed to. Apply them now.\n\n"
                + "\n".join(f"- {r}" for r in recalled)
                + "\n"
            )

        prompt = (
            "You are an agent that learns from your past mistakes via verbal reflection.\n"
            f"\n## Current task\n{task}\n"
            f"{reflections_block}\n"
            "Produce ONLY the answer to the task — no preface, no commentary, "
            "no 'Here is...'. Just the answer itself."
        )
        new_text = str(self.llm.invoke(prompt).content)
        next_trial = state.get("trial", 0) + 1
        return {
            "attempt_text": new_text,
            "trial": next_trial,
            "recalled_reflections": recalled,
        }

    def _evaluate(self, state: ReflexionState) -> dict[str, Any]:
        features = self.evaluator(state["attempt_text"], state["task"])
        success = bool(features.get("passed", False))
        trial_event = {
            "type": "trial",
            "trial": state.get("trial", 0),
            "attempt_text": state["attempt_text"],
            "recalled_reflections": state.get("recalled_reflections", []),
            "features": features,
            "passed": success,
        }
        return {
            "evaluator_features": features,
            "success": success,
            "history": [trial_event],
        }

    def _reflect(self, state: ReflexionState) -> dict[str, Any]:
        feats = state["evaluator_features"]
        failure_summary = "\n".join(
            f"  - {k}: {v}" for k, v in feats.items() if k != "passed"
        )
        prompt = (
            "You attempted a task and the evaluator returned FAIL. Write a verbal "
            "reflection — a self-addressed lesson — so you don't repeat the same "
            "mistake on structurally-similar tasks.\n\n"
            f"## Task\n{state['task']}\n\n"
            f"## Your attempt\n{state['attempt_text']}\n\n"
            f"## Evaluator features (objective)\n{failure_summary}\n\n"
            "Write the reflection in SECOND PERSON ('You under-counted syllables...'). "
            "Be concrete about which feature failed and the precise change to make. "
            "This text will be stored verbatim and recalled the next time you face a "
            "structurally-similar task."
        )
        sr = self._reflector.invoke(prompt)
        # Persist into episodic memory immediately so the *next* trial in this
        # same run AND future run() calls can recall it.
        self.episodic.record(
            sr.reflection,
            trial=state.get("trial", 0),
            task=state["task"],
            root_cause=sr.root_cause,
            correction=sr.correction,
        )
        reflection_event = {
            "type": "reflection",
            "trial": state.get("trial", 0),
            "reflection": sr.reflection,
            "root_cause": sr.root_cause,
            "correction": sr.correction,
        }
        return {
            "reflection_text": sr.reflection,
            "root_cause": sr.root_cause,
            "correction": sr.correction,
            "history": [reflection_event],
        }

    def _finalize(self, state: ReflexionState) -> dict[str, Any]:
        return {"final_output": state.get("attempt_text", "")}

    # ------------------------------------------------------------------ #
    #  Router                                                             #
    # ------------------------------------------------------------------ #

    def _should_reflect(self, state: ReflexionState) -> str:
        if state.get("success", False):
            return "finalize"
        if state.get("trial", 0) >= self.max_trials:
            return "finalize"
        return "reflect"

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(ReflexionState)
        g.add_node("attempt", self._attempt)
        g.add_node("evaluate", self._evaluate)
        g.add_node("reflect", self._reflect)
        g.add_node("finalize", self._finalize)

        g.add_edge(START, "attempt")
        g.add_edge("attempt", "evaluate")
        g.add_conditional_edges(
            "evaluate",
            self._should_reflect,
            {"reflect": "reflect", "finalize": "finalize"},
        )
        g.add_edge("reflect", "attempt")
        g.add_edge("finalize", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        memory_before = len(self.episodic.episodes)
        graph = self.build()
        final_state = graph.invoke(
            {"task": task, "max_trials": self.max_trials},
            config={"recursion_limit": max(50, self.max_trials * 6)},
        )
        history = final_state.get("history", [])
        first_trial = next((h for h in history if h.get("type") == "trial"), {})
        return ArchitectureResult(
            output=final_state.get("final_output", final_state.get("attempt_text", "")),
            state={
                "trials_used": final_state.get("trial", 0),
                "success": final_state.get("success", False),
                "total_reflections_in_memory": len(self.episodic.episodes),
            },
            trace=history,
            metadata={
                "total_trials": final_state.get("trial", 0),
                "trials_to_success": (
                    final_state.get("trial", 0) if final_state.get("success") else None
                ),
                "succeeded": final_state.get("success", False),
                "reflections_recorded_this_run": (
                    len(self.episodic.episodes) - memory_before
                ),
                "reflections_recalled_first_trial": len(
                    first_trial.get("recalled_reflections", [])
                ),
                "total_reflections_in_memory": len(self.episodic.episodes),
                "max_trials": self.max_trials,
                "evaluator_features": final_state.get("evaluator_features", {}),
            },
        )
