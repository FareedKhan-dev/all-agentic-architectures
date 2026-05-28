"""Reflexive Metacognitive — agent reasons about its OWN capabilities, chooses action class.

A safety-first pattern for high-stakes advisory (medical, legal, finance). Before
answering, the agent classifies the task against its **self-model** — what kinds
of questions it knows it can handle vs. what should be escalated. Four routes:

  - **answer**: agent has high confidence and the question is in-domain → direct answer.
  - **use_tool**: agent needs external lookup → call a tool first.
  - **partial**: agent answers what it can but flags what it can't.
  - **escalate**: agent declines to answer and routes to a human / domain expert.

The decision is itself structured-output, so the routing is auditable.

Compared to Dry-Run (notebook 14): Dry-Run gates SIDE-EFFECTS before they happen.
Reflexive Metacognitive gates ANSWERS based on the agent's awareness of its own
limits. They compose well — high-stakes pipelines often use both.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult

DEFAULT_SELF_MODEL = (
    "You are a general-purpose AI assistant. Your capabilities:\n"
    "  - STRONG: general knowledge questions, coding, writing, summarisation, math.\n"
    "  - MODERATE: current events (depends on training cutoff), niche technical domains.\n"
    "  - WEAK: medical diagnosis, legal advice, financial advice, anything requiring "
    "    professional credentials or real-time live data the user could verify themselves.\n"
    "Your training cutoff is at most 2 years ago — facts about the last 6 months may be stale."
)


class _MetaDecision(BaseModel):
    """The agent's metacognitive verdict on how to handle the incoming question."""

    capability_match: int = Field(
        ge=1,
        le=5,
        description=(
            "How well does this question match your STRONG capability area? "
            "5 = squarely in your strong zone; 1 = clearly outside your competence "
            "(needs human expert)."
        ),
    )
    requires_external_lookup: bool = Field(
        description=(
            "True iff a definitive answer requires LIVE data (current prices, "
            "today's weather, current laws) that the LLM cannot have."
        ),
    )
    requires_credentials: bool = Field(
        description=(
            "True iff the answer would constitute professional advice that legally "
            "or ethically requires credentials the LLM doesn't have (medical "
            "diagnosis, legal counsel, fiduciary advice)."
        ),
    )
    route: Literal["answer", "use_tool", "partial", "escalate"] = Field(
        description=(
            "How to handle the question:\n"
            "  - 'answer': you have high capability_match, not requires_external_lookup, "
            "    not requires_credentials. Direct answer.\n"
            "  - 'use_tool': capability is fine but requires_external_lookup=True.\n"
            "  - 'partial': moderate capability_match; answer what you can, flag the rest.\n"
            "  - 'escalate': requires_credentials=True OR capability_match<=2."
        ),
    )
    reason: str = Field(description="One short sentence explaining the route choice.")


class ReflexiveMetacognitiveState(TypedDict, total=False):
    task: str
    decision: dict[str, Any]
    final_answer: str
    deterministic_override: bool


class ReflexiveMetacognitive(Architecture):
    """Self-aware agent that picks one of four routes per task."""

    name = "reflexive_metacognitive"
    description = (
        "Agent reasons about its OWN capability against the incoming task and "
        "routes to one of: answer / use_tool / partial / escalate. A Python "
        "deterministic post-check overrides the LLM's route if "
        "requires_credentials=True (always escalate, regardless of confidence)."
    )
    reference = "Metacognition in LLMs; Constitutional AI (Anthropic 2022)."

    def __init__(
        self,
        self_model: str | None = None,
        capability_threshold: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.self_model = self_model or DEFAULT_SELF_MODEL
        self.capability_threshold = capability_threshold
        self._decider = self.llm.with_structured_output(_MetaDecision)

    def _classify(self, state: ReflexiveMetacognitiveState) -> dict[str, Any]:
        prompt = (
            f"## Your self-model\n{self.self_model}\n\n"
            f"## Incoming question\n{state['task']}\n\n"
            "## Your task\n"
            "Apply your self-model to this question and decide how to handle it. "
            "Be HONEST about your limits — escalating is preferable to giving "
            "ungrounded credentialed advice."
        )
        decision = self._decider.invoke(prompt)
        data = decision.model_dump()
        data["llm_route"] = data["route"]  # preserve original
        # DETERMINISTIC OVERRIDE: if requires_credentials, always escalate, regardless of LLM's route.
        if decision.requires_credentials:
            data["route"] = "escalate"
            data["override"] = "python_credentials_override"
        elif decision.capability_match <= 2:
            data["route"] = "escalate"
            data["override"] = "python_low_capability_override"
        else:
            data["override"] = None
        return {
            "decision": data,
            "deterministic_override": data["override"] is not None,
        }

    def _answer(self, state: ReflexiveMetacognitiveState) -> dict[str, Any]:
        prompt = (
            f"## Your self-model\n{self.self_model}\n\n"
            f"## Question\n{state['task']}\n\n"
            "Answer directly. You have determined this is squarely within your competence."
        )
        return {"final_answer": str(self.llm.invoke(prompt).content)}

    def _use_tool(self, state: ReflexiveMetacognitiveState) -> dict[str, Any]:
        # In a production system, this routes to ToolUse (notebook 02). Here we record the recommendation.
        return {
            "final_answer": (
                "[USE_TOOL recommended] This question requires live data beyond my training. "
                f"In a production pipeline I would now invoke a tool (e.g. web search). "
                f"Reason: {state['decision'].get('reason', '')}"
            )
        }

    def _partial(self, state: ReflexiveMetacognitiveState) -> dict[str, Any]:
        prompt = (
            f"## Question\n{state['task']}\n\n"
            "Provide a PARTIAL answer: state what you can address with confidence, "
            "and EXPLICITLY flag what you cannot. Do NOT bluff."
        )
        return {"final_answer": str(self.llm.invoke(prompt).content)}

    def _escalate(self, state: ReflexiveMetacognitiveState) -> dict[str, Any]:
        decision = state["decision"]
        override = decision.get("override")
        return {
            "final_answer": (
                f"[ESCALATE — declined to answer]\n"
                f"This question requires expertise / credentials I don't have. "
                f"You should consult a qualified professional.\n"
                f"  decision route: {decision.get('route')}\n"
                f"  override: {override or '(no Python override; LLM chose escalate)'}\n"
                f"  reason: {decision.get('reason', '')}"
            )
        }

    def _route(self, state: ReflexiveMetacognitiveState) -> str:
        return state["decision"]["route"]

    def build(self) -> Any:
        g: StateGraph = StateGraph(ReflexiveMetacognitiveState)
        g.add_node("classify", self._classify)
        g.add_node("answer", self._answer)
        g.add_node("use_tool", self._use_tool)
        g.add_node("partial", self._partial)
        g.add_node("escalate", self._escalate)
        g.add_edge(START, "classify")
        g.add_conditional_edges(
            "classify",
            self._route,
            {
                "answer": "answer",
                "use_tool": "use_tool",
                "partial": "partial",
                "escalate": "escalate",
            },
        )
        for n in ("answer", "use_tool", "partial", "escalate"):
            g.add_edge(n, END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke({"task": task})
        d = final_state.get("decision", {})
        return ArchitectureResult(
            output=final_state.get("final_answer", ""),
            state={
                "route": d.get("route", "?"),
                "llm_route": d.get("llm_route", "?"),
                "override": d.get("override"),
            },
            trace=[{"type": "decision", **d}],
            metadata={
                "route": d.get("route", "?"),
                "llm_route": d.get("llm_route", "?"),
                "override_applied": final_state.get("deterministic_override", False),
                "capability_match": d.get("capability_match", 0),
                "requires_external_lookup": d.get("requires_external_lookup", False),
                "requires_credentials": d.get("requires_credentials", False),
            },
        )
