"""Blackboard — distributed multi-agent system; agents self-elect based on shared state.

The Blackboard pattern (classical AI, 1980s — HEARSAY-II speech-recognition system)
is a multi-agent architecture **without** a central supervisor. Instead, all agents
read a shared *blackboard* (the global state), and each round every agent decides
for itself whether it has something valuable to contribute given the current state.
The most confident bidder wins and writes to the blackboard. Repeat until no agent
wants to contribute (or budget runs out).

Compared to Multi-Agent (notebook 05) — which has a central Supervisor that *picks*
the next agent — Blackboard is fully decentralised: agents *self-elect*. This is
useful when:
  - You don't know in advance which expert will be relevant at each step.
  - The team composition is dynamic (agents can join / leave mid-run).
  - Opportunistic, exploratory tasks where the "right" next move isn't obvious.

Cost: each round costs N+1 LLM calls (N bids + 1 act) — Blackboard is more
expensive than Multi-Agent but more flexible.

Origin: HEARSAY-II (Erman et al., 1980), classical AI Blackboard architecture.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


# ---------------------------------------------------------------------------
# Default knowledge-source roster — easily overridden
# ---------------------------------------------------------------------------
DEFAULT_KNOWLEDGE_SOURCES: dict[str, str] = {
    "optimist": (
        "You are the OPTIMIST. Argue FOR the thesis with the strongest available "
        "evidence. Focus on positive trends, breakthrough technologies, success cases."
    ),
    "skeptic": (
        "You are the SKEPTIC. Argue AGAINST the thesis or surface its weakest "
        "points. Focus on counter-examples, historical failures, structural obstacles."
    ),
    "historian": (
        "You are the HISTORIAN. Bring historical analogies and base rates. "
        "Find past situations similar to the current question and what they teach."
    ),
    "quantitative": (
        "You are the QUANTITATIVE analyst. Bring numbers, percentages, growth "
        "rates, statistics. Insist on data-grounded claims."
    ),
}


# ---------------------------------------------------------------------------
# Bid schema — each agent's self-assessment per round
# ---------------------------------------------------------------------------
class _AgentBid(BaseModel):
    """One agent's bid to contribute on the current round."""

    will_contribute: bool = Field(
        description=(
            "True iff you have substantive NEW value to add given the current "
            "blackboard. False if your perspective has already been covered or "
            "the conversation is exhausted."
        )
    )
    confidence: int = Field(
        ge=1,
        le=5,
        description="How confident are you the contribution would advance the analysis? 1-5.",
    )
    one_line_preview: str = Field(
        description=(
            "One sentence preview of WHAT you would contribute. "
            "If will_contribute=False, write '(nothing to add)'."
        )
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class BlackboardState(TypedDict, total=False):
    task: str
    blackboard: Annotated[list[dict[str, Any]], operator.add]
    round: int
    max_rounds: int
    next_agent: str
    last_bids: dict[str, dict[str, Any]]
    final_synthesis: str


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class Blackboard(Architecture):
    """Distributed multi-agent system with opportunistic, bid-driven contribution."""

    name = "blackboard"
    description = (
        "Multiple agents share a global blackboard. Each round, every agent "
        "self-assesses whether it has something to contribute; the highest-confidence "
        "bidder writes. Repeat until no agent has anything to add. No central supervisor."
    )
    reference = "Erman et al., The Hearsay-II Speech-Understanding System (1980)"

    def __init__(
        self,
        knowledge_sources: dict[str, str] | None = None,
        max_rounds: int = 6,
        min_confidence: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.knowledge_sources = knowledge_sources or DEFAULT_KNOWLEDGE_SOURCES
        self.max_rounds = max_rounds
        self.min_confidence = min_confidence
        self._bid_llm = self.llm.with_structured_output(_AgentBid)

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_blackboard(entries: list[dict[str, Any]]) -> str:
        if not entries:
            return "(empty — be the first to contribute)"
        return "\n".join(
            f"  [round {e['round']}] {e['agent'].upper()}: {e['content']}"
            for e in entries
        )

    def _bidding_round(self, state: BlackboardState) -> dict[str, Any]:
        round_num = state.get("round", 0)
        if round_num >= self.max_rounds:
            return {"next_agent": "FINISH", "last_bids": {}}

        blackboard_str = self._format_blackboard(state.get("blackboard", []))
        invocation_counts = _count_invocations(state.get("blackboard", []))
        counts_str = (
            ", ".join(f"{a}={c}" for a, c in invocation_counts.items())
            or "(no one has contributed yet)"
        )
        bids: dict[str, dict[str, Any]] = {}

        for name, sys_prompt in self.knowledge_sources.items():
            my_count = invocation_counts.get(name, 0)
            fairness_note = (
                f"You have already contributed {my_count} time(s) this run. "
                if my_count > 0
                else "You have NOT contributed yet this run. "
            )
            prompt = (
                f"## Your role\n{sys_prompt}\n\n"
                f"## Question\n{state['task']}\n\n"
                f"## Blackboard so far\n{blackboard_str}\n\n"
                f"## Contribution counts so far\n{counts_str}\n\n"
                f"## Your status\n{fairness_note}"
                "Other agents who have not yet spoken should generally get priority — "
                "if you've already contributed 2+ times, set will_contribute=False unless "
                "you have something genuinely critical to add that no other role can.\n\n"
                "## Your bid\n"
                "Decide whether you have substantive NEW value to add THIS ROUND given "
                "the points above. Be honest — over-bidding when others haven't spoken "
                "creates a one-sided synthesis."
            )
            bid = self._bid_llm.invoke(prompt)
            bids[name] = bid.model_dump()

        # Pick winner: highest-confidence willing bidder above min_confidence.
        eligible = [
            (n, b) for n, b in bids.items()
            if b["will_contribute"] and b["confidence"] >= self.min_confidence
        ]
        if not eligible:
            return {"next_agent": "FINISH", "last_bids": bids}

        winner_name, _ = max(eligible, key=lambda x: x[1]["confidence"])
        return {"next_agent": winner_name, "last_bids": bids}

    def _act(self, state: BlackboardState) -> dict[str, Any]:
        name = state["next_agent"]
        sys_prompt = self.knowledge_sources[name]
        preview = state.get("last_bids", {}).get(name, {}).get("one_line_preview", "")
        blackboard_str = self._format_blackboard(state.get("blackboard", []))

        prompt = (
            f"## Your role\n{sys_prompt}\n\n"
            f"## Question\n{state['task']}\n\n"
            f"## Blackboard so far\n{blackboard_str}\n\n"
            f"## Your bid preview\n{preview}\n\n"
            "## Your contribution\n"
            "Write 2-4 sentences. Be CONCRETE. Reference numbers, examples, or "
            "historical events when relevant. Do NOT repeat what's already on the "
            "blackboard — add new value."
        )
        contribution = str(self.llm.invoke(prompt).content).strip()
        return {
            "blackboard": [
                {
                    "agent": name,
                    "content": contribution,
                    "round": state.get("round", 0),
                }
            ],
            "round": state.get("round", 0) + 1,
        }

    def _synthesize(self, state: BlackboardState) -> dict[str, Any]:
        entries = state.get("blackboard", [])
        if not entries:
            return {
                "final_synthesis": "(no agent had anything to contribute — nothing to synthesize)"
            }
        bb = "\n".join(
            f"### Round {e['round']} — {e['agent'].title()}\n{e['content']}"
            for e in entries
        )
        prompt = (
            f"## Question\n{state['task']}\n\n"
            f"## Blackboard contents\n{bb}\n\n"
            "Synthesise a 150-200 word balanced answer drawing on ALL the contributions "
            "above. Preserve the multi-perspective nature — don't pick a side. "
            "End with one sentence of overall conclusion."
        )
        return {"final_synthesis": str(self.llm.invoke(prompt).content)}

    # ------------------------------------------------------------------ #
    #  Router                                                             #
    # ------------------------------------------------------------------ #

    def _route(self, state: BlackboardState) -> str:
        nxt = state.get("next_agent", "FINISH")
        return "synthesize" if nxt == "FINISH" else "act"

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(BlackboardState)
        g.add_node("bidding", self._bidding_round)
        g.add_node("act", self._act)
        g.add_node("synthesize", self._synthesize)

        g.add_edge(START, "bidding")
        g.add_conditional_edges(
            "bidding", self._route, {"act": "act", "synthesize": "synthesize"}
        )
        g.add_edge("act", "bidding")
        g.add_edge("synthesize", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        # Each round = 1 bidding + 1 act, so we budget generously.
        config = {"recursion_limit": 4 * self.max_rounds + 10}
        final_state = graph.invoke(
            {"task": task, "round": 0, "max_rounds": self.max_rounds},
            config=config,
        )

        contributions = final_state.get("blackboard", [])
        trace = [{"type": "contribution", **c} for c in contributions]
        return ArchitectureResult(
            output=final_state.get("final_synthesis", ""),
            state={
                "task": task,
                "agent_invocation_counts": _count_invocations(contributions),
            },
            trace=trace,
            metadata={
                "total_rounds": final_state.get("round", 0),
                "agents_available": len(self.knowledge_sources),
                "agents_who_contributed": len({c["agent"] for c in contributions}),
                "max_rounds": self.max_rounds,
            },
        )


def _count_invocations(contributions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in contributions:
        counts[c["agent"]] = counts.get(c["agent"], 0) + 1
    return counts
