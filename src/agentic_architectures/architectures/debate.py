"""Multi-Agent Debate — N agents argue, vote, converge.

K rounds × N agents. Each round, every agent emits its answer plus a brief
critique of the others' prior answers. Final answer = majority vote on the
last round's answers (deterministic-picker via `collections.Counter`).

Sister to **Ensemble** (nb 13): both vote, but Ensemble's voters are
independent; Debate's voters see each other's prior answers and can update.
The cross-pollination is the whole point — it should let an initially-wrong
agent change its mind when confronted with stronger arguments.

Origin: Du et al., *Improving Factuality and Reasoning in Language Models
through Multiagent Debate* (2023). https://arxiv.org/abs/2305.14325
"""

from __future__ import annotations

import operator
from collections import Counter
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class _DebateResponse(BaseModel):
    """One agent's per-round output."""

    answer: str = Field(
        description="JUST the final answer in the requested format — no preface, "
                    "no critique here. (Critiques go in the other field.)"
    )
    critique_of_others: str = Field(
        description="2-3 sentences engaging with the other agents' prior answers — "
                    "identify their strongest argument and your strongest counter, "
                    "or note where you agree. On round 1, write '(round 1 — no prior answers)'."
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class DebateState(TypedDict, total=False):
    task: str
    n_agents: int
    n_rounds: int
    round: int
    # rounds[r] = list of {agent_id, answer, critique} (length n_agents)
    rounds: Annotated[list[list[dict[str, str]]], operator.add]
    final_answer: str
    final_tally: dict[str, int]
    history: Annotated[list[dict[str, Any]], operator.add]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class Debate(Architecture):
    """N agents × K rounds debate → majority vote on final answers."""

    name = "debate"
    description = (
        "Multi-agent debate: N agents each produce an answer + critique per round, "
        "see each other's prior answers, revise. Final = Python-majority-vote over "
        "the last round's answers."
    )
    reference = "https://arxiv.org/abs/2305.14325"

    AGENT_PERSONAS: list[str] = [
        "You are Agent A: rigorous, demands step-by-step reasoning before committing to an answer.",
        "You are Agent B: skeptical, actively looks for counterexamples and edge cases.",
        "You are Agent C: pragmatic, focuses on which answer best fits all available evidence.",
    ]

    def __init__(
        self,
        n_agents: int = 3,
        n_rounds: int = 2,
        agent_personas: list[str] | None = None,
        sample_temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if n_agents < 2:
            raise ValueError("Debate needs at least 2 agents.")
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.sample_temperature = sample_temperature
        self.personas = list(agent_personas) if agent_personas else list(self.AGENT_PERSONAS[:n_agents])
        if len(self.personas) < n_agents:
            # Pad with generic personas if user supplied too few
            self.personas += [f"You are Agent {chr(65+i)}: thoughtful, careful, honest."
                              for i in range(len(self.personas), n_agents)]
        # Pre-bind structured-output sampler
        self._responder = self.llm.bind(temperature=self.sample_temperature).with_structured_output(_DebateResponse)

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _round(self, state: DebateState) -> dict[str, Any]:
        round_num = state.get("round", 0) + 1
        prior = state.get("rounds", [])
        prior_block = ""
        if prior:
            last = prior[-1]
            prior_block = "\n## Prior round answers from all agents\n" + "\n".join(
                f"  - Agent {chr(65+r['agent_id'])}: '{r['answer']}'  — critique: {r['critique'][:200]}"
                for r in last
            )

        responses: list[dict[str, str]] = []
        for i in range(state.get("n_agents", self.n_agents)):
            persona = self.personas[i % len(self.personas)]
            prompt = (
                f"{persona}\n\n"
                f"# Task\n{state['task']}\n"
                f"{prior_block}\n\n"
                f"You are now in round {round_num} of {state.get('n_rounds', self.n_rounds)}. "
                "Read the prior round's answers (if any) and decide: do you stand by your "
                "previous answer (or this round's first instinct), or did another agent's "
                "argument shift your view? Then emit your answer and a brief critique."
            )
            try:
                resp = self._responder.invoke(prompt)
                responses.append({
                    "agent_id": str(i),
                    "answer": resp.answer.strip(),
                    "critique": resp.critique_of_others.strip(),
                })
            except Exception as e:
                responses.append({
                    "agent_id": str(i),
                    "answer": "",
                    "critique": f"(response failed: {e})",
                })

        # Normalise agent_id to int for downstream consumers
        for r in responses:
            r["agent_id"] = int(r["agent_id"])  # type: ignore[assignment]

        return {
            "round": round_num,
            "rounds": [responses],
            "history": [{
                "stage": f"round_{round_num}",
                "answers": [r["answer"] for r in responses],
            }],
        }

    def _vote(self, state: DebateState) -> dict[str, Any]:
        rounds = state.get("rounds", [])
        if not rounds:
            return {"final_answer": "", "final_tally": {}}
        last = rounds[-1]
        norm = lambda s: s.strip().lower().rstrip(".")
        norm_to_raw: dict[str, str] = {}
        for r in last:
            n = norm(r["answer"])
            if n and n not in norm_to_raw:
                norm_to_raw[n] = r["answer"]
        tally = Counter(norm(r["answer"]) for r in last if r["answer"].strip())
        if not tally:
            return {"final_answer": "", "final_tally": {}}
        winner_norm, _ = tally.most_common(1)[0]
        return {
            "final_answer": norm_to_raw.get(winner_norm, winner_norm),
            "final_tally": dict(tally),
            "history": [{"stage": "vote", "tally": dict(tally), "winner": norm_to_raw.get(winner_norm, winner_norm)}],
        }

    # ------------------------------------------------------------------ #
    #  Router                                                             #
    # ------------------------------------------------------------------ #

    def _should_continue(self, state: DebateState) -> str:
        if state.get("round", 0) >= state.get("n_rounds", self.n_rounds):
            return "vote"
        return "round"

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(DebateState)
        g.add_node("round", self._round)
        g.add_node("vote", self._vote)
        g.add_edge(START, "round")
        g.add_conditional_edges(
            "round", self._should_continue,
            {"round": "round", "vote": "vote"},
        )
        g.add_edge("vote", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke(
            {"task": task, "n_agents": self.n_agents, "n_rounds": self.n_rounds},
            config={"recursion_limit": max(50, self.n_rounds * 4)},
        )
        rounds = final_state.get("rounds", [])
        tally = final_state.get("final_tally", {})
        # Per-round agreement track
        round_unique = [len({(r["answer"] or "").strip().lower() for r in rd if r["answer"]}) for rd in rounds]
        return ArchitectureResult(
            output=final_state.get("final_answer", ""),
            state={
                "n_rounds_run": len(rounds),
                "final_unique": round_unique[-1] if round_unique else 0,
                "tally": tally,
            },
            trace=final_state.get("history", []),
            metadata={
                "n_agents": self.n_agents,
                "n_rounds": self.n_rounds,
                "rounds": rounds,
                "round_unique_answer_count": round_unique,
                "final_tally": tally,
                "convergence": (
                    "converged" if len(set([(r["answer"] or "").strip().lower() for r in rounds[-1]] if rounds else [])) <= 1
                    else "partial" if rounds and round_unique[-1] < self.n_agents
                    else "no_convergence"
                ),
            },
        )
