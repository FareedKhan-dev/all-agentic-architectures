"""Mixture of Agents (MoA) — layered proposer-aggregator LLM collaboration.

A multi-layer architecture where each layer has N LLM agents. Layer 0 agents
generate proposals from the task directly. Layer k+1 agents see the task AND
all outputs from layer k, producing refined proposals. A final aggregator
synthesises the last layer's outputs. Each layer adds AUXILIARY references
(tool-call output, reasoning traces, etc.) that enrich the prompt context.

This is ENTIRELY different from Ensemble (notebook 13):
- Ensemble: all voters see the SAME flattened prompt; parallel fan-out/fan-in
- MoA: layered pipeline, each agent in layer k sees ALL layer k-1 outputs

Origin: Wang et al., *Mixture-of-Agents Enhances Large Language Model
Capabilities* (2024).  https://arxiv.org/abs/2406.04692

State machine
-------------
    [task] → layer_0 (N agents → N outputs) ──────┐
                                                    │
    layer_1 (N agents → N outputs each reading     │
             all layer_0 outputs) ─────────────────┐
                                                    │
    layer_k (repeat) ───────────────────────────────┤
                                                    │
    [aggregate] ←───────────────────────────────────┘
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


# ---------------------------------------------------------------------------
# Pydantic schema — one per structured-output LLM call
# ---------------------------------------------------------------------------
class _MoAProposal(BaseModel):
    """A single agent's response in one MoA layer."""

    refined_answer: str = Field(
        description="The agent's refined response to the task, incorporating any prior-layer insights."
    )
    key_insight: str = Field(
        description="The single most important insight this agent added beyond what prior layers provided."
    )
    confidence_indicators: list[str] = Field(
        default_factory=list,
        description="1-3 categorical features this agent used to assess answer quality "
        "(e.g. 'factual grounding: strong', 'reasoning depth: adequate', 'completeness: partial'). "
        "These are CATEGORICAL signals — NOT numeric scores. The aggregator uses Python logic, "
        "not an LLM-as-Scorer argmax, to decide weightings.",
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class MoAState(TypedDict, total=False):
    task: str
    layer_outputs: Annotated[list[dict[str, Any]], operator.add]
    current_layer: int
    num_layers: int
    agents_per_layer: int
    final_output: str


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class MixtureOfAgents(Architecture):
    """Multi-layer proposer-aggregator with cross-layer context propagation."""

    name = "mixture_of_agents"
    description = (
        "Layered LLM collaboration: layer 0 agents propose independently; "
        "each subsequent layer sees all prior outputs and refines them; "
        "a final aggregator synthesises the last layer's results. "
        "Distinct from Ensemble (same-task parallel voters) — MoA layers "
        "build on each other's context progressively."
    )
    reference = "https://arxiv.org/abs/2406.04692"

    def __init__(
        self,
        num_layers: int = 2,
        agents_per_layer: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.agents_per_layer = agents_per_layer
        self._proposer = self.llm.with_structured_output(_MoAProposal)

    # ------------------------------------------------------------------ #
    #  Nodes
    # ------------------------------------------------------------------ #

    def _layer_generate(self, state: MoAState) -> dict[str, Any]:
        """Run all agents in the current layer."""
        current_layer = state.get("current_layer", 0)
        task = state["task"]
        num = state.get("num_layers", self.num_layers)
        agents_per = state.get("agents_per_layer", self.agents_per_layer)
        all_prior = state.get("layer_outputs", [])

        layer_results: list[dict[str, Any]] = []

        for agent_idx in range(agents_per):
            persona_idx = agent_idx % 3
            personas = {
                0: "Decomposer",
                1: "Synthesizer",
                2: "Critic",
            }
            persona = personas[persona_idx]

            persona_instructions = {
                "Decomposer": "Break the task into sub-components and build your answer step by step. Be systematic and explicit about your reasoning chain.",
                "Synthesizer": "Look for patterns across information sources and weave them into a coherent, unified answer. Prioritise clarity and narrative flow.",
                "Critic": "Identify gaps, edge cases, assumptions, and potential weaknesses in existing proposals. Your role is to strengthen the answer by surfacing what others missed.",
            }

            # Build context from prior layers
            context_parts: list[str] = [f"## Task\n{task}\n"]
            if current_layer > 0 and all_prior:
                context_parts.append(
                    "## Outputs from previous layers\n"
                    "Below are all proposals from earlier layers. "
                    "Use these to INFORM and IMPROVE your own answer — "
                    "do not just repeat them.\n"
                )
                for prior in all_prior:
                    context_parts.append(
                        f"### Layer {prior['layer']}, Agent {prior['agent_idx']} ({prior['persona']})\n"
                        f"{prior['refined_answer']}\n"
                        f"Key insight: {prior['key_insight']}\n"
                    )

            context_parts.append(
                f"## Your role: {persona} (layer {current_layer}, agent {agent_idx + 1}/{agents_per})\n"
                f"{persona_instructions[persona]}\n\n"
                f"Produce your refined answer, key insight, and confidence indicators."
            )

            prompt = "\n".join(context_parts)
            proposal = self._proposer.invoke(prompt)

            layer_results.append(
                {
                    "layer": current_layer,
                    "agent_idx": agent_idx,
                    "persona": persona,
                    "refined_answer": proposal.refined_answer,
                    "key_insight": proposal.key_insight,
                    "confidence_indicators": proposal.confidence_indicators,
                }
            )

        return {"layer_outputs": layer_results, "current_layer": current_layer + 1}

    def _aggregate(self, state: MoAState) -> dict[str, Any]:
        """Synthesise the final layer's outputs into one answer."""
        all_outputs = state.get("layer_outputs", [])
        if not all_outputs:
            return {"final_output": "(no layer outputs to aggregate)"}

        # Find the last layer's outputs
        last_layer = max(o["layer"] for o in all_outputs)
        last_layer_outputs = [o for o in all_outputs if o["layer"] == last_layer]

        # DETERMINISTIC picker: tally categorical confidence indicators across
        # all agents to decide which signals are strongest — no LLM-as-Scorer argmax.
        indicator_tally: dict[str, int] = {}
        for o in last_layer_outputs:
            for ind in o.get("confidence_indicators", []):
                # Extract the label (before the colon) as the categorical signal
                label = ind.split(":")[0].strip().lower() if ":" in ind else ind.strip().lower()
                indicator_tally[label] = indicator_tally.get(label, 0) + 1

        # Build the synthesis prompt
        sections = []
        for o in last_layer_outputs:
            sections.append(
                f"### Agent {o['agent_idx'] + 1} — {o['persona']}\n"
                f"**Answer:** {o['refined_answer']}\n"
                f"**Key insight:** {o['key_insight']}\n"
                f"**Indicators:** {', '.join(o['confidence_indicators'])}\n"
            )

        sections_text = "\n".join(sections)
        tally_text = (
            f"\n\n## Cross-agent indicator tally (Python-computed)\n"
            f"{', '.join(f'{k}={v}' for k, v in sorted(indicator_tally.items(), key=lambda x: -x[1]))}\n"
        )

        prompt = (
            f"## Original task\n{state['task']}\n\n"
            f"## Final layer ({last_layer}) proposals from {len(last_layer_outputs)} agents\n"
            f"{sections_text}"
            f"{tally_text}\n"
            "## Your task as Aggregator\n"
            "Synthesise a SINGLE final answer that:\n"
            "1. Gives the most likely correct answer in 1-2 sentences.\n"
            "2. Identifies points of AGREEMENT across agents.\n"
            "3. Notes any GENUINE DISAGREEMENT that remains.\n"
            "4. Uses the indicator tally to inform your weighting — "
            "agents consistently flagging the same strength/weakness deserve more weight.\n"
            "Do NOT introduce new facts beyond what the agents provided."
        )
        final = self.llm.invoke(prompt).content
        return {"final_output": str(final)}

    # ------------------------------------------------------------------ #
    #  Router
    # ------------------------------------------------------------------ #

    def _should_continue(self, state: MoAState) -> str:
        num = state.get("num_layers", self.num_layers)
        if state.get("current_layer", 0) >= num:
            return "aggregate"
        return "layer"

    # ------------------------------------------------------------------ #
    #  Build + run
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(MoAState)
        g.add_node("layer", self._layer_generate)
        g.add_node("aggregate", self._aggregate)

        g.add_edge(START, "layer")
        g.add_conditional_edges(
            "layer",
            self._should_continue,
            {"layer": "layer", "aggregate": "aggregate"},
        )
        g.add_edge("aggregate", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke(
            {
                "task": task,
                "current_layer": 0,
                "num_layers": self.num_layers,
                "agents_per_layer": self.agents_per_layer,
            },
            config={"recursion_limit": (self.num_layers * 2) + 5},
        )
        all_outputs = final_state.get("layer_outputs", [])
        total_layers = max((o["layer"] for o in all_outputs), default=-1) + 1

        return ArchitectureResult(
            output=final_state.get("final_output", ""),
            state={
                "total_layers_executed": total_layers,
                "agents_per_layer": self.agents_per_layer,
                "total_agent_calls": len(all_outputs),
            },
            trace=list(all_outputs),
            metadata={
                "num_layers": self.num_layers,
                "layers_executed": total_layers,
                "agents_per_layer": self.agents_per_layer,
                "total_agent_calls": len(all_outputs),
            },
        )
