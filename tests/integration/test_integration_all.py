"""Integration smoke tests — ONE happy-path per architecture (all 35).

Gated by `RUN_INTEGRATION=1`. Run with:

    RUN_INTEGRATION=1 pytest tests/integration -v

Each test instantiates a real LLM (Nebius / Llama by default), invokes the
architecture once on a representative input, and asserts the output is
non-empty + has the expected shape. We do NOT assert correctness here —
that's what the benchmark suite does. This is a "did it execute end-to-end
without exceptions" sweep.

Total runtime at RUN_INTEGRATION=1 with default model: ~15-30 minutes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tests.conftest import RUN_INTEGRATION

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not RUN_INTEGRATION, reason="set RUN_INTEGRATION=1 to enable"),
]


def _llm(model: str = "meta-llama/Llama-3.3-70B-Instruct", temperature: float = 0.2):
    from agentic_architectures import get_llm
    return get_llm(provider="nebius", model=model, temperature=temperature)


def _qwen():
    return _llm(model="Qwen/Qwen3-235B-A22B-Thinking-2507-fast", temperature=0.4)


# ========================================================================== #
#  Phase 2 architectures (1-17)
# ========================================================================== #

def test_reflection_real() -> None:
    from agentic_architectures.architectures import Reflection
    arch = Reflection(llm=_llm(), max_iterations=2, target_score=7)
    r = arch.run("Write a haiku about a glacier.")
    assert r.output and r.metadata["iterations"] >= 1


def test_tool_use_real() -> None:
    from agentic_architectures.architectures import ToolUse
    from agentic_architectures.tools import web_search_tool
    arch = ToolUse(llm=_llm(), tools=[web_search_tool(max_results=3)], max_rounds=3)
    r = arch.run("What is the current price of Apple stock?")
    assert r.output


def test_react_real() -> None:
    from agentic_architectures.architectures import ReAct
    from agentic_architectures.tools import web_search_tool
    arch = ReAct(llm=_llm(), tools=[web_search_tool(max_results=2)])
    r = arch.run("Who is the current CEO of Microsoft?")
    assert r.output


def test_planning_real() -> None:
    from agentic_architectures.architectures import Planning
    arch = Planning(llm=_llm())
    r = arch.run("Plan a 3-day Tokyo itinerary for a vegetarian on $200/day.")
    assert r.output


def test_multi_agent_real() -> None:
    from agentic_architectures.architectures import MultiAgent
    arch = MultiAgent(llm=_llm())
    r = arch.run("Write a 2-paragraph tech blog post about transformer architectures.")
    assert r.output


def test_pev_real() -> None:
    from agentic_architectures.architectures import PEV
    arch = PEV(llm=_llm())
    r = arch.run("Write a 3-step plan to bake a cake.")
    assert r.output


def test_blackboard_real() -> None:
    from agentic_architectures.architectures import Blackboard
    arch = Blackboard(llm=_llm())
    r = arch.run("Analyse the historical and economic impact of the steam engine.")
    assert r.output


def test_episodic_semantic_real() -> None:
    from agentic_architectures.architectures import EpisodicSemanticAgent
    arch = EpisodicSemanticAgent(llm=_llm())
    arch.run("Hi! My name is Alex and I have a cat named Mochi.")
    r = arch.run("What is my cat's name?")
    assert "mochi" in r.output.lower()


def test_tree_of_thoughts_real() -> None:
    from agentic_architectures.architectures import TreeOfThoughts
    arch = TreeOfThoughts(llm=_llm(), branching=2, beam_width=2, max_depth=2)
    r = arch.run("Game of 24. Numbers: [4, 6, 8, 12]. Find an expression that equals 24.")
    assert r.output


def test_mental_loop_real() -> None:
    from agentic_architectures.architectures import MentalLoop
    arch = MentalLoop(llm=_llm(), branching=2, max_steps=2)
    r = arch.run("Plan a 30-minute meal: pasta with vegetables.")
    assert r.output


def test_meta_controller_real() -> None:
    from agentic_architectures.architectures import MetaController, Reflection, ToolUse
    from agentic_architectures.tools import web_search_tool
    arch = MetaController(
        llm=_llm(),
        roster={
            "reflection": Reflection(llm=_llm()),
            "tool_use": ToolUse(llm=_llm(), tools=[web_search_tool(max_results=2)]),
        },
    )
    r = arch.run("Write a haiku about a glacier.")
    assert r.output


def test_graph_memory_real() -> None:
    from agentic_architectures.architectures import GraphMemoryAgent
    arch = GraphMemoryAgent(llm=_llm())
    arch.run("Ada Lovelace wrote the first computer algorithm in 1843.")
    r = arch.run("Who wrote the first computer algorithm?")
    assert "ada" in r.output.lower()


def test_ensemble_real() -> None:
    from agentic_architectures.architectures import Ensemble
    arch = Ensemble(llm=_llm(), aggregation="majority_vote")
    r = arch.run("Is water at 100°C boiling at sea level? Answer YES or NO.")
    assert r.output


def test_dry_run_real() -> None:
    from agentic_architectures.architectures import DryRun
    arch = DryRun(llm=_llm())
    r = arch.run("Delete all files in /etc")
    # Architecture should classify this as destructive and block / reviewer-route
    assert r.metadata


def test_rlhf_real() -> None:
    from agentic_architectures.architectures import RLHFSelfImprovement
    arch = RLHFSelfImprovement(llm=_llm(), max_iterations=1, target_score=7)
    r = arch.run("Write a 50-word product tagline for an artisanal coffee shop.")
    assert r.output and "final_score" in r.metadata


def test_cellular_automata_real() -> None:
    from agentic_architectures.architectures import CellularAutomata
    arch = CellularAutomata(
        llm=_llm(),
        rule_prompt="A cell that is 'tree' next to 'fire' becomes 'fire'. 'Fire' becomes 'ash' next step. 'ash' stays 'ash'. 'empty' stays 'empty'.",
        allowed_states=["empty", "tree", "fire", "ash"],
        grid_size=(3, 3),
        max_steps=2,
    )
    r = arch.run("Initial state: a 3x3 grid with one fire cell in the middle, surrounded by trees.")
    assert r.output


def test_reflexive_metacognitive_real() -> None:
    from agentic_architectures.architectures import ReflexiveMetacognitive
    arch = ReflexiveMetacognitive(llm=_llm())
    r = arch.run("Diagnose my chest pain symptoms (occasional, worse on exertion).")
    # Should escalate medical concerns
    assert r.output


# ========================================================================== #
#  Phase 3 architectures (18-35)
# ========================================================================== #

def test_reflexion_real() -> None:
    from agentic_architectures.architectures import Reflexion
    arch = Reflexion(llm=_llm(), max_trials=2)
    r = arch.run(
        'Write a haiku about glacier. spec=topic=glacier; required_words=silence,centuries'
    )
    assert r.output and "total_trials" in r.metadata


def test_self_discover_real() -> None:
    from agentic_architectures.architectures import SelfDiscover
    arch = SelfDiscover(llm=_qwen())
    r = arch.run("Order these by size, smallest first: elephant, mouse, dog. Return comma-separated.")
    assert "mouse" in r.output.lower()


def test_chain_of_verification_real() -> None:
    from agentic_architectures.architectures import ChainOfVerification
    arch = ChainOfVerification(llm=_llm())
    r = arch.run("Name 3 US presidents from the 1800s.")
    assert r.output


def test_self_consistency_real() -> None:
    from agentic_architectures.architectures import SelfConsistency
    arch = SelfConsistency(llm=_llm(), n_samples=3, sample_temperature=0.7)
    r = arch.run("What is 5 squared? Return only the integer.")
    assert r.output.strip() == "25"


def test_lats_real() -> None:
    from agentic_architectures.architectures import LATS
    arch = LATS(llm=_qwen(), max_iterations=2, branching=2, max_depth=2)
    r = arch.run("Game of 24. Numbers: [4, 6, 8, 12]. Combine all four with +-*/ to make 24.")
    assert r.output


def test_agentic_rag_real() -> None:
    from agentic_architectures.architectures import AgenticRAG
    from agentic_architectures.data import STARDUST_CORPUS
    arch = AgenticRAG(llm=_llm(), documents=STARDUST_CORPUS, max_iterations=3)
    r = arch.run("What propellant does the Phoenix-2 engine use?")
    assert "methalox" in r.output.lower()


def test_corrective_rag_real() -> None:
    from agentic_architectures.architectures import CorrectiveRAG
    from agentic_architectures.data import STARDUST_CORPUS
    arch = CorrectiveRAG(llm=_llm(), documents=STARDUST_CORPUS, top_k=3)
    r = arch.run("What is the maximum payload of the Stardust 9 rocket?")
    assert r.output


def test_self_rag_real() -> None:
    from agentic_architectures.architectures import SelfRAG
    from agentic_architectures.data import STARDUST_CORPUS
    arch = SelfRAG(llm=_llm(), documents=STARDUST_CORPUS, top_k=3)
    r = arch.run("Who founded Stardust Aerospace?")
    assert r.output


def test_adaptive_rag_real() -> None:
    from agentic_architectures.architectures import AdaptiveRAG
    from agentic_architectures.data import STARDUST_CORPUS
    arch = AdaptiveRAG(llm=_llm(), documents=STARDUST_CORPUS, top_k=3)
    r = arch.run("What is 7 plus 8? Return just the integer.")
    assert "15" in r.output


def test_graph_rag_real() -> None:
    from agentic_architectures.architectures import GraphRAG
    from agentic_architectures.data import STARDUST_CORPUS
    arch = GraphRAG(llm=_llm(), documents=STARDUST_CORPUS[:6], max_communities=3)
    r = arch.run("What are the main themes of this knowledge base?")
    assert r.output


def test_debate_real() -> None:
    from agentic_architectures.architectures import Debate
    arch = Debate(llm=_llm(), n_agents=2, n_rounds=2)
    r = arch.run("What is 12 times 12? Return only the integer.")
    assert r.output


def test_voyager_real() -> None:
    from agentic_architectures.architectures import Voyager
    arch = Voyager(llm=_llm())
    r = arch.run("Compute the factorial of 5. Return just the integer.")
    assert r.output.strip() == "120"
    assert r.metadata["execution_ok"] is True


def test_storm_real() -> None:
    from agentic_architectures.architectures import STORM
    arch = STORM(llm=_llm(), n_perspectives=2, questions_per_perspective=1)
    r = arch.run("The role of agentic AI architectures in 2024.")
    assert r.output


def test_memgpt_real() -> None:
    from agentic_architectures.architectures import MemGPT
    arch = MemGPT(llm=_llm(), context_limit=3, max_iterations=3)
    arch.run("Remember this: My favourite colour is teal.")
    arch.run("Remember this: I have a cat named Mochi.")
    r = arch.run("What is my favourite colour?")
    assert r.output


def test_constitutional_ai_real() -> None:
    from agentic_architectures.architectures import ConstitutionalAI
    arch = ConstitutionalAI(llm=_llm(), max_iterations=1)
    r = arch.run("Share your personal opinion about which programming language is best.")
    assert r.output


def test_swe_agent_real() -> None:
    from agentic_architectures.architectures import SWEAgent
    work = Path(tempfile.mkdtemp(prefix="swe_test_"))
    (work / "factorial.py").write_text(
        "def factorial(n):\n    return n * factorial(n - 1)\n\n"
        "if __name__ == '__main__':\n    assert factorial(0) == 1\n    print('PASS')\n"
    )
    arch = SWEAgent(llm=_llm(), working_dir=work, max_iterations=6)
    r = arch.run(
        "There is a bug in factorial.py (missing base case). Read it, fix it, "
        "run_check to verify, then answer."
    )
    assert r.output


def test_computer_use_real() -> None:
    from agentic_architectures.architectures import ComputerUse
    arch = ComputerUse(
        llm=_llm(),
        initial_screen={"url": "https://example.com/login", "elements": ["username_field"], "fields": {}, "submitted": False},
        blocked_domains=["evil-phishing.com"],
        max_iterations=3,
    )
    r = arch.run("Click the username_field and type 'alice', then answer 'done'.")
    assert r.output or r.metadata


def test_browser_agent_real() -> None:
    from agentic_architectures.architectures import BrowserAgent
    arch = BrowserAgent(llm=_llm(), max_iterations=4, headless=True)
    try:
        r = arch.run("Navigate to https://example.com and report the main heading.")
        assert "example" in r.output.lower()
    finally:
        arch.close()


def test_browser_agent_safety_gate_real() -> None:
    from agentic_architectures.architectures import BrowserAgent
    arch = BrowserAgent(
        llm=_llm(),
        max_iterations=3,
        headless=True,
        blocked_domains=["evil-phishing.com"],
    )
    try:
        r = arch.run("Navigate to https://evil-phishing.com/login")
        assert r.metadata["n_blocked"] >= 1
    finally:
        arch.close()


def test_agent_workflow_memory_real() -> None:
    from agentic_architectures.architectures import AgentWorkflowMemory
    arch = AgentWorkflowMemory(llm=_llm())
    arch.run("Summarise and categorise: 'A new study shows octopus REM sleep states.'")
    r = arch.run("Summarise and categorise: 'Congress passed the infrastructure bill on Tuesday.'")
    assert r.output
