"""Library of agentic architectures.

Each architecture is a class implementing `agentic_architectures.architectures.base.Architecture`.
Architectures are registered lazily — importing this package does not import every
LangChain integration. Import the specific architecture you need:

    from agentic_architectures.architectures.reflection import Reflection
    arch = Reflection()
    result = arch.run("Write a Python function that ...")
"""

from __future__ import annotations

from agentic_architectures.architectures.adaptive_rag import AdaptiveRAG
from agentic_architectures.architectures.agent_workflow_memory import AgentWorkflowMemory
from agentic_architectures.architectures.agentic_rag import AgenticRAG
from agentic_architectures.architectures.base import Architecture, ArchitectureResult
from agentic_architectures.architectures.blackboard import Blackboard
from agentic_architectures.architectures.browser_agent import BrowserAgent
from agentic_architectures.architectures.cellular_automata import CellularAutomata
from agentic_architectures.architectures.chain_of_verification import ChainOfVerification
from agentic_architectures.architectures.computer_use import ComputerUse
from agentic_architectures.architectures.constitutional_ai import ConstitutionalAI
from agentic_architectures.architectures.corrective_rag import CorrectiveRAG
from agentic_architectures.architectures.debate import Debate
from agentic_architectures.architectures.dry_run import DryRun
from agentic_architectures.architectures.ensemble import Ensemble
from agentic_architectures.architectures.episodic_semantic import EpisodicSemanticAgent
from agentic_architectures.architectures.graph_memory import GraphMemoryAgent
from agentic_architectures.architectures.graph_rag import GraphRAG
from agentic_architectures.architectures.lats import LATS
from agentic_architectures.architectures.memgpt import MemGPT
from agentic_architectures.architectures.mental_loop import MentalLoop
from agentic_architectures.architectures.meta_controller import MetaController
from agentic_architectures.architectures.multi_agent import MultiAgent
from agentic_architectures.architectures.pev import PEV
from agentic_architectures.architectures.planning import Planning
from agentic_architectures.architectures.react import ReAct
from agentic_architectures.architectures.reflection import Reflection
from agentic_architectures.architectures.reflexion import Reflexion
from agentic_architectures.architectures.reflexive_metacognitive import ReflexiveMetacognitive
from agentic_architectures.architectures.rlhf import RLHFSelfImprovement
from agentic_architectures.architectures.self_consistency import SelfConsistency
from agentic_architectures.architectures.self_discover import SelfDiscover
from agentic_architectures.architectures.self_rag import SelfRAG
from agentic_architectures.architectures.storm import STORM
from agentic_architectures.architectures.swe_agent import SWEAgent
from agentic_architectures.architectures.tool_use import ToolUse
from agentic_architectures.architectures.tree_of_thoughts import TreeOfThoughts
from agentic_architectures.architectures.voyager import Voyager

__all__ = [
    "AdaptiveRAG",
    "AgentWorkflowMemory",
    "AgenticRAG",
    "Architecture",
    "ArchitectureResult",
    "Blackboard",
    "BrowserAgent",
    "CellularAutomata",
    "ChainOfVerification",
    "ComputerUse",
    "ConstitutionalAI",
    "CorrectiveRAG",
    "Debate",
    "DryRun",
    "Ensemble",
    "EpisodicSemanticAgent",
    "GraphMemoryAgent",
    "GraphRAG",
    "LATS",
    "MemGPT",
    "MentalLoop",
    "MetaController",
    "MultiAgent",
    "PEV",
    "Planning",
    "RLHFSelfImprovement",
    "ReAct",
    "Reflection",
    "Reflexion",
    "ReflexiveMetacognitive",
    "SelfConsistency",
    "SelfDiscover",
    "STORM",
    "SWEAgent",
    "SelfRAG",
    "ToolUse",
    "TreeOfThoughts",
    "Voyager",
]
