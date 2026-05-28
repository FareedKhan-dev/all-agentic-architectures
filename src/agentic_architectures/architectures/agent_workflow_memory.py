"""Agent Workflow Memory (AWM) — mine reusable workflows from past successful traces.

After each solved task, the architecture extracts a high-level **workflow**
(a 3-6 step recipe) and stores it in a vector-indexed library. Future tasks
retrieve the most-similar workflow and use it as a prior — instead of starting
from scratch, the agent re-traces a proven recipe and adapts it.

Sister to [Voyager (nb 29)](./29_voyager.ipynb) (skills = executable code) and
[Reflexion (nb 18)](./18_reflexion.ipynb) (memory of failures): AWM stores
**recipes for success** at the task-strategy level.

Origin: Wang et al., *Agent Workflow Memory* (2024).
https://arxiv.org/abs/2409.07429
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult
from agentic_architectures.memory.vector import VectorMemory


class _Workflow(BaseModel):
    """Extracted reusable recipe."""

    task_type: str = Field(description="A short label for the kind of task this recipe solves.")
    steps: list[str] = Field(
        description="3-6 high-level recipe steps. Each generalisable — "
                    "no task-specific entities, just the strategy.",
        min_length=2, max_length=8,
    )


class _Answer(BaseModel):
    answer: str = Field(description="JUST the final answer; no preface, no explanation.")


class AWMState(TypedDict, total=False):
    task: str
    retrieved_workflow: dict[str, Any] | None
    answer: str
    extracted_workflow: dict[str, Any] | None
    history: Annotated[list[dict[str, Any]], operator.add]


class AgentWorkflowMemory(Architecture):
    """Retrieve workflow → use as prior → answer → extract new workflow."""

    name = "agent_workflow_memory"
    description = (
        "After each task, extract a high-level workflow recipe and store it. "
        "Future tasks retrieve similar workflows and use them as priors."
    )
    reference = "https://arxiv.org/abs/2409.07429"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.workflows: list[dict[str, Any]] = []
        self._index = VectorMemory(collection_name="awm_workflows")
        self._extractor = self.llm.with_structured_output(_Workflow)
        self._answerer = self.llm.with_structured_output(_Answer)

    def _retrieve(self, state: AWMState) -> dict[str, Any]:
        if not self.workflows:
            return {
                "retrieved_workflow": None,
                "history": [{"stage": "retrieve", "library_size": 0}],
            }
        try:
            docs = self._index.search(state["task"], k=1)
        except Exception:
            docs = []
        if not docs:
            return {"retrieved_workflow": None, "history": [{"stage": "retrieve", "n_hits": 0}]}
        # Find the workflow object
        match_label = docs[0].metadata.get("task_type", "")
        wf = next((w for w in self.workflows if w["task_type"] == match_label), None)
        return {
            "retrieved_workflow": wf,
            "history": [{"stage": "retrieve", "matched_workflow": match_label}],
        }

    def _answer(self, state: AWMState) -> dict[str, Any]:
        wf = state.get("retrieved_workflow")
        if wf:
            steps_block = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(wf["steps"]))
            prompt = (
                f"# Task\n{state['task']}\n\n"
                f"# Reusable workflow recipe (from your library, type='{wf['task_type']}')\n"
                f"{steps_block}\n\n"
                f"Follow this recipe to solve the task. Output JUST the final answer."
            )
        else:
            prompt = (
                f"# Task\n{state['task']}\n\n"
                "Solve from scratch (no prior workflow available). Output JUST the final answer."
            )
        try:
            ans = self._answerer.invoke(prompt)
            return {
                "answer": ans.answer.strip(),
                "history": [{"stage": "answer", "used_workflow": wf is not None}],
            }
        except Exception as e:
            return {"answer": f"(answer error: {e})", "history": [{"stage": "answer", "error": str(e)}]}

    def _extract_workflow(self, state: AWMState) -> dict[str, Any]:
        try:
            wf = self._extractor.invoke(
                f"# Task that was just solved\n{state['task']}\n\n"
                f"# Answer that worked\n{state['answer']}\n\n"
                "Extract a 3-6 step REUSABLE WORKFLOW (recipe) that another agent could "
                "follow to solve structurally-similar tasks. Generalise — strip "
                "task-specific entities; keep the strategy."
            )
            workflow = {"task_type": wf.task_type, "steps": list(wf.steps)}
            self.workflows.append(workflow)
            self._index.add([
                Document(page_content=workflow["task_type"] + " — " + "; ".join(workflow["steps"]),
                         metadata={"task_type": workflow["task_type"]})
            ])
            return {
                "extracted_workflow": workflow,
                "history": [{"stage": "extract", "task_type": workflow["task_type"], "library_size_after": len(self.workflows)}],
            }
        except Exception as e:
            return {"extracted_workflow": None, "history": [{"stage": "extract", "error": str(e)}]}

    def build(self) -> Any:
        g: StateGraph = StateGraph(AWMState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("answer", self._answer)
        g.add_node("extract", self._extract_workflow)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "answer")
        g.add_edge("answer", "extract")
        g.add_edge("extract", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        library_before = len(self.workflows)
        graph = self.build()
        final_state = graph.invoke({"task": task}, config={"recursion_limit": 25})
        retrieved = final_state.get("retrieved_workflow")
        extracted = final_state.get("extracted_workflow")
        return ArchitectureResult(
            output=final_state.get("answer", ""),
            state={
                "library_size": len(self.workflows),
                "library_grew": len(self.workflows) > library_before,
                "used_retrieved_workflow": retrieved is not None,
            },
            trace=final_state.get("history", []),
            metadata={
                "library_size_before": library_before,
                "library_size_after": len(self.workflows),
                "library_grew": len(self.workflows) > library_before,
                "used_retrieved_workflow": retrieved is not None,
                "retrieved_workflow_type": (retrieved or {}).get("task_type", ""),
                "extracted_workflow_type": (extracted or {}).get("task_type", ""),
                "extracted_steps": (extracted or {}).get("steps", []),
            },
        )
