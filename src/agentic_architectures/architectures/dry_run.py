"""Dry-Run — propose action → simulate effects → approve → execute (or skip).

A safety pattern for **irreversible** or **expensive** operations: shell commands,
SQL writes, deployments, sending emails, file modifications. Before running the
real action, the agent:
  1. **Proposes** a concrete action.
  2. **Dry-runs** the action — predicts effects WITHOUT executing.
  3. Routes through an **approval check** (LLM safety reviewer here; in
     production, often human-in-the-loop).
  4. Either **executes** (mocked in this demo for safety) or **skips**.

Compared to **Mental Loop** (notebook 10): Mental Loop generates K candidate
actions and picks the best; Dry-Run has *one* specific action in mind and
checks it before doing it. Dry-Run is appropriate when the candidate has
already been chosen and the question is "is this safe?".

Compared to **PEV** (notebook 06): PEV verifies AFTER each step actually
runs; Dry-Run verifies BEFORE the step runs. Use Dry-Run for pre-execution
safety, PEV for post-execution correctness.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class _ProposedAction(BaseModel):
    """One concrete proposed action."""

    action_type: Literal["shell", "sql", "api", "file_modify", "email", "deploy"] = Field(
        description="The category of action being proposed."
    )
    command: str = Field(
        description=(
            "The exact concrete action — a shell command, SQL statement, "
            "HTTP request body, file edit diff, etc. Be SPECIFIC, not abstract."
        )
    )
    purpose: str = Field(description="One sentence explaining WHY this action is being proposed.")
    target_resources: list[str] = Field(
        default_factory=list,
        description="Specific resources (files, tables, endpoints) the action will touch.",
    )


class _DryRunOutcome(BaseModel):
    """Predicted effects of running the proposed action — without actually running it."""

    predicted_effects: list[str] = Field(
        description="3-6 concrete effects that would happen if this action runs. Use specifics."
    )
    estimated_affected_count: int = Field(
        ge=0,
        description="Estimated number of items affected (rows / files / recipients / etc.). Use a single point estimate.",
    )
    irreversibility: int = Field(
        ge=1, le=5,
        description=(
            "How irreversible is this action? 1 = trivially undone, 5 = catastrophic / "
            "data lost forever. Use the schema rubric: deleted files = 4-5, "
            "config change = 2-3, read-only API call = 1."
        ),
    )
    safety_concerns: list[str] = Field(
        default_factory=list,
        description="Concrete safety issues a reviewer should consider before approving.",
    )


class _ApprovalDecision(BaseModel):
    """The safety reviewer's verdict on whether to execute."""

    approved: bool = Field(description="True iff the action is safe to execute given the dry-run prediction.")
    severity: Literal["low", "medium", "high", "block"] = Field(
        description=(
            "Risk level. 'low' = routine; 'medium' = proceed with logging; "
            "'high' = proceed but flag for human review; 'block' = do not execute."
        )
    )
    reason: str = Field(description="One sentence explaining the verdict.")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class DryRunState(TypedDict, total=False):
    task: str
    proposed_action: dict[str, Any]
    dry_run: dict[str, Any]
    approval: dict[str, Any]
    execution_outcome: str
    irreversibility_threshold: int  # configurable Python-side hard cap


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class DryRun(Architecture):
    """Propose → simulate → approve → execute-or-skip safety pattern."""

    name = "dry_run"
    description = (
        "Pre-execution safety pattern: agent proposes a concrete action, predicts "
        "its effects via a dry-run, an approval step gates execution. The execute "
        "step is mocked in this demo (prints would-have-happened) to keep "
        "side-effect-free for educational use."
    )
    reference = "https://en.wikipedia.org/wiki/Dry_run_(testing)"

    def __init__(
        self,
        irreversibility_threshold: int = 4,
        require_human_approval_above_severity: Literal["low", "medium", "high"] | None = "high",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            irreversibility_threshold: Python hard-cap. Any predicted irreversibility
                >= this value BLOCKS execution regardless of the LLM approval. Default 4.
            require_human_approval_above_severity: Severity levels at or above this
                require human-in-the-loop in production (in this demo, marked but auto-mocked).
        """
        super().__init__(**kwargs)
        self.irreversibility_threshold = irreversibility_threshold
        self.require_human_approval_above_severity = require_human_approval_above_severity
        self._proposer = self.llm.with_structured_output(_ProposedAction)
        self._dry_runner = self.llm.with_structured_output(_DryRunOutcome)
        self._approver = self.llm.with_structured_output(_ApprovalDecision)

    # ------------------------------------------------------------------ #
    #  Nodes                                                              #
    # ------------------------------------------------------------------ #

    def _propose(self, state: DryRunState) -> dict[str, Any]:
        prompt = (
            "You must propose a single CONCRETE action that accomplishes the task. "
            "Specify the exact command / query / payload — not a vague description.\n\n"
            f"## Task\n{state['task']}"
        )
        action = self._proposer.invoke(prompt)
        return {"proposed_action": action.model_dump()}

    def _dry_run(self, state: DryRunState) -> dict[str, Any]:
        action = state["proposed_action"]
        prompt = (
            "Mentally simulate this action WITHOUT actually running it. Predict "
            "what would happen, how many items are affected, and how reversible "
            "the change is.\n\n"
            f"## Action type\n{action['action_type']}\n\n"
            f"## Command\n{action['command']}\n\n"
            f"## Purpose\n{action['purpose']}\n\n"
            f"## Target resources\n{action.get('target_resources', [])}\n\n"
            "Be specific and concrete in predicted_effects — list actual files, "
            "rows, or downstream consequences. Use the schema's irreversibility "
            "rubric strictly."
        )
        outcome = self._dry_runner.invoke(prompt)
        return {"dry_run": outcome.model_dump()}

    def _approve(self, state: DryRunState) -> dict[str, Any]:
        action = state["proposed_action"]
        dry = state["dry_run"]

        # DETERMINISTIC pre-check: if irreversibility >= threshold, block unconditionally.
        # This is the deterministic-picker pattern again — Python decides on the hard
        # case, LLM only decides the soft cases.
        if dry["irreversibility"] >= self.irreversibility_threshold:
            return {
                "approval": {
                    "approved": False,
                    "severity": "block",
                    "reason": (
                        f"Python hard-cap: predicted irreversibility "
                        f"{dry['irreversibility']}/5 ≥ threshold "
                        f"{self.irreversibility_threshold}. Action blocked "
                        "regardless of LLM approval."
                    ),
                    "decided_by": "python_hard_cap",
                }
            }

        # Soft case: ask the LLM reviewer.
        prompt = (
            "You are a SAFETY REVIEWER. Decide whether to approve the proposed action "
            "given the dry-run prediction.\n\n"
            f"## Proposed action\n"
            f"  type: {action['action_type']}\n"
            f"  command: {action['command']}\n"
            f"  purpose: {action['purpose']}\n\n"
            f"## Dry-run prediction\n"
            f"  predicted effects: {dry['predicted_effects']}\n"
            f"  estimated affected count: {dry['estimated_affected_count']}\n"
            f"  irreversibility: {dry['irreversibility']}/5\n"
            f"  safety concerns: {dry['safety_concerns']}\n\n"
            "Approve or reject. Use severity 'block' if you see any safety concern that "
            "isn't acceptable risk. Be conservative."
        )
        verdict = self._approver.invoke(prompt)
        result = verdict.model_dump()
        result["decided_by"] = "llm_reviewer"
        return {"approval": result}

    def _execute(self, state: DryRunState) -> dict[str, Any]:
        action = state["proposed_action"]
        approval = state["approval"]
        # In a real system, this would invoke the real side-effect (shell, SQL,
        # API, etc.). In this educational demo, we ONLY record what would have
        # happened. Real execution is intentionally out of scope.
        outcome = (
            f"[MOCK EXECUTION] Would have run: `{action['command']}`\n"
            f"  type: {action['action_type']}\n"
            f"  approved by: {approval.get('decided_by', '?')} "
            f"(severity={approval.get('severity', '?')})\n"
            f"  predicted to affect {state['dry_run']['estimated_affected_count']} item(s)\n"
            f"  irreversibility: {state['dry_run']['irreversibility']}/5"
        )
        return {"execution_outcome": outcome}

    def _skip(self, state: DryRunState) -> dict[str, Any]:
        approval = state.get("approval", {})
        return {
            "execution_outcome": (
                f"[SKIPPED — not executed] "
                f"reason: {approval.get('reason', 'no approval')} "
                f"(severity={approval.get('severity', 'block')}, "
                f"decided_by={approval.get('decided_by', '?')})"
            )
        }

    # ------------------------------------------------------------------ #
    #  Router                                                             #
    # ------------------------------------------------------------------ #

    def _route_after_approve(self, state: DryRunState) -> str:
        return "execute" if state["approval"].get("approved") else "skip"

    # ------------------------------------------------------------------ #
    #  Build + run                                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> Any:
        g: StateGraph = StateGraph(DryRunState)
        g.add_node("propose", self._propose)
        g.add_node("dry_run", self._dry_run)
        g.add_node("approve", self._approve)
        g.add_node("execute", self._execute)
        g.add_node("skip", self._skip)
        g.add_edge(START, "propose")
        g.add_edge("propose", "dry_run")
        g.add_edge("dry_run", "approve")
        g.add_conditional_edges(
            "approve", self._route_after_approve,
            {"execute": "execute", "skip": "skip"},
        )
        g.add_edge("execute", END)
        g.add_edge("skip", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke({
            "task": task,
            "irreversibility_threshold": self.irreversibility_threshold,
        })

        proposed = final_state.get("proposed_action", {})
        dry = final_state.get("dry_run", {})
        approval = final_state.get("approval", {})

        return ArchitectureResult(
            output=final_state.get("execution_outcome", ""),
            state={
                "action_type": proposed.get("action_type", ""),
                "command": proposed.get("command", ""),
                "approved": approval.get("approved", False),
                "decided_by": approval.get("decided_by", "?"),
            },
            trace=[
                {"type": "proposed_action", **proposed},
                {"type": "dry_run", **dry},
                {"type": "approval", **approval},
            ],
            metadata={
                "action_type": proposed.get("action_type", ""),
                "irreversibility": dry.get("irreversibility", 0),
                "estimated_affected": dry.get("estimated_affected_count", 0),
                "approved": approval.get("approved", False),
                "severity": approval.get("severity", "?"),
                "decided_by": approval.get("decided_by", "?"),
                "irreversibility_threshold": self.irreversibility_threshold,
            },
        )
