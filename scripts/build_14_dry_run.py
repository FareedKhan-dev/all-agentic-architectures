"""Generate notebooks/14_dry_run.ipynb — propose → simulate → approve → execute pattern."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from notebook_builder import build_notebook, code, md  # noqa: E402


OUT_PATH = Path(__file__).parents[1] / "notebooks" / "14_dry_run.ipynb"


CELLS = [
    md(
        """# 14 · Dry-Run — propose → simulate → approve → execute (or skip)

> **TL;DR.** Before running an *irreversible* action, the agent (a) proposes a concrete command, (b) simulates the predicted effects without running, (c) routes through an approval check that combines an **LLM safety reviewer** with a **Python hard-cap on irreversibility**, and (d) either mock-executes or skips.
>
> **Reach for it when** the action has real-world side effects: shell commands, SQL writes, deployments, sending email, file modifications.
> **Avoid when** the action is trivially reversible or read-only — Tool Use (notebook 02) is cheaper.

| Property | Value |
|---|---|
| Origin | Software testing practice ("dry-run" flags in `rsync`, `apt`, `terraform plan`); agentic version: pre-execution safety check |
| Approval style | LLM reviewer + **deterministic Python hard-cap** on irreversibility |
| External tools needed? | No (execution is mocked in this demo) |
| Cost | ≈ 3 LLM calls (propose + dry-run + approve) per task |
| Composability | Reuses the same `with_structured_output` pattern as PEV (nb 06) and Mental Loop (nb 10) |

This is the *pre-execution* counterpart to **PEV** (notebook 06). PEV verifies *after* each step actually runs; Dry-Run verifies *before* the step runs. Both compose well — a production pipeline often uses Dry-Run for the action approval AND PEV for the per-result verification."""
    ),
    md(
        """## 2 · Architecture at a glance

```mermaid
flowchart LR
    A([task]) --> P[Propose<br/><sub>structured-output<br/>_ProposedAction</sub>]
    P --> D[Dry-Run<br/><sub>predict effects<br/>+ irreversibility 1-5</sub>]
    D --> H{Python hard-cap<br/>irreversibility ≥ threshold?}
    H -->|yes| Sk[Skip<br/><sub>BLOCKED by Python</sub>]
    H -->|no| AR[Approve<br/><sub>LLM safety reviewer</sub>]
    AR -->|approved| E[Execute<br/><sub>MOCK — record only</sub>]
    AR -->|rejected| Sk
    E --> Z([outcome])
    Sk --> Z

    style P fill:#e3f2fd,stroke:#1976d2
    style D fill:#fff3e0,stroke:#f57c00
    style H fill:#fce4ec,stroke:#c2185b
    style AR fill:#fce4ec,stroke:#c2185b
    style E fill:#e8f5e9,stroke:#388e3c
    style Sk fill:#ffebee,stroke:#c62828
```

**Two-layer gate:** the Python hard-cap on irreversibility runs *inside* the `approve` node and short-circuits the LLM check when the predicted irreversibility is high. This is the **deterministic-picker pattern** again — Python decides the easy / dangerous cases, LLM decides the soft cases."""
    ),
    md(
        """## 3 · Theory

### 3.1 · The "look before you leap" principle

Most software systems support a `--dry-run` flag for the same reason: actions that *modify state* are rarely undoable cleanly. Once an email is sent / a row is deleted / a deploy goes out, you can't pretend it didn't happen. Dry-Run pulls the *prediction* of the action's effects out into its own step where it can be inspected (or approved by a human) before the real thing runs.

### 3.2 · The three schemas

```python
class _ProposedAction(BaseModel):
    action_type: Literal["shell", "sql", "api", "file_modify", "email", "deploy"]
    command: str            # the EXACT command/query/payload
    purpose: str            # one sentence
    target_resources: list[str]

class _DryRunOutcome(BaseModel):
    predicted_effects: list[str]
    estimated_affected_count: int   # how many items
    irreversibility: int            # 1-5; 5 = catastrophic
    safety_concerns: list[str]

class _ApprovalDecision(BaseModel):
    approved: bool
    severity: Literal["low", "medium", "high", "block"]
    reason: str
```

The Pydantic schemas double as the *contract* between agent layers: the proposer can't emit a vague action; the dry-runner must commit to a number; the approver must commit to a boolean.

### 3.3 · The deterministic Python hard-cap

The Mental Loop pattern again: **don't trust the LLM with binary safety decisions when you can encode the rule in Python**.

```python
if predicted_irreversibility >= self.irreversibility_threshold:
    return {"approved": False, "decided_by": "python_hard_cap", ...}
# else, fall through to LLM safety reviewer
```

With `threshold=4` (default), any action the dry-runner labels as catastrophic-or-worse is BLOCKED before the LLM reviewer even sees it. The LLM only gets to weigh in on the gray-area cases.

This protects against three real failure modes:
1. **Sycophantic LLM reviewer.** A same-model reviewer is often too permissive — it agrees the action is fine because it "looks plausible".
2. **Adversarial prompt-injection.** A malicious prompt could nudge the LLM reviewer to approve a dangerous action; the Python hard-cap can't be talked out of it.
3. **Calibration drift.** LLM judgements vary across runs; deterministic rules don't.

### 3.4 · Mock execution

The `_execute` node in this demo **doesn't actually run anything** — it records what *would have* happened. This keeps the educational notebook side-effect-free. In production, you'd replace `_execute` with the real side-effect (shell exec, SQL, HTTP call) — but only after the approval gate has passed.

### 3.5 · Where Dry-Run sits

| Pattern | When the check runs | Best for |
|---|---|---|
| Tool Use (nb 02) | n/a — no check | Read-only / trivially reversible actions |
| **Dry-Run** *(this notebook)* | **Before execution** | Irreversible or expensive actions |
| PEV (nb 06) | After execution | Verify the outcome was as intended |
| Reflexive Metacognitive (nb 17) | Agent self-decides whether to act / escalate / refuse | High-stakes advisory (medical, legal, finance) |
| Mental Loop (nb 10) | Before execution, on K candidates | Choosing among options before committing |

Dry-Run + PEV is a common combination: Dry-Run gates *whether* to act, PEV verifies *whether the action achieved the goal*.

### 3.6 · What goes wrong (you'll see in § 9)

1. **Over-conservative LLM reviewer.** Same-model LLM reviewers often reject even routine safe actions because they "could" be misused. Mitigation: different model in the reviewer seat; or tighter approval prompt with concrete examples of "safe to approve".
2. **Under-conservative LLM reviewer.** Flip side: reviewer approves something the Python hard-cap would have blocked. The hard-cap exists exactly for this.
3. **Bad dry-run prediction.** If `estimated_affected_count` is wildly off (5 vs 5000), the safety check loses meaning. Mitigation: ground the dry-run in real schema / file inspection when possible.
4. **Mock-execute illusion.** Easy to forget the execute step is mocked. Production code MUST replace it with real side-effect AND keep the approval gate.
"""
    ),
    md("""## 4 · Setup"""),
    code(
        """from agentic_architectures import get_llm, enable_langsmith, settings
from agentic_architectures.architectures import DryRun
from agentic_architectures.ui import print_md, print_header, print_step

enable_langsmith()
print_header(f\"Provider: {settings.llm_provider}  ·  Model: {settings.llm_model}\")"""
    ),
    md(
        """## 5 · Library walkthrough

Source: [`src/agentic_architectures/architectures/dry_run.py`](../src/agentic_architectures/architectures/dry_run.py).

The architecture has four nodes plus a router:

| Node | Calls | Output |
|---|---|---|
| `_propose` | `with_structured_output(_ProposedAction)` | concrete command + purpose |
| `_dry_run` | `with_structured_output(_DryRunOutcome)` | predicted effects + irreversibility |
| `_approve` | **Python hard-cap first**, then `with_structured_output(_ApprovalDecision)` | approved? + decided_by |
| `_execute` (mock) | none | record what would have run |
| `_skip` | none | record skip reason |

The router `_route_after_approve` directs to `_execute` if approved, `_skip` otherwise."""
    ),
    code(
        """from agentic_architectures.architectures.dry_run import _ProposedAction, _DryRunOutcome, _ApprovalDecision
import json
print('--- ProposedAction schema ---')
print(json.dumps(_ProposedAction.model_json_schema(), indent=2)[:300] + '...')
print()
print('--- DryRunOutcome schema (note `irreversibility: 1-5`) ---')
print(json.dumps(_DryRunOutcome.model_json_schema(), indent=2)[:400] + '...')"""
    ),
    md(
        """## 6 · State"""
    ),
    md(
        """| Field | Set by |
|---|---|
| `task` | caller |
| `proposed_action` | `_propose` |
| `dry_run` | `_dry_run` |
| `approval` | `_approve` (includes `decided_by: "python_hard_cap" \\| "llm_reviewer"`) |
| `execution_outcome` | `_execute` or `_skip` |"""
    ),
    md(
        """## 7 · Build the graph

5 nodes: propose → dry_run → approve → (execute | skip). The split-after-approve is the only conditional edge."""
    ),
    code(
        """from IPython.display import Image, display
arch = DryRun(irreversibility_threshold=4)
graph = arch.build()
display(Image(graph.get_graph().draw_mermaid_png()))"""
    ),
    md(
        """## 8 · Live run — **three** tasks of escalating risk

To show all branches of the architecture, we run three tasks:

1. **Routine** — should approve and mock-execute.
2. **Moderate-risk** — depends on LLM reviewer judgement.
3. **Catastrophic** — Python hard-cap should fire (decided_by=`python_hard_cap`)."""
    ),
    code(
        """TASKS = [
    (\"routine\", \"Append a single line 'TODO: review pull requests' to a personal notes file at ~/notes.md\"),
    (\"moderate\", \"Send a transactional email to 50 customers notifying them of next week's planned maintenance window\"),
    (\"destructive\", \"DELETE all rows from the production users table where signup_date is older than 30 days\"),
]

results = []
for tag, task in TASKS:
    r = arch.run(task)
    results.append((tag, task, r))
    print_step(
        f\"[{tag.upper()}] approved={r.state['approved']}  ·  decided_by={r.state['decided_by']}  ·  irreversibility={r.metadata['irreversibility']}/5\",
        f\"command: {r.state['command'][:150]}\\n  outcome: {r.output[:300]}\"
    )
    print()

# Summary
from collections import Counter
gate_counter = Counter(r.state['decided_by'] for _, _, r in results)
print_header(\"Gate-decision distribution across the 3 tasks\")
print(f\"  by_python_hard_cap: {gate_counter.get('python_hard_cap', 0)}\")
print(f\"  by_llm_reviewer:    {gate_counter.get('llm_reviewer', 0)}\")"""
    ),
    md(
        """### 8.0 · What just happened, briefly

Three signals to inspect:

- **`decided_by` distribution** — `python_hard_cap` should fire on the destructive task (irreversibility 5/5). If it fires on the routine task, threshold is set too low.
- **`approved` distribution** — routine should be True, destructive should be False. Moderate-risk is the interesting case: same-model LLM reviewers tend to be conservative-default.
- **`irreversibility` per task** — should escalate roughly 1-2 → 3 → 5 across the 3 tasks. Inverse = LLM is mis-rating risk."""
    ),
    md("""### 8.1 · Detail of each proposed action + dry-run"""),
    code(
        """for tag, task, r in results:
    print_header(f\"=== {tag.upper()} ===\")
    print_step(\"task\", task)
    proposed = r.trace[0]
    dry = r.trace[1]
    approval = r.trace[2]
    print_step(\"proposed action_type\", proposed.get('action_type', '?'))
    print_step(\"proposed command\", proposed.get('command', '')[:200])
    print_step(\"dry-run predicted_effects\", \"; \".join(dry.get('predicted_effects', [])[:3]))
    print_step(\"dry-run irreversibility\", f\"{dry.get('irreversibility', '?')}/5  (affected~{dry.get('estimated_affected_count', '?')})\")
    print_step(\"approval verdict\", f\"approved={approval.get('approved', '?')}  severity={approval.get('severity', '?')}  decided_by={approval.get('decided_by', '?')}\")
    print_step(\"approval reason\", approval.get('reason', '')[:300])
    print()"""
    ),
    md(
        """## 9 · What we just observed

*(Automatically tailored from the actual captured run by `scripts/tailor_14_commentary.py`.)*"""
    ),
    md(
        """## 10 · Try different irreversibility thresholds

The hard-cap threshold is the single most important production knob. Higher threshold = more actions reach the LLM reviewer. Lower = more actions blocked unconditionally by Python."""
    ),
    code(
        """for threshold in [3, 5]:
    print_header(f\"--- irreversibility_threshold = {threshold} ---\")
    arch_t = DryRun(irreversibility_threshold=threshold)
    r = arch_t.run(\"Send a marketing email blast to all 50,000 subscribers about a 24-hour flash sale\")
    print(f\"  irreversibility predicted: {r.metadata['irreversibility']}/5\")
    print(f\"  approved: {r.state['approved']}  ·  decided_by: {r.state['decided_by']}\")
    print(f\"  reason: {r.trace[2].get('reason', '')[:200]}\")
    print()"""
    ),
    md(
        """## 11 · Failure modes, safety, extensions

### 11.1 · Where this breaks

| Failure | Mechanism | Mitigation |
|---|---|---|
| **Over-conservative LLM reviewer** | Same-model reviewer rejects safe routine actions | Different model in reviewer seat; or tighter prompt with examples of "safe to approve" |
| **Under-conservative LLM reviewer** | Reviewer approves a dangerous action | **Python hard-cap is the safety net** — design the threshold conservatively |
| **Bad irreversibility estimate** | Dry-runner labels routine as 5/5 or destructive as 1/5 | Ground the dry-run in real artifacts (schema inspection, file listing); rubric examples in the schema description |
| **Mock-execute illusion** | Forget the demo's `_execute` is mocked | In production code, replace `_execute` with the real side-effect AND keep the approval gate untouched |
| **Adversarial prompt injection** | Malicious prompt nudges LLM reviewer to approve | Python hard-cap can't be talked out of it; layered defense |

### 11.2 · Production safety

- **Don't remove the hard-cap.** It's the deterministic backstop for sycophantic / poisoned LLM reviewers.
- **Log every (task, command, decided_by, approved) tuple.** Audit trail.
- **Human-in-the-loop for high severity.** The library has `require_human_approval_above_severity` parameter for this; the demo auto-mocks for educational purposes only.

### 11.3 · Three extensions

1. **Real execution + rollback.** Replace mock `_execute` with the real side-effect, *and* capture a rollback handle (DB transaction, git commit hash, file backup). If post-execution PEV (nb 06) detects failure, roll back.
2. **Multi-level approval.** Severity `medium` requires LLM-reviewer approval; severity `high` requires human; severity `block` cannot be overridden.
3. **Sandboxed dry-run.** Instead of LLM-predicting effects, actually run the action in a sandbox (Docker container, DB transaction-with-rollback) and observe real effects.

### 11.4 · What to read next

- [**06 · PEV**](./06_pev.ipynb) — verify the outcome *after* execution.
- [**10 · Mental Loop**](./10_mental_loop.ipynb) — pick the best action *before* dry-running.
- [**17 · Reflexive Metacognitive**](./17_reflexive_metacognitive.ipynb) — agent decides when to act vs escalate.

### 11.5 · References

1. The Unix `rsync --dry-run` flag and similar — classical software-testing tradition.
2. *Terraform plan* — same pattern in infrastructure-as-code.
3. Anthropic *Constitutional AI* — LLM safety reviewers with explicit rule sets.
"""
    ),
]


def main() -> None:
    out = build_notebook(CELLS, OUT_PATH)
    print(f"wrote: {out}  ({sum(len(c[1]) for c in CELLS)} chars across {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
