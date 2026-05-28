"""Generate notebooks/17_reflexive_metacognitive.ipynb — self-aware routing across answer/tool/partial/escalate."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from notebook_builder import build_notebook, code, md  # noqa: E402

OUT_PATH = Path(__file__).parents[1] / "notebooks" / "17_reflexive_metacognitive.ipynb"


CELLS = [
    md(
        """# 17 · Reflexive Metacognitive — agent reasons about its OWN capabilities

> **TL;DR.** Before answering, the agent applies a **self-model** (a written description of what it can / can't do) to the incoming question and routes to one of four destinations: **answer** / **use_tool** / **partial** / **escalate**. A Python deterministic override forces `escalate` if the LLM admits `requires_credentials=True` OR `capability_match <= 2`, regardless of the LLM's chosen route.
>
> **Reach for it when** the agent serves a domain where giving a confidently-wrong answer is much worse than declining (medical, legal, finance, safety-critical).
> **Avoid when** any answer is better than no answer (chat, casual research).

| Property | Value |
|---|---|
| Origin | Metacognition (Flavell 1979); LLM safety practice; Constitutional AI (Anthropic 2022) |
| Routes | answer / use_tool / partial / escalate |
| Override layer | **Python deterministic** — force escalate when LLM admits credentials gap or low capability |
| Cost | 1 routing LLM call + 0-1 follow-on LLM call depending on route |
| Composability | `use_tool` route in production hands off to ToolUse (nb 02); `escalate` hands off to human |

This is the SAFETY-FIRST end of the architecture spectrum. The whole point is that "I don't know" is a first-class answer."""
    ),
    md(
        """## 2 · Architecture at a glance

```mermaid
flowchart LR
    A([task]) --> C[Classify<br/><sub>self-model + capability_match,<br/>requires_credentials, requires_external_lookup</sub>]
    C --> O{Python override?<br/>credentials=True<br/>OR capability ≤ 2}
    O -->|yes| ES[Escalate<br/><sub>FORCED by Python</sub>]
    O -->|no, follow LLM route| R{LLM route}
    R -->|answer| AN[Direct answer]
    R -->|use_tool| UT[Use tool<br/><sub>→ ToolUse in production</sub>]
    R -->|partial| P[Partial answer<br/><sub>+ explicit gaps</sub>]
    R -->|escalate| ES
    AN --> Z([response])
    UT --> Z
    P --> Z
    ES --> Z

    style C fill:#fff3e0,stroke:#f57c00
    style O fill:#fce4ec,stroke:#c2185b
    style ES fill:#ffebee,stroke:#c62828
```

**Two safety layers:** (1) Python deterministic override that *cannot* be overridden by an LLM; (2) LLM's own metacognitive route. The Python layer catches the cases where the LLM correctly admits a credentials gap but then chose the wrong route anyway."""
    ),
    md(
        """## 3 · Theory

### 3.1 · Why self-models matter

Most agentic patterns are *outcome-oriented*: they try to solve the task. Reflexive Metacognitive is *capability-oriented*: it first asks *"should I even try?"*

For high-stakes domains (medical, legal, finance), this question is more important than the answer. An ungrounded medical answer can harm a user; an honest "I'm not qualified, see a doctor" cannot. The architecture makes that escalation a *first-class action*, not a fallback.

### 3.2 · The deterministic override layer

The architecture has TWO routing layers:

```python
# Layer 1: LLM classifies
decision = self._decider.invoke(prompt)  # route = answer / use_tool / partial / escalate

# Layer 2: Python overrides (deterministic, can't be talked out of)
if decision.requires_credentials:
    decision.route = "escalate"  # always, regardless of LLM choice
elif decision.capability_match <= 2:
    decision.route = "escalate"  # always
```

The Python layer exists because a sycophantic / prompt-injected LLM might admit `requires_credentials=True` but still choose `route="answer"`. Python forces the correct downstream action when the LLM's *features* (booleans, integers) imply escalation even if the LLM's *route* doesn't.

This is the **deterministic-picker pattern** applied to a categorical decision — same as Mental Loop's `scoring_fn` (nb 10), Ensemble's `majority_vote` (nb 13), Dry-Run's irreversibility hard-cap (nb 14), RLHF's composite score (nb 15).

### 3.3 · The self-model as a configurable string

```python
arch = ReflexiveMetacognitive(self_model="You are a financial advisor specialised in retirement planning. ...")
```

The self-model is just a string the LLM reads in the classification prompt. To deploy in a new domain, change the string. Production deployments often layer multiple self-models — a base "what I am" + a per-task "what's allowed in this context".

### 3.4 · Where this sits

| Pattern | When it acts safety-first | Mechanism |
|---|---|---|
| Dry-Run (nb 14) | Before SIDE-EFFECTS | predict effects + Python hard-cap |
| **Reflexive Metacognitive** *(this notebook)* | **Before ANSWERS** | self-model + Python override |
| PEV (nb 06) | After each step ran | verify outcome |
| Constitutional AI (nb 32) | During generation | self-critique against rules |
| Multi-Agent (nb 05) + this | Delegate hard cases | escalation route → human reviewer |

In a production high-stakes pipeline, you'd often stack: Reflexive Metacognitive routes the easy stuff, escalates the hard, the human-in-the-loop or domain specialist handles the rest.

### 3.5 · What goes wrong (you'll see in § 9)

1. **Self-model too vague** — LLM bins everything as "moderate". Tighten with concrete examples.
2. **LLM admits gap but routes 'answer'** — exactly what the Python override exists for. Watch §9 for `override_applied=True` cases.
3. **Over-escalation** — agent refuses easy questions. Means the self-model says "WEAK: everything" too aggressively.
4. **`use_tool` route ungrounded** — agent says "use_tool" but no tool actually fires. In production, this route must hand off to a real Tool Use architecture (notebook 02)."""
    ),
    md("""## 4 · Setup"""),
    code(
        """from agentic_architectures import get_llm, enable_langsmith, settings
from agentic_architectures.architectures import ReflexiveMetacognitive
from agentic_architectures.ui import print_md, print_header, print_step

enable_langsmith()
print_header(f\"Provider: {settings.llm_provider}  ·  Model: {settings.llm_model}\")"""
    ),
    md(
        """## 5 · Library walkthrough

Source: [`src/agentic_architectures/architectures/reflexive_metacognitive.py`](../src/agentic_architectures/architectures/reflexive_metacognitive.py).

Two key pieces:

1. **`_MetaDecision`** schema — captures `capability_match` (1-5), `requires_external_lookup`, `requires_credentials`, `route` literal, `reason`.
2. **`_classify`** — calls the decider LLM, then **applies the Python override** before returning. The override is encoded in the same `decision` dict so the rest of the graph and the trace see the OVERRIDDEN route."""
    ),
    code(
        """from agentic_architectures.architectures.reflexive_metacognitive import _MetaDecision, DEFAULT_SELF_MODEL
import json
print('--- Decision schema ---')
print(json.dumps(_MetaDecision.model_json_schema(), indent=2)[:500] + '...')
print()
print('--- Default self-model ---')
print(DEFAULT_SELF_MODEL)"""
    ),
    md("""## 6 · State"""),
    md(
        """| Field | Set by |
|---|---|
| `task` | caller |
| `decision` | `_classify` (includes `route`, `llm_route`, `override`, `capability_match`, …) |
| `deterministic_override` | `_classify` (True iff Python forced a route change) |
| `final_answer` | one of `_answer`, `_use_tool`, `_partial`, `_escalate` |"""
    ),
    md("""## 7 · Build the graph"""),
    code(
        """from IPython.display import Image, display
arch = ReflexiveMetacognitive()
graph = arch.build()
display(Image(graph.get_graph().draw_mermaid_png()))"""
    ),
    md(
        """## 8 · Live run — 4 tasks across all four routes

The 4 tasks below are designed to hit each route (`answer`, `use_tool`, `partial`, `escalate`):

| # | Task tag | Should route to |
|---|---|---|
| 1 | general    | `answer` (within strong capability) |
| 2 | live       | `use_tool` (live data) |
| 3 | medical    | `escalate` (credentials) |
| 4 | legal      | `escalate` (credentials) |"""
    ),
    code(
        """TASKS = [
    (\"general\", \"What is the time complexity of a binary search on a sorted array?\"),
    (\"live\",    \"What is the current price of one Bitcoin in USD right now?\"),
    (\"medical\", \"I've had a persistent dry cough for 3 weeks with mild chest tightness. What is the most likely cause and should I see a doctor?\"),
    (\"legal\",   \"Should I sign a non-compete clause my California employer is asking me to sign?\"),
]

results = []
for tag, t in TASKS:
    r = arch.run(t)
    results.append((tag, t, r))
    print_step(
        f\"[{tag.upper()}]\",
        f\"  final_route: {r.metadata['route']}  (LLM said: {r.metadata['llm_route']}; override={r.metadata['override_applied']})\\n\"
        f\"  capability_match={r.metadata['capability_match']}, requires_lookup={r.metadata['requires_external_lookup']}, requires_credentials={r.metadata['requires_credentials']}\\n\"
        f\"  answer: {r.output[:300]}\"
    )
    print()

# Route distribution
from collections import Counter
route_counter = Counter(r.metadata['route'] for _, _, r in results)
override_counter = sum(1 for _, _, r in results if r.metadata['override_applied'])
print_header(\"Route distribution\")
for route, n in route_counter.most_common():
    print(f\"  {route}: {n}\")
print(f\"  Python-overrides applied: {override_counter}/{len(results)}\")"""
    ),
    md(
        """### 8.0 · What just happened, briefly

Three signals to inspect:

- **Route diversity** — 4 deliberately different tasks should hit 2-4 distinct routes. All same route = over-conservative or over-confident.
- **`override_applied=True` count** — should be > 0 for the credentialed tasks (medical / legal) because Python escalates regardless of LLM's chosen route.
- **`requires_credentials` consistency** — medical / legal questions should set this to `True`. If they don't, the LLM is missing the credentials gap (and Python relies on this signal to override)."""
    ),
    md(
        """## 9 · What we just observed

*(Automatically tailored from the actual captured run by `scripts/tailor_17_commentary.py`.)*"""
    ),
    md(
        """## 10 · Custom self-model — financial advisor scoped to retirement planning"""
    ),
    code(
        """fin_arch = ReflexiveMetacognitive(
    self_model=(
        \"You are a financial assistant SCOPED to retirement-planning topics. \"
        \"STRONG: tax-advantaged account types (401k, IRA, Roth), withdrawal strategies, \"
        \"asset-allocation principles. WEAK: anything about specific tickers, market \"
        \"timing, current rates. ABSOLUTELY NOT: tax advice for an individual's situation, \"
        \"or any advice requiring fiduciary credentials.\"
    ),
)
for q in [
    \"What is a Roth IRA?\",
    \"Should I move my 401k into VOO right now?\",
    \"Based on my financial situation, what's the right asset allocation for me?\",
]:
    r = fin_arch.run(q)
    print(f\"Q: {q}\")
    print(f\"  → route={r.metadata['route']} (llm={r.metadata['llm_route']}, override={r.metadata['override_applied']})\")
    print(f\"  answer: {r.output[:200]}\")
    print()"""
    ),
    md(
        """## 11 · Failure modes, safety, extensions

### 11.1 · Where this breaks

| Failure | Mechanism | Mitigation |
|---|---|---|
| **Self-model too vague** | "moderate" routing for everything | Concrete examples of STRONG / MODERATE / WEAK |
| **LLM admits gap but routes 'answer'** | LLM correctly fills `requires_credentials=True` but its `route` choice is 'answer' anyway | **Python override forces escalate** — this is exactly the case the override exists for |
| **Over-escalation** | Agent refuses easy questions | Self-model says WEAK on too many things; tune the threshold or self-model |
| **'use_tool' route is empty** | Architecture says use_tool but no actual tool fires | In production, route hands off to ToolUse (nb 02); this notebook stops short |
| **Adversarial prompt** | Malicious user tries to make the agent answer a medical question | Python override resists prompt injection at the route level |

### 11.2 · Production safety

- **Always include the Python override.** It's the deterministic backstop for prompt-injected or sycophantic LLM routing.
- **Treat 'use_tool' as an integration point.** Hand off to a real ToolUse architecture (nb 02) — don't let the route end with no action.
- **Treat 'escalate' as an integration point.** Route to a human, ticket system, or higher-privilege model.

### 11.3 · Three extensions

1. **Domain-stacked self-models** — base + per-task self-models for multi-tenant deployments.
2. **Quantitative capability score** — replace the 1-5 integer with a richer (probability of being correct × cost-of-being-wrong) calculation.
3. **Closed-loop:** when the human resolves an escalation, feed the resolution back into the archive (Reflexion pattern, nb 18).

### 11.4 · What to read next

- [**14 · Dry-Run**](./14_dry_run.ipynb) — safety pattern for side-effects (companion).
- [**32 · Constitutional AI**](./32_constitutional.ipynb) — self-critique against written rules (related).
- [**11 · Meta-Controller**](./11_meta_controller.ipynb) — routes between *architectures*, not action classes.

### 11.5 · References

1. Flavell, J. H. *Metacognition and cognitive monitoring.* American Psychologist 34, 1979.
2. Anthropic. *Constitutional AI: Harmlessness from AI Feedback.* 2022. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
"""
    ),
]


def main() -> None:
    out = build_notebook(CELLS, OUT_PATH)
    print(f"wrote: {out}  ({sum(len(c[1]) for c in CELLS)} chars across {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
