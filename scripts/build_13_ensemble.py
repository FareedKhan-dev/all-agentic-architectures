"""Generate notebooks/13_ensemble.ipynb — multi-perspective ensemble."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from notebook_builder import build_notebook, code, md  # noqa: E402


OUT_PATH = Path(__file__).parents[1] / "notebooks" / "13_ensemble.ipynb"


CELLS = [
    md(
        """# 13 · Ensemble — N parallel voters + aggregator

> **TL;DR.** Run K voter agents (each with a different perspective system prompt) against the **same** task. An aggregator LLM synthesises their answers into one balanced response. Compared to Multi-Agent (notebook 05), where specialists divide labour, Ensemble has all voters answer the **full** task — value comes from *diverse opinions*, not divided work.
>
> **Reach for it when** the question is contested / has multiple legitimate framings (forecasts, judgement calls, fact-checking).
> **Avoid when** the answer is objectively correct or wrong — Ensemble adds noise, not signal.

| Property | Value |
|---|---|
| Origin | "Wisdom of crowds" (Surowiecki 2004); modern LLM ensembles in practice |
| Voter selection | Different *perspectives* (analytical / skeptical / pragmatic by default) |
| Aggregation modes | `llm_synth` (default), `highest_confidence`, `majority_vote` (extension) |
| External tools needed? | No |
| Cost | K voter calls + 1 aggregator call |
| Composability | Each voter is just an LLM with a system prompt — could be replaced with full architectures |

This pattern is *structurally similar* to Blackboard (notebook 07) but the key difference is **everyone answers the full question at once**, no turn-taking, no bidding. Cheaper than Blackboard for the same number of opinions."""
    ),
    md(
        """## 2 · Architecture at a glance

```mermaid
flowchart TB
    A([task]) --> V[Vote round<br/><sub>each voter independently<br/>answers the SAME task</sub>]
    V --> O1[analytical opinion]
    V --> O2[skeptical opinion]
    V --> O3[pragmatic opinion]
    O1 --> Ag[Aggregate<br/><sub>llm_synth: balanced synthesis<br/>OR highest_confidence pick</sub>]
    O2 --> Ag
    O3 --> Ag
    Ag --> Z([final answer])

    style V fill:#e3f2fd,stroke:#1976d2
    style Ag fill:#fff3e0,stroke:#f57c00
```

**Fan-out then fan-in.** The voters operate in parallel (we run them sequentially in the demo for clarity — a real production path uses LangGraph parallel branches). The aggregator sees ALL voter opinions and produces one balanced synthesis."""
    ),
    md(
        """## 3 · Theory

### 3.1 · Why use multiple voters at all?

A single LLM has *systematic biases* — recency bias toward training-data viewpoints, sycophancy toward the question's framing, mode collapse toward "safe" hedged answers. Putting the *same* question to one LLM 3 times with the same prompt mostly produces the same answer (LLMs are mostly deterministic at temperature 0).

Ensemble breaks the bias by **varying the prompt perspective**. Each voter is *forced* to look at the question through a different lens:
- The Analytical voter focuses on data / evidence / mechanism.
- The Skeptical voter looks for what could go wrong / what's missing.
- The Pragmatic voter focuses on what actually ships / works in practice.

These three perspectives produce *substantively different* answers — not because the model knows different facts, but because each prompt activates a different reasoning pattern.

### 3.2 · The structured-output `_VoterOpinion`

```python
class _VoterOpinion(BaseModel):
    bottom_line: str                    # 1-2 sentence direct answer
    key_points: list[str]               # 2-4 supporting points
    confidence: int = Field(ge=1, le=5) # self-reported confidence
```

Three crucial design choices:

1. **Bottom-line first** — forces each voter to commit to a directional answer (yes / no / depends) before listing supporting points. Without this, voters hedge endlessly.
2. **Key points are a *list*** — not free prose. Easier to compare across voters in the aggregator.
3. **Self-reported confidence** — drives the `highest_confidence` aggregator mode. Note: LLM-self-reported confidence is *noisy* (see § 11.1).

### 3.3 · Aggregation modes

| Mode | What it does | When to use |
|---|---|---|
| **`llm_synth`** (default) | Aggregator LLM weaves all K opinions into one balanced response | Long-form questions, contested topics, when you want nuance preserved |
| **`highest_confidence`** | Pick the voter with highest self-reported confidence | Short factual answers; deferring to the most assured voice |
| **`majority_vote`** (extension) | Tally categorical answers, return mode | Yes/no, A/B/C, classification |

The default `llm_synth` aggregator's prompt explicitly asks it to:
1. State the most-likely answer.
2. Identify points of *agreement* across voters.
3. Identify points of *genuine disagreement* (not paper over them).
4. End with a hedged recommendation.

That structure forces the aggregator to preserve the multi-perspective nature, not flatten it.

### 3.4 · Where Ensemble sits

| Pattern | Voters on same task? | Coordination | Use when |
|---|---|---|---|
| ReAct (nb 03) | n/a | n/a | single focused query |
| Multi-Agent (nb 05) | no — different sub-tasks | central supervisor | task spans domains |
| Blackboard (nb 07) | no — turn-taking, dynamic | distributed bidding | exploratory |
| **Ensemble** *(this notebook)* | **yes — full task each** | **fan-out / fan-in** | contested / forecasting / fact-checking |
| Self-Consistency (nb 21) | yes — same prompt, N samples | majority vote | classification / arithmetic where vote tally helps |
| Multi-Agent Debate (nb 28) | yes — adversarial back-and-forth | converge via critique | controversial topics where iterative refinement helps |

### 3.5 · What goes wrong (you'll see in § 9)

1. **Flat confidence scores** — same Llama-as-Scorer pathology as ToT/Mental Loop. The bottom-line answers differ but confidence values are similar. Watch for it in § 9.
2. **Aggregator washout** — synthesis blends opinions so much that the minority view disappears. The aggregator prompt explicitly forbids this but it still happens.
3. **Hidden conformity** — all 3 voters arrive at the same answer despite different prompts. Either the question wasn't really contested, or the perspective prompts weren't different enough.
4. **Adversarial perspective wins** — Skeptical voter is the loudest because "what could go wrong" is easy to generate. Aggregator may over-weight skepticism.
"""
    ),
    md("""## 4 · Setup"""),
    code(
        """from agentic_architectures import get_llm, enable_langsmith, settings
from agentic_architectures.architectures import Ensemble
from agentic_architectures.architectures.ensemble import DEFAULT_VOTERS
from agentic_architectures.ui import print_md, print_header, print_step

enable_langsmith()
print_header(f\"Provider: {settings.llm_provider}  ·  Model: {settings.llm_model}\")
print_md(f\"Default voters: **{', '.join(DEFAULT_VOTERS.keys())}**\")"""
    ),
    md(
        """## 5 · Library walkthrough

Source: [`src/agentic_architectures/architectures/ensemble.py`](../src/agentic_architectures/architectures/ensemble.py).

Two nodes:

1. **`_vote`** — runs each voter LLM with their unique perspective prompt; collects `_VoterOpinion` structured outputs into a list.
2. **`_aggregate`** — branches on `aggregator_mode`:
   - `llm_synth`: LLM synthesises a balanced response.
   - `highest_confidence`: returns the most-confident voter's opinion verbatim.

The voters are run sequentially for trace clarity; a production path can fan them out with `langgraph.graph.parallel` for an N× latency win."""
    ),
    code(
        """from agentic_architectures.architectures.ensemble import _VoterOpinion, DEFAULT_VOTERS
import json
print('--- VoterOpinion schema ---')
print(json.dumps(_VoterOpinion.model_json_schema(), indent=2)[:400] + '...')
print()
print('--- Voter perspectives ---')
for name, prompt in DEFAULT_VOTERS.items():
    print(f'\\n  {name}:')
    print(f'  {prompt}')"""
    ),
    md(
        """## 6 · State

| Field | Type | Set by |
|---|---|---|
| `task` | `str` | caller |
| `voter_opinions` | `list[dict]` (one per voter) | `_vote` (appended) |
| `aggregated_answer` | `str` | `_aggregate` |
| `aggregator_mode` | `Literal[...]` | caller / default |"""
    ),
    md("""## 7 · Build the graph"""),
    code(
        """from IPython.display import Image, display
arch = Ensemble()
graph = arch.build()
display(Image(graph.get_graph().draw_mermaid_png()))"""
    ),
    md(
        """## 8 · Live run — contested forecasting question

Concrete task: a *contested* forward-looking question where reasonable people genuinely disagree. The point: see if our 3 perspectives produce 3 different answers, and how the aggregator synthesises them."""
    ),
    code(
        """TASK = (
    \"Will electric vehicles account for over 50% of new car sales globally by 2030? \"
    \"Answer YES or NO with a 2-3 sentence rationale.\"
)

result = arch.run(TASK)

print_header(\"Aggregated answer (llm_synth)\")
print_md(result.output)
print()
print(f\"VOTERS: {result.state['voters_used']}\")
print(f\"CONFIDENCES: {result.metadata['confidences']}\")
print(f\"CONFIDENCE_SPREAD: {result.metadata['confidence_spread']}\")"""
    ),
    md(
        """### 8.0 · What just happened, briefly

Three things to look at:

- **Voter disagreement** — read each voter's bottom_line in §8.1. If all 3 agree, the question wasn't actually contested or the perspective prompts didn't activate distinct framings.
- **Confidence spread** — if everyone is 4/5 (flat), the LLM-as-Scorer pathology again (see Mental Loop nb 10 §9). Bottom-line *content* discrimination matters more than the confidence number on a contested question.
- **Aggregator quality** — does the synthesis preserve genuine disagreement or wash it out?"""
    ),
    md("""### 8.1 · Per-voter opinions"""),
    code(
        """for t in result.trace:
    print_step(
        f\"=== {t['voter'].upper()}  (confidence {t['confidence']}/5) ===\",
        f\"BOTTOM LINE: {t['bottom_line']}\"
    )
    for pt in t.get('key_points', []):
        print_step(\"    point\", pt[:200])
    print()"""
    ),
    md(
        """## 9 · What we just observed

*(Automatically tailored from the actual captured run by `scripts/tailor_13_commentary.py`.)*"""
    ),
    md(
        """## 10 · `majority_vote` mode — the deterministic-picker fix

The default `llm_synth` mode in § 8 uses the *content* of each voter's `bottom_line`, so it works even when confidence scores are flat. But what if you need a *decisive single answer* (YES / NO / A / B) — not a balanced synthesis?

The naive approach is `highest_confidence` mode: pick the voter with the highest self-reported confidence number. **This is broken on Llama** — confidences come back `[4, 4, 4]` and the argmax is arbitrary.

The fix is `majority_vote` mode: voters emit a categorical answer (e.g. `"YES"`), and **Python tallies the votes** deterministically. Confidence numbers are ignored entirely. This is the same pattern as Mental Loop's `scoring_fn` (notebook 10) — **let the LLM predict the underlying signal, let Python compute the picker**."""
    ),
    code(
        """print_header(\"Mode: majority_vote (deterministic Python picker)\")
mv_arch = Ensemble(aggregator_mode=\"majority_vote\")
mv_result = mv_arch.run(
    \"Will electric vehicles account for over 50% of new car sales globally by 2030? Answer YES, NO, or UNCERTAIN.\"
)
print_md(mv_result.output[:700])
print()
print(f\"VOTE_TALLY (Python-computed): {mv_result.metadata['vote_tally']}\")
print(f\"CATEGORICAL_ANSWERS (LLM-supplied): {mv_result.metadata['categorical_answers']}\")
print(f\"CONFIDENCES (unused for argmax — note flatness): {mv_result.metadata['confidences']}\")"""
    ),
    md(
        """## 11 · Failure modes, safety, extensions

### 11.1 · Where this breaks

| Failure | Mechanism | Mitigation |
|---|---|---|
| **Flat confidences** | Same Llama-as-Scorer pathology — all confidence 4/5 | **Use `aggregator_mode="majority_vote"`** (see § 10) — Python tallies discrete `categorical_answer` values, sidestepping the flat-confidence noise |
| **Aggregator washout** | Synthesis blends opinions so minority view disappears | Aggregator prompt EXPLICITLY asks to preserve disagreement (we do this); still happens occasionally |
| **Hidden conformity** | All 3 voters give the same answer despite different prompts | Run higher temperature; or use genuinely-different LLMs (gpt-4o + claude + llama) |
| **Voters skip categorical_answer** | Llama leaves optional field null even after instruction | Library has keyword-fallback: scans `bottom_line` for YES/NO/UNCERTAIN if `categorical_answer` is missing |
| **Adversarial-voice winner** | Skeptical voter's "what could go wrong" is loudest | Counterbalance with explicit "what could go right" voter |
| **Cost** | K voter calls + 1 aggregator = K+1 LLM calls per task | Don't use Ensemble for cheap tasks; reserve for high-stakes decisions |

### 11.2 · Production safety

- **Confidence is unreliable.** Don't expose voters' self-reported confidence to users as if it were calibrated probability — it isn't.
- **Aggregator is a single point of failure.** If the aggregator LLM gets a bad seed, the whole ensemble's output is corrupted. Run the aggregator 2-3 times and pick the best (meta-ensemble).
- **Voter diversity matters more than count.** 3 *genuinely different* voters > 7 paraphrase-of-same voters.

### 11.3 · Three extensions

1. **Parallel voter execution.** Replace the sequential `_vote` loop with `langgraph.graph.parallel` — K× latency win.
2. **Voters are full architectures.** Instead of 3 LLM calls with different prompts, use 3 *different architectures* (ReAct + Reflection + Planning) on the same task. Higher cost, much richer ensemble.
3. **Confidence-weighted aggregation.** Replace the LLM aggregator with a weighted majority vote where each voter's contribution is weighted by their confidence. (Use after fixing self-reported confidence calibration.)

### 11.4 · What to read next

- [**05 · Multi-Agent**](./05_multi_agent.ipynb) — specialists with DIFFERENT sub-tasks vs Ensemble's same-task voters.
- [**07 · Blackboard**](./07_blackboard.ipynb) — distributed bidding instead of fan-out.
- [**21 · Self-Consistency**](./21_self_consistency.ipynb) — N samples + majority vote (Ensemble's simpler cousin).
- [**28 · Multi-Agent Debate**](./28_agent_debate.ipynb) — adversarial back-and-forth instead of one-shot vote.

### 11.5 · References

1. Surowiecki, J. *The Wisdom of Crowds.* 2004.
2. Wang, X. et al. *Self-Consistency Improves Chain-of-Thought Reasoning in Language Models.* ICLR 2023. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
3. Du, Y. et al. *Improving Factuality and Reasoning in Language Models through Multiagent Debate.* 2023. [arXiv:2305.14325](https://arxiv.org/abs/2305.14325)
"""
    ),
]


def main() -> None:
    out = build_notebook(CELLS, OUT_PATH)
    print(f"wrote: {out}  ({sum(len(c[1]) for c in CELLS)} chars across {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
