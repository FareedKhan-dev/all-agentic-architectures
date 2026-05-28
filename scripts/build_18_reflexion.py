"""Generate notebooks/18_reflexion.ipynb — try, evaluate, verbal reflect, retry."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from notebook_builder import build_notebook, code, md  # noqa: E402

OUT_PATH = Path(__file__).parents[1] / "notebooks" / "18_reflexion.ipynb"


CELLS = [
    md(
        """# 18 · Reflexion — verbal self-reflection stored in episodic memory

> **TL;DR.** Try → evaluate → if failed, write a *verbal lesson* about why → store the lesson in `EpisodicMemory` → retry. Future tasks recall relevant lessons before attempting, so the agent improves across calls *without* gradient updates.
>
> **Reach for it when** the agent will see many structurally-similar tasks and you can write a precise evaluator that surfaces *which* feature failed.
> **Avoid when** tasks are one-shot (nothing to transfer to), or when no reliable evaluator exists (the lesson will be garbage).

| Property | Value |
|---|---|
| Origin | Shinn et al., *Reflexion: Language Agents with Verbal RL* (2023). [arXiv:2303.11366](https://arxiv.org/abs/2303.11366) |
| Loop body | attempt → evaluate → (pass? finalize : reflect → attempt) |
| Memory | `arch.episodic: EpisodicMemory` — **persists across `run()` calls** |
| Evaluator | Pluggable `Callable[[candidate, task], dict]`; default is pure-Python deterministic checker |
| Cost | 1 LLM call per attempt + 1 per reflection; no gradient updates ever |

**Why this is different from Reflection (nb 01).** Reflection critiques and rewrites *within a single task*, then forgets. Reflexion writes a transferable *lesson* that gets stored and recalled on *future* tasks. The pattern is "verbal RL" — the trajectory + evaluator feedback is converted into a natural-language policy update that lives in episodic memory.

**Why this is different from RLHF self-improvement (nb 15).** Both archive across `run()` calls, but RLHF stores *positive examples* (final accepted outputs) while Reflexion stores *negative-experience lessons* (corrections for past failures). The two are complementary — you could combine them."""
    ),
    md(
        """## 2 · Architecture at a glance

```mermaid
flowchart LR
    A([task]) --> AT[Attempt<br/><sub>prompt prepends recalled lessons</sub>]
    AT --> EV[Evaluate<br/><sub>deterministic Python checker</sub>]
    EV -->|passed OR trial>=max| F[Finalize]
    EV -->|else| RF[Reflect<br/><sub>write a verbal lesson</sub>]
    RF --> AT
    RF -.records.-> M[(EpisodicMemory<br/>arch.episodic<br/>persists across run calls)]
    AT -.recalls.-> M
    F --> Z([final output])

    style AT fill:#e3f2fd,stroke:#1976d2
    style EV fill:#fff3e0,stroke:#f57c00
    style RF fill:#fce4ec,stroke:#c2185b
    style M  fill:#f3e5f5,stroke:#7b1fa2
```

The dotted side-edges are the load-bearing detail: `reflect` writes into the memory; the *next* `attempt` (in this run *or any future `run()` call*) reads from it. That cross-call persistence is the entire point of Reflexion."""
    ),
    md(
        """## 3 · Theory

### 3.0 · Why a deterministic Python checker, not LLM-as-Judge

Llama-3.3-70B (and most instruction-tuned LLMs) compress numerical LLM-as-Scorer outputs to a flat band — see Mental Loop nb 10 §11 and Ensemble nb 13 §11. If we let the LLM emit a single `quality_score: 1-10`, the score band collapses to ~`[4,4,4]` regardless of rubric strictness, and the deciding signal becomes effectively arbitrary.

**Reflexion sidesteps this entirely** by defaulting to a pure-Python evaluator (`default_haiku_checker`):

```python
def default_haiku_checker(candidate, task_spec):
    # ... parse spec for required words + topic
    syllable_counts = [count_syllables(line) for line in three_lines]
    return {
        "syllable_counts": syllable_counts,                  # e.g. [5,8,5]
        "meets_5_7_5":  syllable_counts == [5, 7, 5],         # bool
        "required_words_present": all(w in text for w in required),
        "passed": <all the above are True>,
    }
```

The `passed` boolean is *the* deciding signal. No LLM is involved in pass/fail. The reflection LLM only writes the *lesson*, which is free-form text — flat-scoring has nothing to compress there.

For tasks where no deterministic checker exists, the architecture accepts an optional `evaluator` callable. If you wrap an LLM-as-Judge there, follow the deterministic-picker pattern: have the LLM commit to independent boolean features and let Python compose `passed`. See `_ReflexionEvaluation` schema in [`reflexion.py`](../src/agentic_architectures/architectures/reflexion.py) for the template.

### 3.1 · Why store the *full verbal reflection*, not feature embeddings

You could imagine storing only `(failed_features → corrective_action)` tuples — compact, structured, easy to retrieve. The Reflexion paper rejects this on purpose: the *next* model reading the lesson is also an LLM, and LLMs reason better over natural language than over structured deltas. A two-sentence "You under-counted syllables in line 2 — count out loud before submitting" steers the next attempt; a JSON blob doesn't.

The `_SelfReflection` Pydantic schema enforces second-person voice and demands explicit `root_cause` + `correction` fields, then composes them into a `reflection` paragraph. The paragraph is what gets stored in episodic memory; the structured fields are kept on the trace for the §9 commentary table.

### 3.2 · Where this sits in the agent taxonomy

| Pattern | Persists across calls? | Stores what? | When to reach for it |
|---|---|---|---|
| [Reflection (nb 01)](./01_reflection.ipynb) | no | nothing | quality matters, one-shot |
| [Episodic+Semantic (nb 08)](./08_episodic_semantic_memory.ipynb) | yes | conversations + facts | personal assistant continuity |
| [RLHF self-improvement (nb 15)](./15_rlhf_self_improvement.ipynb) | yes | **positive** examples (good outputs) | many similar tasks, compound quality |
| **Reflexion (this nb)** | **yes** | **negative-experience lessons** (verbal corrections) | learn from mistakes across similar tasks |

The orthogonal axes: *positive vs negative examples* (RLHF vs Reflexion) and *experience-summary vs raw memory* (Reflexion vs Episodic).

### 3.3 · Failure modes preview

You'll see all of these surface in §8 or §9 below:

1. **Demo too easy.** If the agent passes on trial 1 every time, no reflections are written — the demo doesn't actually exercise the architecture. The §9 tailor auto-flags this.
2. **Lesson too task-specific.** Lessons phrased as "next time, use the word 'centuries' in line 2" don't transfer. The schema's `correction` field asks for *generalisable* corrections, but the LLM can still narrow them.
3. **Recall misses.** EpisodicMemory uses vector similarity; if the embedding model can't see structural similarity between tasks, recalled lessons will be empty even when memory is non-empty.
4. **Loop exhaustion.** If `max_trials` is too low (or the task is too hard), the loop terminates failed. Reflections are still recorded — they benefit *later* tasks even though this one failed."""
    ),
    md("""## 4 · Setup"""),
    code(
        """from agentic_architectures import get_llm, enable_langsmith, settings
from agentic_architectures.architectures import Reflexion
from agentic_architectures.ui import print_md, print_header, print_step

enable_langsmith()
print_header(f"Provider: {settings.llm_provider}  ·  Model: {settings.llm_model}")"""
    ),
    md(
        """## 5 · Library walkthrough

Source: [`src/agentic_architectures/architectures/reflexion.py`](../src/agentic_architectures/architectures/reflexion.py).

Three things make `Reflexion` distinct from nb 01 Reflection:

1. **`self.episodic: EpisodicMemory`** — created once in `__init__`, mutated by every failed-trial `reflect` node, queried by every `attempt` node. Persists across `run()` calls on the same instance.
2. **`evaluator` is a pluggable `Callable`** — defaults to `default_haiku_checker` (pure Python). You can swap in any function returning `{... "passed": bool}`.
3. **`_SelfReflection` schema** demands three structured fields (`root_cause`, `correction`, `reflection`) so the stored memory is *actionable*, not vague.

The Pydantic schema for the verbal reflection:"""
    ),
    code(
        """from agentic_architectures.architectures.reflexion import _SelfReflection, default_haiku_checker
import json
print("=== _SelfReflection schema ===")
print(json.dumps(_SelfReflection.model_json_schema(), indent=2)[:600] + '...')
print()
print("=== default_haiku_checker docstring ===")
print(default_haiku_checker.__doc__)"""
    ),
    md("""## 6 · State"""),
    md(
        """| Field | Type | Set by |
|---|---|---|
| `task` | `str` | caller |
| `max_trials` | `int` | caller (from `arch.max_trials`) |
| `trial` | `int` (0-indexed; incremented in `attempt`) | `_attempt` |
| `attempt_text` | `str` (latest candidate) | `_attempt` |
| `recalled_reflections` | `list[str]` (pulled from `self.episodic`) | `_attempt` |
| `evaluator_features` | `dict` (objective features from checker) | `_evaluate` |
| `success` | `bool` (= `evaluator_features['passed']`) | `_evaluate` |
| `reflection_text` / `root_cause` / `correction` | `str` | `_reflect` |
| `history` | `Annotated[list[dict], operator.add]` — one entry per trial + one per reflection | `_evaluate` + `_reflect` |
| `final_output` | `str` | `_finalize` |
| `arch.episodic` *(instance attribute)* | `EpisodicMemory` — **persists across `run()` calls** | `_reflect` side-effect |"""
    ),
    md("""## 7 · Build the graph"""),
    code(
        """from IPython.display import Image, display
arch = Reflexion(max_trials=3, reflections_to_recall=3)
graph = arch.build()
display(Image(graph.get_graph().draw_mermaid_png()))"""
    ),
    md(
        """## 8 · Live run — 3 structurally-similar haiku tasks

We run **three constrained-haiku tasks** through ONE `Reflexion` instance so failed-trial reflections accumulate. Each task: write a haiku on a different topic, containing two required words, in strict 5-7-5 syllables. The deterministic checker validates all three constraints.

**Watch for:**
- Trial count **drops** across tasks if memory transfers (task 2 / task 3 should benefit from task 1's lessons).
- `REFLECTIONS_RECALLED` should be **≥1** on tasks 2 and 3 (assuming memory was populated).
- If task 1 passes on trial 1, the demo is uninstructive — no reflections were written. The §9 tailor flags this."""
    ),
    code(
        """TASKS = [
    {"tag": "glacier", "topic": "glacier", "required_words": ["silence", "centuries"]},
    {"tag": "subway",  "topic": "subway",  "required_words": ["midnight", "rumble"]},
    {"tag": "library", "topic": "library", "required_words": ["paper", "dust"]},
]

arch = Reflexion(max_trials=3, reflections_to_recall=3)

trials_per_task = []
for t in TASKS:
    w1, w2 = t["required_words"]
    task_text = (
        f'Write a haiku about a {t["topic"]}. '
        f'The haiku MUST contain the words "{w1}" and "{w2}". '
        f'It MUST follow strict 5-7-5 syllables. '
        f'spec=topic={t["topic"]}; required_words={w1},{w2}'
    )
    r = arch.run(task_text)
    feats = r.metadata["evaluator_features"]
    # Collect per-trial syllable counts + word-presence flags from trace events
    per_trial_syllables = [h["features"]["syllable_counts"] for h in r.trace if h.get("type") == "trial"]
    per_trial_words = [h["features"]["required_words_present"] for h in r.trace if h.get("type") == "trial"]
    print(f"TASK_TAG: {t['tag']}")
    print(f"  TRIALS_USED: {r.metadata['total_trials']}")
    print(f"  SUCCESS: {r.metadata['succeeded']}")
    print(f"  REFLECTIONS_RECALLED: {r.metadata['reflections_recalled_first_trial']}")
    print(f"  REFLECTIONS_IN_MEMORY_AFTER: {r.metadata['total_reflections_in_memory']}")
    print(f"  SYLLABLES_PER_TRIAL: {per_trial_syllables}")
    print(f"  WORDS_PRESENT_PER_TRIAL: {per_trial_words}")
    print(f"  FINAL: {r.output.strip().replace(chr(10), ' / ')}")
    print()
    trials_per_task.append(r.metadata["total_trials"])

print(f"TRIALS_PER_TASK: {trials_per_task}   total_reflections_in_memory={len(arch.episodic.episodes)}")"""
    ),
    md(
        """### 8.0 · What just happened, briefly

Three signals to read:

1. **`TRIALS_PER_TASK` should trend downward** if reflection memory is transferring. A pattern like `[2, 1, 1]` means task 1 needed reflection, then later tasks benefitted. A flat `[1, 1, 1]` means the demo was too easy; a flat `[3, 3, 3]` means reflections aren't actually helping.
2. **`REFLECTIONS_RECALLED` ≥ 1 on tasks 2 and 3.** If it's 0 when memory is non-empty, FAISS recall isn't finding the lessons — usually means the embedding model is too weak for the task texts.
3. **`SUCCESS` per task.** Reflexion isn't magic — if max_trials runs out, the loop terminates failed. Reflections are still written and benefit *future* tasks."""
    ),
    md("""### 8.1 · Inspect the lesson library"""),
    code(
        """print(f"Total lessons stored: {len(arch.episodic.episodes)}")
print()
for i, ep in enumerate(arch.episodic.episodes, 1):
    md_block = ep.metadata or {}
    print(f"[lesson {i}]  (recorded on task='{md_block.get('task', '?')[:50]}...', trial={md_block.get('trial')})")
    print(f"  root_cause:  {md_block.get('root_cause', '(missing)')}")
    print(f"  correction:  {md_block.get('correction', '(missing)')}")
    print(f"  reflection:  {ep.content}")
    print()"""
    ),
    md(
        """## 9 · What we just observed

*(Automatically tailored from the actual captured run by `scripts/tailor_18_commentary.py`.)*"""
    ),
    md(
        """## 10 · Contrast — same task with EMPTY memory

To make the memory-transfer effect concrete, re-run **task 1** with a brand-new `Reflexion()` instance (empty episodic memory) and compare trial count to what the warm-memory run did."""
    ),
    code(
        """fresh = Reflexion(max_trials=3, reflections_to_recall=3)
t = TASKS[0]
w1, w2 = t["required_words"]
task_text = (
    f'Write a haiku about a {t["topic"]}. '
    f'The haiku MUST contain the words "{w1}" and "{w2}". '
    f'It MUST follow strict 5-7-5 syllables. '
    f'spec=topic={t["topic"]}; required_words={w1},{w2}'
)
r_cold = fresh.run(task_text)
print(f"COLD_START_TASK: {t['tag']}")
print(f"COLD_START_TRIALS: {r_cold.metadata['total_trials']}")
print(f"COLD_START_SUCCESS: {r_cold.metadata['succeeded']}")
print(f"COLD_START_REFLECTIONS_RECALLED: {r_cold.metadata['reflections_recalled_first_trial']}")
print()
print(f"(For comparison: warm-memory run of task '{t['tag']}' used {trials_per_task[0]} trials with "
      f"the warm `arch` instance.)")"""
    ),
    md(
        """## 11 · Failure modes, safety, extensions

### 11.1 · Where this breaks

| Failure | Mechanism | Mitigation |
|---|---|---|
| **Lesson contamination** | A lesson from task A misfires on unrelated task B because vector similarity is fooled by surface words | Tag lessons with task-type metadata; filter recall by tag |
| **Over-generalisation** | "Always use shorter words" → the agent applies it where unnecessary | Reflection LLM prompted to write *conditional* lessons; review periodically |
| **Checker false-negative** | The naive syllable counter mis-counts a word; agent writes a lesson about a non-failure | Use a stronger syllable library (e.g. `pyphen`); audit checker against ground truth |
| **Memory bloat** | Hundreds of near-duplicate lessons clog recall | Dedup on cosine similarity; or prune lessons older than N runs |
| **Demo too easy** | Llama passes trial 1 every time; no reflections ever written | Choose harder constraints (handoff §10 row 18 already calls this out); §9 tailor auto-flags |

### 11.2 · Production safety

- **Per-user `collection_name`.** Lessons learned about one user's tasks must not bleed into another user's. Construct `EpisodicMemory(collection_name=f"reflexion_{user_id}")` per session.
- **Lesson review pipeline.** A poisoned lesson ("Always insert 'lorem ipsum' in line 2") will degrade every future call. Surface new lessons for human review before they enter the active store.
- **Decay / pruning.** Lessons from older runs may be stale (model version changed, task distribution shifted). Track lesson age; prune or re-weight on a schedule.

### 11.3 · Three extensions

1. **Tagged retrieval.** Store each lesson with a `task_type` label; restrict recall to matching labels. Eliminates cross-task contamination.
2. **Lesson dedup.** After each `reflect`, embed the new lesson and cosine-compare against the last K. If above threshold, *update* the existing lesson (increment a `seen_count`) instead of appending.
3. **Hybrid evaluator (`_ReflexionEvaluation`).** Use the deterministic checker for hard constraints (syllables, required words) AND an LLM-as-Judge (with the deterministic-picker pattern) for soft quality. Compose pass/fail from both.

### 11.4 · What to read next

- [**01 · Reflection**](./01_reflection.ipynb) — same loop topology, no cross-call memory.
- [**08 · Episodic + Semantic Memory**](./08_episodic_semantic_memory.ipynb) — the raw `EpisodicMemory` API that Reflexion consumes.
- [**15 · RLHF Self-Improvement**](./15_rlhf_self_improvement.ipynb) — sister pattern; stores positive examples instead.
- [**29 · Voyager**](./29_voyager_skill_library.ipynb) — extends the memory-of-experience idea to memory-of-skills (reusable code).

### 11.5 · References

1. Shinn, N. et al. *Reflexion: Language Agents with Verbal Reinforcement Learning.* NeurIPS 2023. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
2. Madaan, A. et al. *Self-Refine: Iterative Refinement with Self-Feedback.* NeurIPS 2023. [arXiv:2303.17651](https://arxiv.org/abs/2303.17651) — the underlying self-critique loop.
3. Park, J. S. et al. *Generative Agents.* UIST 2023. [arXiv:2304.03442](https://arxiv.org/abs/2304.03442) — episodic memory + reflection pattern in agent simulations.
4. Wang, G. et al. *Voyager: An Open-Ended Embodied Agent with Large Language Models.* 2023. [arXiv:2305.16291](https://arxiv.org/abs/2305.16291) — skill-library extension."""
    ),
]


def main() -> None:
    out = build_notebook(CELLS, OUT_PATH)
    print(f"wrote: {out}  ({sum(len(c[1]) for c in CELLS)} chars across {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
