# The deterministic-picker pattern

This is the central technical pattern of the repo. It shows up in 8+ architectures and is the universal escape from the LLM-as-Scorer flat-band pathology.

## The problem

Ask Llama-3.3-70B (or most instruction-tuned LLMs) to emit a single numeric quality score on a 1-5 or 1-10 scale, and you get this:

```
sample 1: score=4/5
sample 2: score=4/5
sample 3: score=4/5
sample 4: score=4/5
```

Regardless of how strict the rubric. Even when prompted "be calibrated, reserve 5/5 for genuine excellence", the model collapses to a narrow band. This is documented in [Tree of Thoughts (nb 09)](../architectures/09_tree_of_thoughts.ipynb), [Mental Loop (nb 10)](../architectures/10_mental_loop.ipynb), [Ensemble (nb 13)](../architectures/13_ensemble.ipynb).

Architectures that *depend on* the score to pick something (beam search, MCTS, ranked retrieval, accept-or-reject loops) become **arbitrary** — there's no signal to discriminate on.

## The fix

Don't ask the LLM for a number. Ask it for **categorical features** the score will be composed from, then have **Python** compose the deciding signal:

```python
class _EditorCritique(BaseModel):
    is_on_brief: bool                          # LLM commits to a bool, not a number
    word_count: int
    has_concrete_imagery: bool
    avoids_cliches: bool
    is_engaging: bool

def _composite_score(features: dict, wc_range: tuple) -> int:
    score = 4 * features["is_on_brief"]
    score += 2 if wc_range[0] <= features["word_count"] <= wc_range[1] else 0
    score += 2 * features["has_concrete_imagery"]
    score += 1 * features["avoids_cliches"]
    score += 1 * features["is_engaging"]
    return score  # 0-10, with REAL SPREAD
```

The LLM can't flat-band 5 independent booleans the way it flat-bands one number. Python's `score` now ranges over `[0, 10]` honestly because it depends on 5 separate commitments.

## Why this works

1. **Granular commitment**. Saying "yes, this avoids clichés" is a different cognitive operation than saying "this is a 6/10".
2. **Auditable**. You can show the user *which* features drove the score.
3. **Python computes the number**. The LLM never emits the deciding signal directly.

## Where the pattern shows up

| Architecture | LLM commits to | Python composes |
|---|---|---|
| [Mental Loop (nb 10)](../architectures/10_mental_loop.ipynb) | `predicted_metric: float` | `scoring_fn(predicted_metric) → int` |
| [Ensemble (nb 13)](../architectures/13_ensemble.ipynb) | `categorical_answer: str` | `Counter(answers).most_common(1)` |
| [Dry-Run (nb 14)](../architectures/14_dry_run.ipynb) | `irreversibility: int 1-5` | `approved = irreversibility < threshold` |
| [RLHF Self-Improvement (nb 15)](../architectures/15_rlhf_self_improvement.ipynb) | 5 booleans + word_count | weighted composite |
| [Reflexive Metacognitive (nb 17)](../architectures/17_reflexive_metacognitive.ipynb) | `requires_credentials: bool, capability_match: int` | `if creds or cap<=2: route='escalate'` |
| [Self-Consistency (nb 21)](../architectures/21_self_consistency.ipynb) | per-sample `answer: str` | `Counter` majority vote |
| [LATS (nb 22)](../architectures/22_lats.ipynb) | `(makes_progress, is_complete, avoids_loops, confidence)` | `5*complete + 2*progress + 1*no_loops + conf_weight` |
| [Corrective RAG (nb 24)](../architectures/24_corrective_rag.ipynb) | per-doc `Literal[relevant, ambiguous, irrelevant]` | route from label counts |
| [Self-RAG (nb 25)](../architectures/25_self_rag.ipynb) | per-doc 3 categorical reflection tokens | Python AND `is_relevant != not_relevant AND is_supported != no_support` |
| [Adaptive RAG (nb 26)](../architectures/26_adaptive_rag.ipynb) | `complexity: Literal[no_retrieval, single, multi]` | `if/elif` route |
| [Debate (nb 28)](../architectures/28_debate.ipynb) | per-agent `answer: str` | `Counter` on final round |
| [Constitutional AI (nb 32)](../architectures/32_constitutional_ai.ipynb) | per-rule `verdict: Literal[pass, fail]` | `all(v == "pass")` |
| [BrowserAgent (nb 34)](../architectures/34_computer_use.ipynb) | structured action with `target` | `_check_safety(action) → allowed: bool` |

## Architecturally immune by design

Some architectures have no LLM-as-Scorer step at all because their decisions are categorical or content-based:

- **Reflexion (nb 18)** — pass/fail is a pure-Python checker (`default_haiku_checker`); recall is vector similarity (FAISS does its job)
- **Self-Discover (nb 19)** — SELECT picks indices; ADAPT/IMPLEMENT produce text; SOLVE produces an answer
- **CoVe (nb 20)** — REVISE makes keep/drop decisions per claim; confidence is categorical
- **GraphRAG (nb 27)** — local vs global is categorical; traversal is mechanical
- **Voyager (nb 29)** — reuse vs write_new is categorical; skills execute in subprocess
- **MemGPT (nb 31)** — action is `Literal[write_to_archival, search_archival, answer]`
- **SWE-Agent (nb 33)** — action is `Literal[list, read, write, run_check, answer]`
- **AWM (nb 35)** — retrieve match / no-match

## The takeaway

Whenever an architecture has a *picker* — a step that ranks, scores, or selects — apply this discipline:

1. **Identify the categorical features** the picker should decide on.
2. **Pydantic-schema them** with strict types (`bool`, `int` with bounds, `Literal[...]`).
3. **Compose the deciding signal in Python** using those features.
4. **Keep the LLM's numeric output (if any) on the trace** for comparison only — never as the deciding value.

This pattern is *architectural*, not a hyperparameter. Once you build it in, the architecture is immune to the flat-band pathology for the lifetime of the codebase.
