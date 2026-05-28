"""Generate notebooks/25_self_rag.ipynb — per-doc reflection tokens."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from notebook_builder import build_notebook, code, md  # noqa: E402

OUT_PATH = Path(__file__).parents[1] / "notebooks" / "25_self_rag.ipynb"


CELLS = [
    md(
        """# 25 · Self-RAG — retrieve-on-demand with reflection tokens

> **TL;DR.** Decide whether to retrieve; if yes, fetch top-k docs; emit **per-doc reflection tokens** (categorical `is_relevant`, `is_supported`, `is_useful`); Python composes a keep/drop boolean per doc; answer from kept docs only.
>
> **Reach for it when** you need finer-grained doc-level quality control than CRAG's whole-batch routing, and want explicit per-doc audit signals.
> **Avoid when** retrievals are usually clean (the per-doc reflection cost is wasted) or when the corpus is small enough that reflection cost dominates.

| Property | Value |
|---|---|
| Origin | Asai et al., *Self-RAG* (2024). [arXiv:2310.11511](https://arxiv.org/abs/2310.11511) |
| Reflection tokens | 3-way categorical per doc: `is_relevant`, `is_supported`, `is_useful` |
| Picker | Python composes keep/drop from token labels — deterministic-picker |
| Default LLM | Llama-3.3-70B |
| Cost | 1 decide + 1 retrieve + `top_k` reflect + 1 answer = `top_k + 3` calls |

**Why deterministic-picker is the load-bearing pattern here.** Self-RAG's reflection tokens were originally LEARNED special tokens (the paper trains a model to emit them). We simulate them via Pydantic structured output. The key fidelity-preserving move: every token is `Literal[...]` — categorical — not a numeric score. Python composes the keep/drop boolean as `is_relevant != 'not_relevant' AND is_supported != 'no_support'`."""
    ),
    md(
        """## 2 · Architecture at a glance

```mermaid
flowchart TB
    A([task]) --> D[DECIDE_RETRIEVAL<br/><sub>bool: parametric or external?</sub>]
    D -->|False| ANS[ANSWER from parametric]
    D -->|True| R[RETRIEVE top-k]
    R --> RF[REFLECT per doc<br/><sub>3 categorical tokens per doc</sub>]
    RF --> CK[COMPOSE_KEEP<br/><sub>Python boolean per doc</sub>]
    CK --> ANS
    ANS --> Z([final])

    style RF fill:#fff3e0,stroke:#f57c00
    style CK fill:#fce4ec,stroke:#c2185b
    style ANS fill:#e8f5e9,stroke:#388e3c
```"""
    ),
    md(
        """## 3 · Theory

### 3.0 · The three reflection tokens

Per Asai et al.:
- **`is_relevant`** — does this doc address the question?
- **`is_supported`** — would using this doc produce a well-grounded answer?
- **`is_useful`** — overall usefulness signal.

Each is a 3-way categorical (`fully_X` / `partially_X` / `not_X` or `no_X`). No numeric scoring.

### 3.1 · How Python composes the keep/drop

```python
def _compose_keep(state):
    return [
        i for i, t in enumerate(state['reflection_tokens'])
        if t['is_relevant'] != 'not_relevant'
        and t['is_supported'] != 'no_support'
    ]
```

This is the deciding signal — Python AND on two categorical commitments. The LLM never emits a numeric usefulness score.

### 3.2 · Where this sits in the RAG family

| Pattern | Granularity of quality control |
|---|---|
| Plain RAG | None |
| [Agentic RAG (nb 23)](./23_agentic_rag.ipynb) | None (agent just answers) |
| [CRAG (nb 24)](./24_corrective_rag.ipynb) | Whole batch (route based on aggregate) |
| **Self-RAG (this nb)** | **Per-document** (drop individual junk docs) |
| [Adaptive RAG (nb 26)](./26_adaptive_rag.ipynb) | Per-task (no/single/multi-step routing) |"""
    ),
    md("""## 4 · Setup"""),
    code(
        """from agentic_architectures import get_llm, enable_langsmith, settings
from agentic_architectures.architectures import SelfRAG
from agentic_architectures.data import STARDUST_CORPUS
from agentic_architectures.ui import print_md, print_header

enable_langsmith()
llm = get_llm(provider="nebius", model="meta-llama/Llama-3.3-70B-Instruct", temperature=0.2)
print_header(f"LLM: {llm.model}  ·  Corpus: {len(STARDUST_CORPUS)} docs")"""
    ),
    md(
        """## 5 · Library walkthrough"""
    ),
    code(
        """from agentic_architectures.architectures.self_rag import _ReflectionTokens, SelfRAG
import json, inspect
print('--- _ReflectionTokens schema ---')
print(json.dumps(_ReflectionTokens.model_json_schema(), indent=2)[:700] + '...')
print()
print('--- _compose_keep (Python) ---')
print(inspect.getsource(SelfRAG._compose_keep))"""
    ),
    md("""## 7 · Build the graph"""),
    code(
        """from IPython.display import Image, display
arch = SelfRAG(llm=llm, documents=STARDUST_CORPUS, top_k=4)
graph = arch.build()
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"(mermaid PNG render unavailable: {e}; see § 2)")
    print(graph.get_graph().draw_mermaid())"""
    ),
    md(
        """## 8 · Live run — 3 tasks

1. **Direct retrieval task** — answer in corpus.
2. **Parametric task** — agent should skip retrieval.
3. **Mismatch task** — corpus has loosely-related docs but no direct answer; reflection should drop most/all docs."""
    ),
    code(
        """TASKS = [
    ("direct",     "When was Stardust Aerospace founded and by whom?"),
    ("parametric", "What is 25 squared? Return only the integer."),
    ("mismatch",   "What is the boiling point of liquid nitrogen?"),
]

for tag, q in TASKS:
    r = arch.run(q)
    print(f"TASK_TAG: {tag}")
    print(f"  TASK: {q}")
    print(f"  NEEDS_RETRIEVAL: {r.metadata['needs_retrieval']}")
    print(f"  N_RETRIEVED: {r.metadata['n_retrieved']}")
    print(f"  N_KEPT: {r.metadata['n_kept']}")
    print(f"  KEPT_INDICES: {r.metadata['kept_indices']}")
    print(f"  N_FULLY_RELEVANT: {r.metadata['n_fully_relevant']}")
    print(f"  N_NO_SUPPORT: {r.metadata['n_no_support']}")
    print(f"  N_VERY_USEFUL: {r.metadata['n_very_useful']}")
    tokens = r.metadata['reflection_tokens']
    for i, t in enumerate(tokens):
        print(f"    doc[{i}]: rel={t['is_relevant']}  sup={t['is_supported']}  use={t['is_useful']}")
    print(f"  FINAL_ANSWER: {r.output[:200]}")
    print()"""
    ),
    md(
        """## 9 · What we just observed

*(Automatically tailored from the actual captured run by `scripts/tailor_25_commentary.py`.)*"""
    ),
    md(
        """## 11 · Failure modes, safety, extensions

| Failure | Mitigation |
|---|---|
| **All docs dropped** (over-strict reflector) | Loosen the keep condition; add `partially_relevant + fully_supported` as keep |
| **Junk doc kept** | Reflector too lenient on `is_relevant`; reword schema field description |
| **No reflection for parametric task** | Architecture skips reflection when `needs_retrieval=False` — correct behaviour |
| **Cost** | `top_k` extra calls per task | Cache reflection per (query_hash, doc_hash) |

Extensions: (1) weighted combination of three tokens, (2) skip reflection if first doc is fully_relevant + fully_supported (early termination), (3) hierarchical reflection on sub-claims.

References:
- Asai et al., *Self-RAG.* 2024. [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)"""
    ),
]


def main() -> None:
    out = build_notebook(CELLS, OUT_PATH)
    print(f"wrote: {out}  ({sum(len(c[1]) for c in CELLS)} chars across {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
