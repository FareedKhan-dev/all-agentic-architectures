"""Generate notebooks/24_corrective_rag.ipynb — grade retrieved docs, fall back to web."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from notebook_builder import build_notebook, code, md  # noqa: E402

OUT_PATH = Path(__file__).parents[1] / "notebooks" / "24_corrective_rag.ipynb"


CELLS = [
    md(
        """# 24 · Corrective RAG (CRAG) — grade docs, fall back to web

> **TL;DR.** Retrieve from corpus → **grade each doc** as relevant/ambiguous/irrelevant via a categorical LLM judgement → Python composes a routing decision (use-retrieved / web-fallback / mixed) → answer.
>
> **Reach for it when** corpus coverage is incomplete and falling back to the web for out-of-corpus questions is acceptable.
> **Avoid when** retrieved docs are always relevant (CRAG adds a grade-call per doc for no benefit) or when web fallback is disallowed (security/compliance).

| Property | Value |
|---|---|
| Origin | Yan et al., *Corrective RAG* (2024). [arXiv:2401.15884](https://arxiv.org/abs/2401.15884) |
| Grader | Categorical per-doc label (`relevant` / `ambiguous` / `irrelevant`) — deterministic-picker |
| Picker | Python composes route from grade counts |
| Web tool | Tavily (`agentic_architectures.tools.web_search_tool`) — gracefully degrades if no `TAVILY_API_KEY` |
| Default LLM | Llama-3.3-70B |
| Cost | 1 retrieve + `top_k` grade-calls + (optional) 1 web-search + 1 answer = `top_k + 2` to `top_k + 3` calls |

**Why deterministic-picker matters here.** If the grader emitted a numeric "relevance score 0-1", we'd be back to the flat-scoring pathology (Mental Loop nb 10 §11). Three categorical labels give the LLM something concrete to commit to per-doc; Python counts the labels to route — the deciding signal `(n_relevant, n_irrelevant)` is fully Python-computed."""
    ),
    md(
        """## 2 · Architecture at a glance

```mermaid
flowchart LR
    A([task]) --> R[RETRIEVE<br/><sub>vector search top-k</sub>]
    R --> G[GRADE<br/><sub>per-doc categorical relevance</sub>]
    G --> RT[ROUTE<br/><sub>Python composes from counts</sub>]
    RT -->|use_retrieved| ANS[ANSWER]
    RT -->|use_web| W[WEB SEARCH] --> ANS
    RT -->|use_mixed| W
    ANS --> Z([final])

    style G fill:#fff3e0,stroke:#f57c00
    style RT fill:#fce4ec,stroke:#c2185b
    style W fill:#e3f2fd,stroke:#1976d2
    style ANS fill:#e8f5e9,stroke:#388e3c
```"""
    ),
    md(
        """## 3 · Theory

### 3.0 · Why categorical relevance, not a score

`_DocGrade.relevance: Literal['relevant', 'ambiguous', 'irrelevant']` — three discrete labels. The grader can't slide-into-a-flat-band because there is no number to slide. Python tallies the labels and the route is deterministic:

```python
rel_frac = n_relevant / n
if rel_frac >= threshold:      → use_retrieved
elif n_irrelevant == n:        → use_web
else:                          → use_mixed
```

### 3.1 · When CRAG beats plain RAG

- **In-corpus query** with high-relevance retrievals → CRAG behaves identically to plain RAG. The grade-call cost is the only overhead.
- **Out-of-corpus query** → plain RAG hallucinates from irrelevant docs; CRAG detects the irrelevance and pivots to web.
- **Partially-in-corpus** (multi-hop where only some facts are in corpus) → CRAG mixes retrieved + web; plain RAG misses the gap.

### 3.2 · Where this sits

| Pattern | Strategy when retrieval is poor |
|---|---|
| Plain RAG | Use whatever was retrieved (will hallucinate from junk) |
| [Agentic RAG (nb 23)](./23_agentic_rag.ipynb) | Agent can choose not to retrieve, but no fallback |
| **CRAG (this nb)** | **Grade docs; fall back to web search if irrelevant** |
| [Self-RAG (nb 25)](./25_self_rag.ipynb) | Self-emitted reflection tokens decide retrieve/answer per claim |
| [Adaptive RAG (nb 26)](./26_adaptive_rag.ipynb) | Pre-route at task-level (no-RAG / single / multi) |"""
    ),
    md("""## 4 · Setup"""),
    code(
        """from agentic_architectures import get_llm, enable_langsmith, settings
from agentic_architectures.architectures import CorrectiveRAG
from agentic_architectures.data import STARDUST_CORPUS
from agentic_architectures.tools import web_search_tool
from agentic_architectures.ui import print_md, print_header, print_step

enable_langsmith()
llm = get_llm(provider="nebius", model="meta-llama/Llama-3.3-70B-Instruct", temperature=0.2)

# Wrap Tavily as a simple callable returning list[str] of snippets.
_tavily = web_search_tool(max_results=3)
def web_search_fn(query: str) -> list[str]:
    try:
        result = _tavily.invoke(query)
        if isinstance(result, list):
            return [str(r.get('content', r))[:400] for r in result]
        return [str(result)[:1000]]
    except Exception as e:
        return [f"(web search unavailable: {e})"]

print_header(f"LLM: {llm.model}  ·  Corpus: {len(STARDUST_CORPUS)} docs  ·  Web fallback: Tavily")"""
    ),
    md(
        """## 5 · Library walkthrough

Source: [`src/agentic_architectures/architectures/corrective_rag.py`](../src/agentic_architectures/architectures/corrective_rag.py).

The `_DocGrade` schema is the deciding commitment per doc; `_route` is pure Python composing the route from label counts."""
    ),
    code(
        """from agentic_architectures.architectures.corrective_rag import _DocGrade, CorrectiveRAG
import json, inspect
print('--- _DocGrade schema ---')
print(json.dumps(_DocGrade.model_json_schema(), indent=2)[:400] + '...')
print()
print('--- _route source ---')
print(inspect.getsource(CorrectiveRAG._route))"""
    ),
    md("""## 6 · Build the graph"""),
    code(
        """from IPython.display import Image, display
arch = CorrectiveRAG(
    llm=llm,
    documents=STARDUST_CORPUS,
    web_search_fn=web_search_fn,
    top_k=3,
    relevance_threshold=0.5,
)
graph = arch.build()
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"(mermaid PNG render unavailable: {e}; see § 2)")
    print(graph.get_graph().draw_mermaid())"""
    ),
    md(
        """## 8 · Live run — 3 task types with varying corpus coverage

1. **In-corpus** — answer is in the Stardust corpus.
2. **Out-of-corpus** — answer requires web search.
3. **Mixed** — both corpus and web add value."""
    ),
    code(
        """TASKS = [
    ("in_corpus",   "What propellant does the Phoenix-2 engine use?"),
    ("out_of_corpus", "What is the current population of Iceland (2024)?"),
    ("mixed",       "Compare the Stardust 9 rocket's payload to SpaceX Falcon 9's payload to LEO."),
]

for tag, q in TASKS:
    r = arch.run(q)
    print(f"TASK_TAG: {tag}")
    print(f"  TASK: {q[:80]}")
    print(f"  N_RETRIEVED: {r.metadata['n_retrieved']}")
    print(f"  N_RELEVANT: {r.metadata['n_relevant']}")
    print(f"  N_AMBIGUOUS: {r.metadata['n_ambiguous']}")
    print(f"  N_IRRELEVANT: {r.metadata['n_irrelevant']}")
    print(f"  RELEVANCE_FRACTION: {r.metadata['relevance_fraction']:.2f}")
    print(f"  ROUTE: {r.metadata['route']}")
    print(f"  N_WEB: {r.metadata['n_web']}")
    print(f"  FINAL_ANSWER: {r.output[:200]}")
    print()"""
    ),
    md(
        """## 9 · What we just observed

*(Automatically tailored from the actual captured run by `scripts/tailor_24_commentary.py`.)*"""
    ),
    md(
        """## 11 · Failure modes, safety, extensions

### 11.1 · Where this breaks

| Failure | Mechanism | Mitigation |
|---|---|---|
| **Grader hallucinates `relevant`** | LLM thinks an off-topic doc relates because of surface keyword overlap | Add a second grade pass; require the grader to QUOTE the relevant sentence |
| **Web fallback unreliable** | Tavily down or rate-limited | Multiple fallback sources; cache web results |
| **Threshold mismatch** | `relevance_threshold=0.5` too lenient — proceeds with mostly-irrelevant docs | Tune per corpus; A/B test |
| **Cost** | top_k grade calls per query | Cache grades by `(query_hash, doc_hash)`; batch grades |

### 11.2 · Production safety

- **Don't trust web fallback for high-stakes answers.** Tavily snippets are arbitrary web text; treat as "loose context" not "authoritative source".
- **Track route distribution.** If `use_web` dominates, the corpus is failing — index more.
- **Audit irrelevant-but-graded-relevant cases.** False positives leak hallucinated answers.

### 11.3 · Three extensions

1. **Query rewriting before web fallback.** Use a small LLM to rewrite the query for web-search-friendliness.
2. **Confidence-weighted answer.** Each grade has a confidence; weight the doc's influence on the answer.
3. **Per-doc citation requirement.** Force the answer to cite which doc supports each claim.

### 11.4 · What to read next

- [**23 · Agentic RAG**](./23_agentic_rag.ipynb) — sibling, doesn't grade.
- [**25 · Self-RAG**](./25_self_rag.ipynb) — reflection-token version.
- [**26 · Adaptive RAG**](./26_adaptive_rag.ipynb) — task-level routing.

### 11.5 · References

1. Yan, S. et al. *Corrective Retrieval Augmented Generation.* 2024. [arXiv:2401.15884](https://arxiv.org/abs/2401.15884)"""
    ),
]


def main() -> None:
    out = build_notebook(CELLS, OUT_PATH)
    print(f"wrote: {out}  ({sum(len(c[1]) for c in CELLS)} chars across {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
