"""Generate notebooks/26_adaptive_rag.ipynb — pre-route by complexity bucket."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from notebook_builder import build_notebook, code, md  # noqa: E402

OUT_PATH = Path(__file__).parents[1] / "notebooks" / "26_adaptive_rag.ipynb"


CELLS = [
    md(
        """# 26 · Adaptive RAG — router picks no/single/multi-step

> **TL;DR.** One LLM call classifies each query into a complexity bucket (`no_retrieval` / `single_step` / `multi_step`); Python routes to the matched strategy. Combines [Meta-Controller (nb 11)](./11_meta_controller.ipynb)'s pre-routing with the RAG family's three execution modes.

| Property | Value |
|---|---|
| Origin | Jeong et al., *Adaptive-RAG* (NAACL 2024). [arXiv:2403.14403](https://arxiv.org/abs/2403.14403) |
| Routing | Categorical (3-way) classifier — deterministic-picker |
| Cost | 1 classify + 1-3 execution calls (depending on bucket) |
| Default LLM | Llama-3.3-70B |"""
    ),
    md(
        """## 2 · Architecture at a glance

```mermaid
flowchart TB
    A([task]) --> C[CLASSIFY<br/><sub>categorical complexity</sub>]
    C -->|no_retrieval| N[Parametric answer]
    C -->|single_step| S[1 retrieve → answer]
    C -->|multi_step| M[2 retrievals → answer]
    N --> Z([final])
    S --> Z
    M --> Z

    style C fill:#fff3e0,stroke:#f57c00
    style N fill:#e3f2fd,stroke:#1976d2
    style S fill:#e8f5e9,stroke:#388e3c
    style M fill:#fce4ec,stroke:#c2185b
```"""
    ),
    md(
        """## 3 · Theory

### 3.0 · Why pre-classify

Self-RAG (nb 25) and CRAG (nb 24) make per-doc decisions *after* retrieval. Agentic RAG (nb 23) makes iterative decisions *during* retrieval. Adaptive RAG makes the **strategy** decision *before* anything else — one classifier call replaces a more expensive routing loop.

Trade-off: cheaper, but locks in the strategy. A misclassified `single_step` query gets one retrieval even if it really needed multi-hop.

### 3.1 · Where this sits

| Pattern | Where the routing happens |
|---|---|
| Plain RAG | Nowhere — always retrieve once |
| [Agentic RAG (nb 23)](./23_agentic_rag.ipynb) | Each loop iteration |
| [CRAG (nb 24)](./24_corrective_rag.ipynb) | After retrieval, on the batch |
| [Self-RAG (nb 25)](./25_self_rag.ipynb) | After retrieval, per-doc |
| **Adaptive RAG (this nb)** | **Pre-retrieval, on the query** |"""
    ),
    md("""## 4 · Setup"""),
    code(
        """from agentic_architectures import get_llm, enable_langsmith, settings
from agentic_architectures.architectures import AdaptiveRAG
from agentic_architectures.data import STARDUST_CORPUS
from agentic_architectures.ui import print_md, print_header

enable_langsmith()
llm = get_llm(provider="nebius", model="meta-llama/Llama-3.3-70B-Instruct", temperature=0.2)
print_header(f"LLM: {llm.model}  ·  Corpus: {len(STARDUST_CORPUS)} docs")"""
    ),
    md("""## 5 · Library walkthrough"""),
    code(
        """from agentic_architectures.architectures.adaptive_rag import _ComplexityClass
import json
print(json.dumps(_ComplexityClass.model_json_schema(), indent=2)[:500] + '...')"""
    ),
    md("""## 7 · Build the graph"""),
    code(
        """from IPython.display import Image, display
arch = AdaptiveRAG(llm=llm, documents=STARDUST_CORPUS, top_k=3)
graph = arch.build()
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"(PNG render unavailable: {e}; see § 2)")
    print(graph.get_graph().draw_mermaid())"""
    ),
    md(
        """## 8 · Live run — 3 tasks of varying complexity

Tasks chosen to exercise each routing bucket."""
    ),
    code(
        """TASKS = [
    ("arithmetic",     "What is 15 plus 27? Return only the integer."),
    ("simple_lookup",  "What propellant does the Stardust 9 rocket use?"),
    ("multi_hop",      "Who founded Stardust Aerospace, and what did the CEO do before founding it?"),
]

for tag, q in TASKS:
    r = arch.run(q)
    print(f"TASK_TAG: {tag}")
    print(f"  TASK: {q}")
    print(f"  ROUTED_TO: {r.metadata['complexity']}")
    print(f"  CLASSIFICATION_RATIONALE: {r.metadata['classification_rationale']}")
    print(f"  RETRIEVAL_COUNT: {r.metadata['retrieval_count']}")
    print(f"  FINAL_ANSWER: {r.output[:300]}")
    print()"""
    ),
    md(
        """## 9 · What we just observed

*(Automatically tailored from the actual captured run by `scripts/tailor_26_commentary.py`.)*"""
    ),
    md(
        """## 11 · Failure modes & extensions

| Failure | Mitigation |
|---|---|
| **Misclassification** | Classifier sends multi_hop to single_step → incomplete answer | Add a post-answer "is the answer complete?" check |
| **Bucket too coarse** | Some queries need 3+ retrievals | Add `deep_multi_step` bucket and an Agentic-RAG-style iterative executor |
| **Cost of classifier call** | One extra call per task | Cache classifications by query template |

Extensions: (1) learned classifier (train on labelled query→complexity pairs), (2) compose with CRAG inside each executor for grade-based fallback, (3) per-bucket different LLMs (cheap LLM for no_retrieval, stronger for multi_step).

Reference: Jeong et al. 2024 — [arXiv:2403.14403](https://arxiv.org/abs/2403.14403)"""
    ),
]


def main() -> None:
    out = build_notebook(CELLS, OUT_PATH)
    print(f"wrote: {out}  ({sum(len(c[1]) for c in CELLS)} chars across {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
