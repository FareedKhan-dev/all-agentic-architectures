"""Generate notebooks/27_graph_rag.ipynb — KG + community summaries."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from notebook_builder import build_notebook, code, md  # noqa: E402

OUT_PATH = Path(__file__).parents[1] / "notebooks" / "27_graph_rag.ipynb"


CELLS = [
    md(
        """# 27 · GraphRAG — knowledge graph + community summaries

> **TL;DR.** At build-time: extract (subject, predicate, object) triples from the corpus → NetworkX graph → detect communities → summarise each via LLM. At query-time: classify local vs global → traverse entity neighbourhood OR consult community summaries → answer.
>
> **Reach for it when** the corpus has rich entity-entity relationships AND you have global theme questions ("what are the main topics?") that vector retrieval can't answer well.

| Property | Value |
|---|---|
| Origin | Edge et al., *From Local to Global: GraphRAG* (Microsoft 2024). [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) |
| Backend | NetworkX (default, in-process) or Neo4j (`GRAPH_BACKEND=neo4j`) |
| Community detection | NetworkX greedy modularity |
| Build cost | 1 extract per doc + 1 summarise per community (precomputed once) |
| Query cost | 1 classify + 1 answer (+ optional traversal) |

Builds on [Graph Memory (nb 12)](./12_graph_memory.ipynb) by adding the **community summarisation** layer — that's what enables answering global questions about themes."""
    ),
    md(
        """## 2 · Architecture at a glance

```mermaid
flowchart TB
    subgraph build [Build phase, one-shot]
        D[docs] --> EX[Extract triples]
        EX --> G[(NetworkX KG)]
        G --> CD[Community detection]
        CD --> CS[Summarise each community]
    end
    subgraph query [Query phase]
        Q([query]) --> C[Classify local/global]
        C -->|local| LN[Entity neighbourhood]
        C -->|global| GS[All community summaries]
        LN --> ANS[Answer from context]
        GS --> ANS
    end
    CS -.-> GS
    G -.-> LN

    style EX fill:#fff3e0,stroke:#f57c00
    style CD fill:#fce4ec,stroke:#c2185b
    style CS fill:#e3f2fd,stroke:#1976d2
    style C fill:#e8f5e9,stroke:#388e3c
```"""
    ),
    md(
        """## 3 · Theory

### 3.0 · Why community summaries

Plain RAG retrieves top-k similar snippets — great for "what is X?" but useless for "what are the main themes across the corpus?". The themes don't live in any single document.

GraphRAG's fix: cluster related entities (via graph community detection), summarise each cluster with the LLM at build-time, then for global questions feed all summaries to the answer-LLM. The summaries are pre-computed; no per-query cost beyond the answer.

### 3.1 · Why pre-classify local vs global

| Query type | Context source | Why |
|---|---|---|
| Local ("who is X?") | Entity neighbourhood (depth-2 subgraph) | Tight, specific facts about the named entity |
| Global ("what are the themes?") | All community summaries | Themes live in the community structure, not in any one doc |

A 3-way categorical classifier could also handle "mixed" — for the demo we keep two buckets for clarity.

### 3.2 · Where this sits in the RAG family

| Pattern | Indexing structure |
|---|---|
| Plain RAG | Flat vector index |
| [Agentic RAG (nb 23)](./23_agentic_rag.ipynb) | Flat vector index (agent decides when) |
| [Self-RAG (nb 25)](./25_self_rag.ipynb) | Flat vector index (per-doc reflection) |
| **GraphRAG (this nb)** | **Knowledge graph with community summaries** |
| [Graph Memory (nb 12)](./12_graph_memory.ipynb) | Knowledge graph (no community summaries) |"""
    ),
    md("""## 4 · Setup"""),
    code(
        """from agentic_architectures import get_llm, enable_langsmith, settings
from agentic_architectures.architectures import GraphRAG
from agentic_architectures.data import STARDUST_CORPUS
from agentic_architectures.ui import print_md, print_header

enable_langsmith()
llm = get_llm(provider="nebius", model="meta-llama/Llama-3.3-70B-Instruct", temperature=0.2)
print_header(f"LLM: {llm.model}  ·  Corpus: {len(STARDUST_CORPUS)} docs")"""
    ),
    md("""## 5 · Library walkthrough"""),
    code(
        """from agentic_architectures.architectures.graph_rag import _QuestionScope, _IngestionTriple
import json
print('--- _QuestionScope schema ---')
print(json.dumps(_QuestionScope.model_json_schema(), indent=2)[:400] + '...')
print()
print('--- _IngestionTriple schema ---')
print(json.dumps(_IngestionTriple.model_json_schema(), indent=2)[:400] + '...')"""
    ),
    md(
        """## 7 · Build the graph + communities

**This cell does the one-time KG build** — extracts triples from each doc, detects communities, summarises each. Expect ~1-2 minutes for a 12-doc corpus."""
    ),
    code(
        """import time
t0 = time.time()
arch = GraphRAG(llm=llm, documents=STARDUST_CORPUS, max_communities=5)
print(f'BUILD_ELAPSED: {time.time()-t0:.1f}s')
print(f'N_COMMUNITIES: {len(arch.communities)}')
print(f'COMMUNITY_SIZES: {[len(c) for c in arch.communities]}')
print()
print('=== COMMUNITY SUMMARIES ===')
for i, s in enumerate(arch.community_summaries):
    print(f'[community {i}, {len(arch.communities[i])} entities]')
    print(f'  {s[:400]}')
    print()"""
    ),
    code(
        """from IPython.display import Image, display
graph = arch.build()
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"(PNG render unavailable: {e}; see § 2)")
    print(graph.get_graph().draw_mermaid())"""
    ),
    md(
        """## 8 · Live run — local vs global queries"""
    ),
    code(
        """TASKS = [
    ("local",  "Who founded Stardust Aerospace and where is it located?"),
    ("global", "What are the main topics / themes covered in the Stardust Aerospace knowledge base?"),
]

for tag, q in TASKS:
    r = arch.run(q)
    print(f"TASK_TAG: {tag}")
    print(f"  TASK: {q}")
    print(f"  SCOPE: {r.metadata['scope']}")
    print(f"  TARGET_ENTITIES: {r.metadata['target_entities']}")
    print(f"  CONTEXT_CHARS: {r.metadata['context_chars']}")
    print(f"  N_COMMUNITIES: {r.metadata['n_communities']}")
    print(f"  COMMUNITY_SIZES: {r.metadata['community_sizes']}")
    print(f"  FINAL_ANSWER: {r.output[:300]}")
    print()"""
    ),
    md(
        """## 9 · What we just observed

*(Automatically tailored from the actual captured run by `scripts/tailor_27_commentary.py`.)*"""
    ),
    md(
        """## 11 · Failure modes & extensions

| Failure | Mitigation |
|---|---|
| **Triple extraction misses key relations** | LLM might extract `Jin-ho Park founded_by Stardust` but skip `Jin-ho Park is CTO` | Multi-pass extraction; explicit role-fact prompt |
| **No communities found** | Graph too sparse (few cross-entity edges) | Lower modularity threshold; cluster on similarity |
| **Community summaries too vague** | Summariser sees too many disconnected facts | Limit to top-K facts per community by centrality |
| **Build cost** | LLM call per doc + per community | Cache extractions; precompute communities offline |

Extensions: (1) hierarchical communities (paper's full recipe), (2) entity-linking dedup (same entity under multiple names), (3) Cypher queries for advanced traversal.

Reference: Edge et al., *GraphRAG.* Microsoft 2024. [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)"""
    ),
]


def main() -> None:
    out = build_notebook(CELLS, OUT_PATH)
    print(f"wrote: {out}  ({sum(len(c[1]) for c in CELLS)} chars across {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
