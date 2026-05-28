# Memory in agents

The library ships 7 distinct memory variants. Each represents a different design choice about *what* to remember and *how* to retrieve it.

## At a glance

| Variant | Stored unit | Scope | Architecture(s) | When to reach for it |
|---|---|---|---|---|
| None | — | per-task | [Reflection (nb 01)](../architectures/01_reflection.ipynb), most loop architectures | When quality matters and tasks are independent |
| Episodic (vector) | conversation turns | across runs | [Episodic+Semantic (nb 08)](../architectures/08_episodic_semantic_memory.ipynb), [Reflexion (nb 18)](../architectures/18_reflexion.ipynb) | Personal assistant continuity; lessons from past failures |
| Archive (positive examples) | accepted outputs | across runs | [RLHF Self-Improvement (nb 15)](../architectures/15_rlhf_self_improvement.ipynb) | Quality compounds across similar tasks |
| Semantic graph | (subject, predicate, object) triples | across runs | [Graph Memory (nb 12)](../architectures/12_graph_memory.ipynb), [GraphRAG (nb 27)](../architectures/27_graph_rag.ipynb) | When entity-relations matter for retrieval |
| Skill library | named Python functions | across runs | [Voyager (nb 29)](../architectures/29_voyager.ipynb) | Reusable executable utilities |
| Tiered (OS-style) | facts in context (RAM) + archival (disk) | across runs | [MemGPT (nb 31)](../architectures/31_memgpt.ipynb) | Bounded context with lossless overflow |
| Workflow library | step recipes (strategies) | across runs | [AWM (nb 35)](../architectures/35_agent_workflow_memory.ipynb) | When *strategy* is what transfers, not facts or code |

## How to think about it

Three orthogonal questions:

1. **What is the atomic stored unit?** Text? Code? A triple? A reflection? A recipe?
2. **What's the retrieval key?** Semantic similarity (vector)? Entity name (graph)? Sequence position (RAM-like)?
3. **What's the persistence scope?** One task? One session? Forever?

A simple decision tree:

```
Need memory at all?
├── No → Reflection / PEV / SelfConsistency
└── Yes
    ├── Stored unit is text + retrieval is similarity?
    │   ├── It's facts / conversation → Episodic+Semantic (nb 08)
    │   ├── It's reflections on failures → Reflexion (nb 18)
    │   ├── It's accepted outputs → RLHF Self-Improvement (nb 15)
    │   └── It's strategy recipes → AWM (nb 35)
    ├── Stored unit is structured (graph triples)?
    │   ├── Just want Q&A over entities → Graph Memory (nb 12)
    │   └── Want global theme queries too → GraphRAG (nb 27)
    ├── Stored unit is executable Python?
    │   └── → Voyager (nb 29)
    └── Context window is the bottleneck?
        └── → MemGPT (nb 31) — context tier + archival overflow
```

## Implementation choices in this library

- **Vector backend**: defaults to FAISS (in-process). Swap to Chroma or Qdrant via `pip install` extras.
- **Graph backend**: defaults to NetworkX (in-process). Swap to Neo4j by setting `GRAPH_BACKEND=neo4j` in `.env`.
- **Persistence to disk**: NOT enabled by default — memory lives in the architecture instance. Persist by serializing `arch.episodic.episodes` / `arch.skills` / etc.

## Combining variants

The architectures aren't mutually exclusive. A natural composition:

- **Reflexion + Voyager**: Reflexion stores "lessons from failure"; Voyager stores "successful code". Use both — failures inform what to avoid; skills are what to reuse.
- **MemGPT + GraphRAG**: tiered memory for the conversation; graph for the corpus knowledge.
- **AWM + Voyager**: workflows say *what to do*; skills say *how to do it*.

See each architecture's §11 "Extensions" section for specific composition recipes.
