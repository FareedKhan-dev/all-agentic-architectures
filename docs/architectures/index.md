# Architectures

Every architecture implements the [`Architecture` ABC](../reference/base.md). Click any row to open the full notebook with live LLM outputs, theory, mermaid diagram, and tailored commentary.

## Reasoning & Reflection

Self-critique loops that drive answer quality up through iteration — and catch hallucinations before they ship.

| Architecture | Pattern | Captured signature |
|---|---|---|
| [Reflection](01_reflection.ipynb) | Generate → critique → refine | 4 iters, scores `9→9→9→8` |
| [Reflexion](18_reflexion.ipynb) | Verbal reflections in episodic memory | trials `[1,3,1]`; memory transferred ✅ |
| [Chain-of-Verification](20_chain_of_verification.ipynb) | Verify each baseline claim independently | Qwen refused hallucination; Llama caught 3 |
| [Self-Discover](19_self_discover.ipynb) | SELECT → ADAPT → IMPLEMENT → SOLVE | 5 modules / 4-step plan / correct answer |
| [Constitutional AI](32_constitutional_ai.ipynb) | Per-rule pass/fail → Python AND → revise | 1/4 passed after 2 iters |

## Sampling & Search

When a single chain of thought isn't enough — draw many, score them, or grow a real search tree.

| Architecture | Pattern | Captured signature |
|---|---|---|
| [Self-Consistency](21_self_consistency.ipynb) | Sample N paths, majority-vote | 7/7 unanimous; baseline 6/7 |
| [Tree of Thoughts](09_tree_of_thoughts.ipynb) | Beam search over thoughts | 16-thought tree, scores `[5,4,4,4,...]` |
| [LATS](22_lats.ipynb) | MCTS-style tree with reward backup | 9 nodes, value spread `[10,5,5,5,3]` |
| [Mental Loop](10_mental_loop.ipynb) | Simulate → score (deterministic-picker) | LLM `[4,4,4]` → Python `[5,4,4]` |
| [Ensemble](13_ensemble.ipynb) | N voters, majority/weighted aggregation | `YES=2, NO=1` keyword-fallback |

## Retrieval (RAG)

Ground every claim. Five distinct retrieval shapes, each suited to a different failure mode.

| Architecture | Pattern | Captured signature |
|---|---|---|
| [Agentic RAG](23_agentic_rag.ipynb) | Agent decides when & what to retrieve | 4 tasks; arithmetic skipped retrieval ✅ |
| [Corrective RAG](24_corrective_rag.ipynb) | Grade docs, fall back to web | Categorical relevance + Python routing |
| [Self-RAG](25_self_rag.ipynb) | Per-doc reflection tokens | direct 1/4 kept; mismatch 0/4 (admits gap ✅) |
| [Adaptive RAG](26_adaptive_rag.ipynb) | Pre-route by query complexity | 3/3 routing accuracy |
| [GraphRAG](27_graph_rag.ipynb) | KG + community summaries | 5 communities sized `[16,10,9,7,5]` |

## Memory

Learn across `.run()` calls. Pick the storage shape that matches your transfer pattern.

| Architecture | Stored unit | Captured signature |
|---|---|---|
| [Episodic + Semantic](08_episodic_semantic_memory.ipynb) | Conversation turns + triples | 11 triples / 6 facts recalled |
| [Graph Memory](12_graph_memory.ipynb) | (subject, predicate, object) triples | 18 triples / 5/5 Q&A |
| [MemGPT](31_memgpt.ipynb) | OS-style context + archival tiers | 5 turns; eviction triggered |
| [Voyager](29_voyager.ipynb) | Reusable Python skills (real subprocess) | 3 tasks; factorial reused ✅ |
| [Agent Workflow Memory](35_agent_workflow_memory.ipynb) | High-level workflow recipes | 3 tasks; library grew 0→1→2→3 |

## Tools & Actions

Act on the real world. From a single search tool to a sandboxed code repo to a real Chromium browser.

| Architecture | Pattern | Captured signature |
|---|---|---|
| [Tool Use](02_tool_use.ipynb) | Agent with one tool | 2 queries; Llama runaway capped |
| [ReAct](03_react.ipynb) | Thought → Action → Observation | 3 thoughts / 2 actions |
| [Planning](04_planning.ipynb) | Decompose → execute → replan | 7 steps, 0 replans |
| [Plan-Execute-Verify (PEV)](06_pev.ipynb) | Post-execution verification per step | 2 pass + 1 fail-accepted |
| [SWE-Agent](33_swe_agent.ipynb) | Sandboxed file-system agent | `[read,write,write,read,run_check,answer]` ✅ |
| [BrowserAgent](34_computer_use.ipynb) | **Real Playwright** + safety gate | All 4 evil-phishing.com attempts blocked ✅ |

## Multi-Agent

Coordinate many minds — specialists, adversarial debate, multi-perspective research.

| Architecture | Pattern | Captured signature |
|---|---|---|
| [Multi-Agent](05_multi_agent.ipynb) | Supervisor + specialists | 3/3 specialists contributed |
| [Blackboard](07_blackboard.ipynb) | Shared workspace + agents | 5 rounds, 3/4 contributed |
| [Debate](28_debate.ipynb) | N agents × K rounds | Group-think failure on Sally trick (instructive) |
| [STORM](30_storm.ipynb) | Multi-perspective research → article | 3 perspectives → 6 web-grounded answers |
| [Meta-Controller](11_meta_controller.ipynb) | Router over architectures | 4/4 routes correct |

## Safety & Routing

Block before harm. Route to the right specialist. Make the deciding signal a deterministic Python check, never an LLM number.

| Architecture | Pattern | Captured signature |
|---|---|---|
| [Dry-Run](14_dry_run.ipynb) | Propose → simulate → approval gate | 3 tasks: approve / reviewer / blocked |
| [Reflexive Metacognitive](17_reflexive_metacognitive.ipynb) | Self-aware capability routing | 4 tasks; 3 Python overrides fired |
| [Computer Use](34_computer_use.ipynb) | Categorical actions + Python safety gate | Real Chromium; safety gate fired ✅ |

## Specialty

Patterns with a unique shape — keep these in mind for when the standard families don't fit.

| Architecture | Pattern | Captured signature |
|---|---|---|
| [RLHF Self-Improvement](15_rlhf_self_improvement.ipynb) | Multi-dim deterministic scoring + archive | composite `[8,8,10]` vs LLM `[8,9,8]` |
| [Cellular Automata](16_cellular_automata.ipynb) | LLM rules over a grid | Forest fire spread captured |

---

### Cross-cutting

- **The [deterministic-picker pattern](../tutorials/deterministic-picker.md)** appears in 13 of 35 architectures — the universal escape from the LLM-as-Scorer flat-band pathology.
- **The [memory tutorial](../tutorials/memory.md)** compares all 7 memory variants side by side.
- **[Adding your own architecture](../tutorials/adding-your-own.md)** follows the 5-step recipe.
