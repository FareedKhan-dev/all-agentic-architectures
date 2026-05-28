"""Generate notebooks/30_storm.ipynb — multi-perspective research → outline → article."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from notebook_builder import build_notebook, code, md  # noqa: E402

OUT_PATH = Path(__file__).parents[1] / "notebooks" / "30_storm.ipynb"

CELLS = [
    md("""# 30 · STORM — multi-perspective research → outline → article

> **TL;DR.** 5-stage pipeline: brainstorm perspectives → questions per perspective → answer each → outline → write article section by section. Composes [Multi-Agent (nb 05)](./05_multi_agent.ipynb) + [Planning (nb 04)](./04_planning.ipynb).

| Property | Value |
|---|---|
| Origin | Shao et al., *STORM* (Stanford 2024). [arXiv:2402.14207](https://arxiv.org/abs/2402.14207) |
| Stages | perspectives → questions → answer → outline → write |
| Cost | 1 + N + (N×K) + 1 + S where N=perspectives, K=questions/p, S=sections |
"""),
    md("""## 2 · Architecture

```mermaid
flowchart LR
    A([topic]) --> P[PERSPECTIVES] --> Q[QUESTIONS] --> AN[ANSWER] --> O[OUTLINE] --> W[WRITE] --> Z([article])
    style P fill:#e3f2fd,stroke:#1976d2
    style W fill:#e8f5e9,stroke:#388e3c
```"""),
    md("""## 3 · Theory

Plain article generation has one LLM "write about X" — produces shallow, single-perspective text. STORM forces breadth through *perspective diversity* and *atomic Q&A research* before writing.

The deciding signals are all structured-output lists (perspectives, questions, sections). No numeric scoring → no flat-scoring risk."""),
    md("""## 4 · Setup"""),
    code("""from agentic_architectures import get_llm, enable_langsmith, settings
from agentic_architectures.architectures import STORM
from agentic_architectures.tools import web_search_tool
from agentic_architectures.ui import print_md, print_header
enable_langsmith()
llm = get_llm(provider="nebius", model="meta-llama/Llama-3.3-70B-Instruct", temperature=0.4)

# Real Tavily web search — every question in the ANSWER stage hits the live web.
_tavily = web_search_tool(max_results=3)
def web_search_fn(query: str) -> list[str]:
    try:
        result = _tavily.invoke(query)
        if isinstance(result, list):
            return [str(r.get('content', r))[:400] for r in result]
        return [str(result)[:1000]]
    except Exception as e:
        return [f"(web search unavailable: {e})"]

print_header(f"LLM: {llm.model}  ·  Web research: Tavily (real)")"""),
    md("""## 7 · Build the graph"""),
    code("""from IPython.display import Image, display
arch = STORM(llm=llm, n_perspectives=3, questions_per_perspective=2, web_search_fn=web_search_fn)
graph = arch.build()
try: display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"(PNG unavailable: {e})")
    print(graph.get_graph().draw_mermaid())"""),
    md("""## 8 · Live run — article on agentic AI in 2024

This will take 1-2 minutes (multiple LLM calls per stage)."""),
    code("""TOPIC = "The rise of agentic AI architectures in 2024"
r = arch.run(TOPIC)
print(f"N_PERSPECTIVES: {r.metadata['n_perspectives']}")
print(f"N_QUESTIONS: {r.metadata['n_questions']}")
print(f"N_SECTIONS: {r.metadata['n_sections']}")
print(f"ARTICLE_CHARS: {r.metadata['article_chars']}")
print()
print("=== PERSPECTIVES ===")
for i, p in enumerate(r.metadata['perspectives'], 1):
    print(f"  [{i}] {p}")
print()
print("=== QUESTIONS ===")
for i, q in enumerate(r.metadata['questions'], 1):
    print(f"  [{i}] {q['question'][:120]}")
print()
print("=== OUTLINE ===")
for s in r.metadata['outline']:
    print(f"  ## {s['title']}")
    for kp in s['key_points'][:3]:
        print(f"    - {kp[:100]}")
print()
print("=== ARTICLE (first 1500 chars) ===")
print(r.output[:1500])"""),
    md("""## 9 · What we just observed

*(Automatically tailored from the actual captured run by `scripts/tailor_30_commentary.py`.)*"""),
    md("""## 11 · Failure modes & extensions

| Failure | Mitigation |
|---|---|
| **Perspective overlap** | All N perspectives are paraphrases | Stricter schema; force "must be substantively different from {prior}" |
| **Hallucinated answers** | LLM answering from no source | Add web search (pass `web_search_fn=`) |
| **Section bloat** | Each section repeats prior section's content | Show prior sections to writer for de-dup |
| **Cost** | 1+N+N*K+1+S calls — easily 12+ for default config | Cache; batch via asyncio |

Extensions: (1) web-search backed answers, (2) iterative outline refinement, (3) per-perspective dedicated LLM (Llama for breadth, Qwen-Thinking for depth-sections).

Reference: Shao et al., *STORM*. 2024. [arXiv:2402.14207](https://arxiv.org/abs/2402.14207)"""),
]

def main():
    out = build_notebook(CELLS, OUT_PATH)
    print(f"wrote: {out}  ({sum(len(c[1]) for c in CELLS)} chars across {len(CELLS)} cells)")

if __name__ == "__main__": main()
