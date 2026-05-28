"""Generate notebooks/29_voyager.ipynb — persistent skill library."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from notebook_builder import build_notebook, code, md  # noqa: E402

OUT_PATH = Path(__file__).parents[1] / "notebooks" / "29_voyager.ipynb"


CELLS = [
    md(
        """# 29 · Voyager — persistent skill library with real subprocess execution

> **TL;DR.** Each task: vector-search the library for a relevant skill; if found, reuse it; otherwise write a new Python function and store it. **The skill code is actually executed in a fresh isolated Python subprocess** — the LLM's predicted result is also captured for comparison.

| Property | Value |
|---|---|
| Origin | Wang et al., *Voyager* (2023). [arXiv:2305.16291](https://arxiv.org/abs/2305.16291) |
| Skill = | named Python function + docstring + example invocation |
| Index | vector store over skill descriptions |
| Persistence | `arch.skills: list[dict]` instance attribute |
| **Execution** | `subprocess.run([sys.executable, '-I', '-c', script], timeout=5)` — fresh isolated interpreter |
| Cost | 1 retrieve + 1 decide + (1 write OR 1 apply) ≈ 3 LLM calls + 1 subprocess per task |

Each skill is run in a **fresh isolated Python subprocess** (`-I` flag = no env vars, no user site, no PYTHONPATH). 5-second timeout caps runaway code. The LLM's predicted result is preserved on the trace so we can compare prediction vs reality."""
    ),
    md(
        """## 2 · Architecture at a glance

```mermaid
flowchart LR
    A([task]) --> R[Retrieve top skill]
    R --> D{Reuse or write?}
    D -->|reuse| AE[Apply existing skill]
    D -->|write_new| W[Write new skill<br/><sub>store in library + index</sub>]
    W --> AN[Apply new skill]
    AE --> Z([answer])
    AN --> Z

    L[(skill library<br/>persists across runs)]
    R <-.search.-> L
    W -.add.-> L

    style D fill:#fff3e0,stroke:#f57c00
    style W fill:#fce4ec,stroke:#c2185b
    style L fill:#f3e5f5,stroke:#7b1fa2
```"""
    ),
    md(
        """## 3 · Theory

### 3.0 · Why a library

Reflexion (nb 18) accumulates *verbal lessons* about past failures. Voyager accumulates *reusable code* for past successes. Both are episodic memory variants; Voyager's value is that the stored artefact is *executable*.

### 3.1 · Why the decider is categorical

`_SkillDecision.action: Literal['reuse', 'write_new']` — categorical, not numeric. Routing is `if action == 'reuse'`. No flat-scoring pathway."""
    ),
    md("""## 4 · Setup"""),
    code(
        """from agentic_architectures import get_llm, enable_langsmith, settings
from agentic_architectures.architectures import Voyager
from agentic_architectures.ui import print_md, print_header
enable_langsmith()
llm = get_llm(provider="nebius", model="meta-llama/Llama-3.3-70B-Instruct", temperature=0.2)
print_header(f"LLM: {llm.model}")"""
    ),
    md("""## 5 · Library walkthrough"""),
    code(
        """from agentic_architectures.architectures.voyager import _NewSkillSpec, _SkillDecision
import json
print('--- _SkillDecision ---')
print(json.dumps(_SkillDecision.model_json_schema(), indent=2)[:300] + '...')
print()
print('--- _NewSkillSpec ---')
print(json.dumps(_NewSkillSpec.model_json_schema(), indent=2)[:500] + '...')"""
    ),
    md("""## 7 · Build the graph"""),
    code(
        """from IPython.display import Image, display
arch = Voyager(llm=llm)
graph = arch.build()
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"(PNG render unavailable: {e}; see § 2)")
    print(graph.get_graph().draw_mermaid())"""
    ),
    md(
        """## 8 · Live run — 3 sequential tasks, library grows

We deliberately pose two similar tasks (factorial of 5, factorial of 7) so Voyager reuses, then a new task (Fibonacci) that forces a new skill."""
    ),
    code(
        """TASKS = [
    ("factorial_5",  "Compute the factorial of 5. Return just the integer."),
    ("factorial_7",  "Compute the factorial of 7. Return just the integer."),
    ("fibonacci_8",  "Compute the 8th Fibonacci number (F(0)=0, F(1)=1). Return just the integer."),
]
EXPECTED = {"factorial_5": "120", "factorial_7": "5040", "fibonacci_8": "21"}

for tag, q in TASKS:
    r = arch.run(q)
    exp = EXPECTED[tag]
    match = exp.strip().lower() in r.output.strip().lower()
    print(f"TASK_TAG: {tag}")
    print(f"  TASK: {q}")
    print(f"  DECISION: {r.metadata['decision']}")
    print(f"  SKILL_NAME: {r.metadata['skill_used_name']}")
    print(f"  INVOCATION: {r.metadata['invocation']}")
    print(f"  LIBRARY_SIZE: {r.metadata['library_size_before']} -> {r.metadata['library_size_after']}")
    print(f"  EXECUTION_OK: {r.metadata['execution_ok']}")
    print(f"  EXECUTED_STDOUT: {r.metadata['executed_stdout']!r}")
    print(f"  LLM_PREDICTED: {r.metadata['llm_predicted']!r}")
    print(f"  ANSWER: {r.output[:100]}")
    print(f"  EXPECTED: {exp}")
    print(f"  MATCH: {match}")
    print()
print(f"FINAL_LIBRARY_SIZE: {len(arch.skills)}")
for s in arch.skills:
    print(f"  - skill `{s['name']}`: {s['description']}")"""
    ),
    md(
        """## 9 · What we just observed

*(Automatically tailored from the actual captured run by `scripts/tailor_29_commentary.py`.)*"""
    ),
    md("""## 11 · Failure modes & extensions

| Failure | Mitigation |
|---|---|
| **Bad skill reused** | Wrong skill retrieved; decider says reuse anyway | Strengthen decider prompt; require code-level check |
| **Skill library bloat** | Many near-duplicate skills | Periodic dedup on description embeddings |
| **Code doesn't actually work** | LLM-written skill has a bug | Sandbox exec + test before storing |

Extensions: (1) sandboxed exec of skills (we predict results via LLM, not actually run code), (2) skill-versioning when a re-written skill supersedes an old one, (3) skill composition (call skill A from skill B).

Reference: Wang et al., *Voyager*. 2023. [arXiv:2305.16291](https://arxiv.org/abs/2305.16291)"""
    ),
]


def main() -> None:
    out = build_notebook(CELLS, OUT_PATH)
    print(f"wrote: {out}  ({sum(len(c[1]) for c in CELLS)} chars across {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
