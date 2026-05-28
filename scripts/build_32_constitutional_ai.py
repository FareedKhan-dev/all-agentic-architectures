"""Generate notebooks/32_constitutional_ai.ipynb — critique-and-revise against a constitution."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from notebook_builder import build_notebook, code, md  # noqa: E402

OUT_PATH = Path(__file__).parents[1] / "notebooks" / "32_constitutional_ai.ipynb"

CELLS = [
    md("""# 32 · Constitutional AI — critique + revise against a written constitution

> **TL;DR.** Generate → critique against each rule of the constitution (categorical pass/fail per rule) → **Python AND** over all passes → if any fail, revise; loop.

| Property | Value |
|---|---|
| Origin | Bai et al. (Anthropic), *Constitutional AI* (2022). [arXiv:2212.08073](https://arxiv.org/abs/2212.08073) |
| Picker | Python `all(v['verdict'] == 'pass' for v in verdicts)` — deterministic-picker |
| Cost | 1 gen + N critique-rule calls + (1 revise per loop) |"""),
    md("""## 2 · Architecture

```mermaid
flowchart LR
    A([task]) --> G[Generate] --> C[Critique<br/><sub>per-rule pass/fail</sub>] --> P{All passed?}
    P -->|yes| F[Finalize] --> Z([final])
    P -->|no| R[Revise] --> C
    style C fill:#fff3e0,stroke:#f57c00
    style P fill:#fce4ec,stroke:#c2185b
```"""),
    md("""## 3 · Theory

Per-rule pass/fail is `Literal['pass', 'fail']`. Python does `all(v['verdict'] == 'pass' for v in verdicts)`. No numeric judgement → no flat-scoring pathology — same pattern as CRAG (nb 24) and Self-RAG (nb 25)."""),
    md("""## 4 · Setup"""),
    code("""from agentic_architectures import get_llm, enable_langsmith, settings
from agentic_architectures.architectures import ConstitutionalAI
from agentic_architectures.architectures.constitutional_ai import DEFAULT_CONSTITUTION
from agentic_architectures.ui import print_md, print_header
enable_langsmith()
llm = get_llm(provider="nebius", model="meta-llama/Llama-3.3-70B-Instruct", temperature=0.4)
print_header(f"LLM: {llm.model}")
print()
print('=== DEFAULT_CONSTITUTION ===')
for i, r in enumerate(DEFAULT_CONSTITUTION): print(f'  [{i}] {r}')"""),
    md("""## 7 · Build the graph"""),
    code("""from IPython.display import Image, display
arch = ConstitutionalAI(llm=llm, max_iterations=2)
graph = arch.build()
try: display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"(PNG unavailable: {e})")
    print(graph.get_graph().draw_mermaid())"""),
    md("""## 8 · Live run — a prompt designed to violate rules

We pick a prompt that tempts the LLM to be verbose, opinionated, and confident-without-citation — likely failing rules 0, 1, and 2 on first generation."""),
    code("""TASK = (
    "In a 5-paragraph rant, share your personal opinion about which Python web framework is "
    "objectively best. Cite no sources; just argue. Use strong claims."
)

r = arch.run(TASK)
print(f"ITERATIONS: {r.metadata['iterations']}")
print(f"ALL_PASSED: {r.metadata['all_passed']}")
print(f"N_PASS: {r.metadata['n_pass']}/{r.metadata['n_rules']}")
print(f"N_FAIL: {r.metadata['n_fail']}")
print()
print('=== RULE VERDICTS (final) ===')
for v in r.metadata['rule_verdicts']:
    icon = '✓' if v['verdict'] == 'pass' else '✗'
    print(f"  [{v['rule_index']}] {icon} {v['verdict']}: {v['rationale'][:120]}")
print()
print('=== FAILURES (drove revision) ===')
for f in r.metadata['failures']:
    print(f"  - {f[:120]}")
print()
print('=== FINAL ANSWER ({} chars) ==='.format(len(r.output)))
print(r.output[:600])"""),
    md("""## 9 · What we just observed

*(Automatically tailored from the actual captured run by `scripts/tailor_32_commentary.py`.)*"""),
    md("""## 11 · Failure modes & extensions

| Failure | Mitigation |
|---|---|
| **Rule conflict** | Two rules contradict on a task | Order rules by priority; first-fail wins |
| **Critique mis-judges** | Says 'pass' for a violation | Add a second critique pass with stricter prompt |
| **Infinite revise loop** | Each revision violates a different rule | Hard cap on `max_iterations`; surface unresolved failures |

Extensions: (1) per-rule severity weights (warning vs error), (2) RL fine-tuning from synthetic CAI critiques (Anthropic's full pipeline), (3) hierarchical constitutions (general principles + domain-specific rules).

Reference: Bai et al., *Constitutional AI*. 2022. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)"""),
]

def main():
    out = build_notebook(CELLS, OUT_PATH)
    print(f"wrote: {out}  ({sum(len(c[1]) for c in CELLS)} chars across {len(CELLS)} cells)")

if __name__ == "__main__": main()
