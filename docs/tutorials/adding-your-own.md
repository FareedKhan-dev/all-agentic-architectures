# Add your own architecture

A 5-step recipe for getting a new pattern into the library + notebook + tests + docs. Mirrors how every architecture in the catalogue was built.

## 1. Write the architecture class

Create `src/agentic_architectures/architectures/<your_name>.py`:

```python
"""<one-line description>.

Origin: <citation>
"""

from __future__ import annotations
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


# 1. State schema
class YourState(TypedDict, total=False):
    task: str
    # ... your state fields
    history: Annotated[list[dict[str, Any]], operator.add]
    final_output: str


# 2. Pydantic schemas (one per structured-output LLM call)
class _YourSchema(BaseModel):
    answer: str = Field(description="...")
    # use Literal[...] / bool / int with bounds for any deciding signals — NOT a numeric score


# 3. The architecture
class YourArchitecture(Architecture):
    name = "your_architecture"
    description = "..."
    reference = "https://arxiv.org/abs/..."

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # ... bind structured-output runnables here
        self._solver = self.llm.with_structured_output(_YourSchema)

    def _your_node(self, state: YourState) -> dict[str, Any]:
        # ... node logic
        return {"final_output": "...", "history": [{"stage": "your_node"}]}

    def build(self) -> Any:
        g: StateGraph = StateGraph(YourState)
        g.add_node("your_node", self._your_node)
        g.add_edge(START, "your_node")
        g.add_edge("your_node", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke({"task": task}, config={"recursion_limit": 25})
        return ArchitectureResult(
            output=final_state.get("final_output", ""),
            trace=final_state.get("history", []),
            metadata={...},   # your numeric/boolean signals for §9 commentary
        )
```

## 2. Register in `__init__.py`

```python
# src/agentic_architectures/architectures/__init__.py
from agentic_architectures.architectures.your_name import YourArchitecture

__all__ = [
    ...,
    "YourArchitecture",   # alphabetical
]
```

## 3. Smoke-test before papermilling

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c \
  "from agentic_architectures.architectures import YourArchitecture; \
   r = YourArchitecture().run('test'); print(r.metadata); print(r.output)"
```

## 4. Write the notebook builder

Use `scripts/build_NN_<existing>.py` (e.g. `build_15_rlhf.py`) as a starting template. Generate the canonical 11-section structure:

1. Title + TL;DR + properties table
2. Mermaid architecture diagram
3. Theory
4. Setup
5. Library walkthrough
6. State schema
7. Build the graph (with PNG render)
8. Live run (with machine-readable `print(f"KEY=value")` for tailor)
9. **PLACEHOLDER** — `## 9 · What we just observed` (tailor will fill)
10. Try other providers / variations
11. Failure modes + safety + extensions + references

Build it:

```bash
.venv/Scripts/python.exe scripts/build_NN_your_name.py
```

## 5. Execute + tailor

```bash
.venv/Scripts/python.exe -m papermill notebooks/NN_your_name.ipynb \
                                       notebooks/NN_your_name.ipynb \
                                       --kernel python3

.venv/Scripts/python.exe scripts/tailor_NN_commentary.py
```

The tailor regex-parses your machine-readable `print()` lines and rewrites §9 with concrete numbers. See `scripts/tailor_15_commentary.py` for the template.

## 6. Add tests

The test framework auto-picks up your architecture via the registry sweep:

```python
# tests/unit/test_registry.py — already parametrized over A.__all__
# So your class gets 3 free tests: metadata / instantiate / build.
```

If your architecture has constructor args beyond `llm=`, add a row to `EXTRA_KWARGS` in `test_registry.py`.

Optional but recommended: a dedicated unit test in `tests/unit/test_pure_python.py` for any pure-Python helpers (deterministic-picker functions).

And one integration test:

```python
# tests/integration/test_integration_all.py
def test_your_architecture_real() -> None:
    from agentic_architectures.architectures import YourArchitecture
    arch = YourArchitecture(llm=_llm())
    r = arch.run("...")
    assert r.output
```

## 7. Add to the benchmark

```yaml
# benchmarks/tasks.yaml
- id: your_task
  kind: ...
  prompt: |
    ...
  expected_contains: ["..."]
  architectures:
    - YourArchitecture
```

Run `python benchmarks/run_benchmark.py --only YourArchitecture` to confirm the cell renders.

## 8. Add to docs nav

```yaml
# mkdocs.yml
nav:
  - Architectures:
      - "Phase N — ...":
          - architectures/NN_your_name.ipynb
```

That's it. The mkdocs-jupyter plugin will render your notebook as a docs page with the same structure as every other architecture — no separate MD file needed.
