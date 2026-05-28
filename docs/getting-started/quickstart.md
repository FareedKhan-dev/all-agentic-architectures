# Quickstart

Once you have [installed](installation.md) the package with a provider extra and set your API key in `.env`, here's the smallest end-to-end run:

```python
from agentic_architectures import get_llm
from agentic_architectures.architectures import Reflection

arch = Reflection(llm=get_llm(), max_iterations=2, target_score=8)
result = arch.run("Write a haiku about a glacier.")

print(result.output)
print("score:", result.metadata["final_score"], "/ 10")
print("iterations:", result.metadata["iterations"])
```

Every architecture follows the same contract:

```python
class Architecture(ABC):
    name: str
    description: str
    reference: str

    def build(self) -> CompiledStateGraph: ...
    def run(self, task: str, **kwargs) -> ArchitectureResult: ...
```

And every `.run()` returns:

```python
@dataclass
class ArchitectureResult:
    output: str                          # the user-facing answer
    state: dict[str, Any]                # architecture-specific state snapshot
    trace: list[dict[str, Any]]          # per-node trace events
    metadata: dict[str, Any]             # numeric / boolean signals
```

So you can compose architectures uniformly:

```python
from agentic_architectures.architectures import Reflection, AgenticRAG, ChainOfVerification
from agentic_architectures.data import STARDUST_CORPUS

llm = get_llm()

for cls in (Reflection, AgenticRAG, ChainOfVerification):
    kwargs = {"documents": STARDUST_CORPUS} if cls is AgenticRAG else {}
    arch = cls(llm=llm, **kwargs)
    r = arch.run("What propellant does the Phoenix-2 engine use?")
    print(f"{cls.__name__:25s} → {r.output[:80]}")
```

## Run all 35 architectures end-to-end

Every notebook is fully runnable. From the repo root:

```bash
.venv/Scripts/python.exe -m papermill notebooks/18_reflexion.ipynb /tmp/out.ipynb --kernel python3
```

Or use the build scripts to regenerate them:

```bash
.venv/Scripts/python.exe scripts/build_18_reflexion.py
.venv/Scripts/python.exe -m papermill notebooks/18_reflexion.ipynb notebooks/18_reflexion.ipynb --kernel python3
.venv/Scripts/python.exe scripts/tailor_18_commentary.py
```

## Next

- [Switch LLM providers](providers.md) with one config change
- [Browse all 35 architectures](../architectures/index.md)
- [See the benchmark leaderboard](../benchmarks.md)
- [Add your own architecture](../tutorials/adding-your-own.md)
