# Contributing

Thanks for your interest in `agentic-architectures`! There are three good ways to contribute:

1. **Add a new architecture** — follow the 5-step recipe.
2. **Improve an existing one** — bug fix, prompt tuning, performance, scoring rubric.
3. **Improve infrastructure** — tests, docs, CI, benchmark tasks.

---

## Development setup

```bash
git clone https://github.com/FareedKhan-dev/all-agentic-architectures.git
cd all-agentic-architectures

python -m venv .venv
.venv\Scripts\activate              # Windows
source .venv/bin/activate           # macOS / Linux

pip install -e ".[dev,test,docs,nebius,faiss,tavily,networkx]"
```

Create `.env` with your provider key (at minimum):

```ini
LLM_PROVIDER=nebius
LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct
NEBIUS_API_KEY=sk-...
TAVILY_API_KEY=tvly-...        # optional, for nb 02/03/05/24/30
LANGSMITH_API_KEY=ls-...       # optional, for tracing
```

Smoke test:

```bash
python -c "from agentic_architectures import get_llm, settings; print(settings.llm_provider, settings.llm_model)"
```

---

## Tests

Three buckets:

```bash
# Fast (always run): unit + notebook integrity. No LLM cost.
pytest -q

# Real LLM happy paths (env-gated, ~$0.50-1 per full run).
RUN_INTEGRATION=1 pytest tests/integration -v

# Benchmark suite — full leaderboard refresh (~$1-2, ~25 min).
python benchmarks/run_benchmark.py
```

CI runs the fast bucket on every PR.

---

## Adding a new architecture (5-step recipe)

Full walkthrough at [docs/tutorials/adding-your-own.md](docs/tutorials/adding-your-own.md). Summary:

1. **Library class** — `src/agentic_architectures/architectures/<your_name>.py` implementing `Architecture`.
2. **Register** in `src/agentic_architectures/architectures/__init__.py`.
3. **Smoke test** with a one-liner.
4. **Build script** — `scripts/build_NN_<name>.py` using the canonical 11-section template.
5. **Tailor script** — `scripts/tailor_NN_commentary.py` to rewrite §9 from captured output.

After that:
- Re-execute via `python -m papermill notebooks/NN_*.ipynb notebooks/NN_*.ipynb --kernel python3`
- Add to `mkdocs.yml` nav under the right family
- Add a row to `benchmarks/tasks.yaml` if it solves a benchmark task
- Add an integration test in `tests/integration/test_integration_all.py`

The `tests/unit/test_registry.py` sweep covers your class automatically once it's registered.

---

## The deterministic-picker discipline

**Whenever your architecture has an LLM-as-Scorer surface — a step that ranks, scores, or selects — apply the deterministic-picker pattern.** Full details in [docs/tutorials/deterministic-picker.md](docs/tutorials/deterministic-picker.md). Short version:

- LLM commits to *categorical* features (booleans, `Literal[...]`, ints with bounds) — never a single numeric score.
- **Python** composes the deciding signal from those features.
- The LLM's numeric output (if any) lives on the trace for comparison, never as the deciding value.

This is the central technical pattern of the repo. Code reviews will check for it.

---

## Commit messages

We follow [Conventional Commits](https://www.conventionalcommits.org/) so [release-please](https://github.com/googleapis/release-please) can auto-generate the CHANGELOG. Examples:

```
feat(reflexion): persist EpisodicMemory across processes
fix(lats): handle empty leaf set in backup phase
docs: add CRAG / Self-RAG comparison
test(integration): add real-LLM happy path for AWM
chore(deps): bump langgraph to 0.2.60
```

---

## Code style

- **Ruff** — `ruff check src/ tests/` and `ruff format --check src/ tests/` must pass.
- **Mypy** — `mypy src/agentic_architectures` should be clean; advisory in CI but expected for new code.
- **Docstrings** — Google style. The first line is the one-sentence summary used by mkdocstrings on the docs site.

---

## Code of Conduct

This project follows the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md). By participating you agree to its terms.
