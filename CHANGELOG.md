# Changelog

All notable changes to this project will be documented in this file. From `0.2.0` onward this file is maintained automatically by [release-please](https://github.com/googleapis/release-please) from [Conventional Commit](https://www.conventionalcommits.org/) messages.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0](https://github.com/FareedKhan-dev/all-agentic-architectures/compare/v0.2.0...v0.3.0) (2026-05-28)


### Features

* 17 foundational architecture patterns ([c09a801](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/c09a8012854069efe6a7d8f6b4a276be0056ca5d))
* 18 advanced architecture patterns ([af34630](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/af3463027095ea114aa30775e86d6f6a4e63e9f9))
* **benchmarks:** 17-task suite + run_benchmark.py + leaderboard ([ad3fd8d](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/ad3fd8d6a74e364dc4a6602e2f92e08da0165e9e))
* **core:** Architecture ABC + factories + memory/tools/ui scaffolding ([7626a5f](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/7626a5f7bcee31354fe836bbf1bbc095604fd550))
* **data:** Stardust sample corpus for the RAG-family notebooks ([79e27bf](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/79e27bf8ac2e9c7e1469b3b1d81785f1cb58d978))
* **scripts:** execute_notebooks.py driver ([bd54cbe](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/bd54cbefb51d6b8cf54ea8be29bfba6b210d1f17))
* **scripts:** notebook_builder helper for the canonical 11-section template ([5828d91](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/5828d91da5169ac6791031d64a25a30141056124))


### Bug Fixes

* **ci:** relax over-strict ruff config and fix the few real lint hits ([890913f](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/890913f0e7371c263496ad6f3924faba22c19f08))
* **ci:** ruff-format whole tree, remove missing mkdocs custom_dir, give pytest dummy keys ([15e6112](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/15e611237379844910ca41d7441ca426d9c95a23))
* **readme:** make architecture-family bento clickable end-to-end ([2f80758](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/2f80758f9d1d2871454249d2c4c942bd1e7767a9))
* **readme:** use relative MD paths so links work on GitHub immediately ([57e948b](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/57e948bab9dadbcdf86e2124197ef1c62a5dc034))
* **tests:** stub get_embeddings() so unit tests never call a real API ([fad1e23](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/fad1e23a858f9153ae7ec05b686863438790f4d1))


### Documentation

* MkDocs Material site with bespoke dark-only aesthetic ([f4bab7a](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/f4bab7aa92d2c69707b47f832d67f52b31d7b7f9))
* **readme:** modern emoji-free README rewrite ([576957b](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/576957b81080597e341346d05c5b1af4a7a4f189))
* **readme:** premium hero, badge row, stats block, star-history chart ([f50cae1](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/f50cae196ba4afc452c81e0286e4c48259f5e690))
* **scripts:** orient visitors who click into scripts/ ([e4b82fb](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/e4b82fb221680d626f5723f7f6eea79d6eb15cc1))


### Tests

* 320 tests covering all 35 architectures + notebooks + helpers ([35cfeff](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/35cfeff6b1ac5bde6e132d2cff7d1d76072d7a8b))


### Chores

* pre-commit + community docs (CONTRIBUTING, COC, SECURITY, CHANGELOG) ([0c94ba8](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/0c94ba84d32b35b8e4f2b3349b69eb7b83b10eb8))
* remove pre-rebuild notebooks and obsolete requirements.txt ([67dacf6](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/67dacf689ec89bde2bc3b91d1e0dcda9657a2e88))

## [0.2.0] — 2026-05-28

### Features

- **18 new architectures** — Reflexion, Self-Discover, Chain-of-Verification, Self-Consistency, LATS, Agentic RAG, Corrective RAG, Self-RAG, Adaptive RAG, GraphRAG, Multi-Agent Debate, Voyager, STORM, MemGPT, Constitutional AI, SWE-Agent, BrowserAgent, Agent Workflow Memory.
- **The deterministic-picker pattern** documented and applied across every LLM-as-Scorer surface (13 architectures).
- **Real browser control** in `BrowserAgent` (notebook 34) via Playwright + headless Chromium, with a hard Python safety gate (`_check_safety()`).
- **Real subprocess execution** in `Voyager` (notebook 29) — skills run in fresh isolated Python processes with a 5-second timeout.
- **Real web research** in `STORM` (notebook 30) — all per-perspective questions are answered via live Tavily search.
- **Shared `STARDUST_CORPUS`** under `agentic_architectures.data` — fictional 12-document knowledge base used by all five RAG-family notebooks.
- **Benchmark suite** at `benchmarks/tasks.yaml` covering all 35 architectures across 17 task families; runner emits `docs/benchmarks.md` leaderboard.

### Tests

- `tests/unit/test_registry.py` — parametrized sweep over every registered architecture (106 test instances).
- `tests/notebooks/test_notebook_integrity.py` — static integrity checks for all 35 notebooks (141 instances).
- `tests/unit/test_pure_python.py` — deep coverage of pure-Python helpers (haiku checker, composite scorers, subprocess executor, safety gate, sandbox path resolver).
- `tests/integration/test_integration_all.py` — env-gated real-LLM happy paths for every architecture (37 instances).

### Documentation

- New MkDocs Material site (dark-mode aesthetic, gradient hero, bento family grid) covering: Getting Started, Architectures (35 embedded notebooks), Tutorials, Benchmarks, auto-generated API reference.
- Tutorials: `deterministic-picker.md`, `memory.md`, `adding-your-own.md`.

### CI / Tooling

- GitHub Actions: `ci.yml` (ruff + mypy + pytest matrix on 3.10/3.11/3.12), `docs.yml` (auto-deploy to GitHub Pages on push to main), `notebook-execute.yml` (manual-only re-execute via papermill — no cron), `release.yml` (release-please → PyPI).
- Dependabot grouped weekly updates for langchain/docs/test-and-lint deps + GitHub Actions.
- Issue templates (bug, feature, new-architecture) + PR template with deterministic-picker checklist.

## [0.1.0] — 2025-09 (initial)

- 17 foundational architectures (Reflection, Tool Use, ReAct, Planning, Multi-Agent, PEV, Blackboard, Episodic+Semantic, ToT, Mental Loop, Meta-Controller, Graph Memory, Ensemble, Dry-Run, RLHF Self-Improvement, Cellular Automata, Reflexive Metacognitive).
- Single-provider (Nebius) implementation.
