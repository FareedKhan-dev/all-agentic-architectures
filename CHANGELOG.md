# Changelog

All notable changes to this project will be documented in this file. From `0.2.0` onward this file is maintained automatically by [release-please](https://github.com/googleapis/release-please) from [Conventional Commit](https://www.conventionalcommits.org/) messages.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0](https://github.com/FareedKhan-dev/all-agentic-architectures/compare/v0.3.0...v1.0.0) (2026-11-16)


### ⚠ BREAKING CHANGES

* **core:** passing an option an architecture does not take raises TypeError.
* **llm:** requires langchain-core, langchain and langgraph 1.x. The default vector store is the in-memory one, and LLM_MODEL no longer defaults to Llama 3.3 70B for every provider.

### Features

* add DeepSeek provider + Mixture of Agents (MoA) architecture ([#18](https://github.com/FareedKhan-dev/all-agentic-architectures/issues/18)) ([377f57e](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/377f57e29609565030836db2f20709944bb7c59c))
* **notebooks:** notebook 36, Mixture of Agents, built and executed ([14419ce](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/14419cefe3c92224d5f175626ef6ddd980cdbd5d))
* **tools:** SEARCH_BACKEND=mcp searches through a keyless MCP server ([db5721b](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/db5721baaeeb256032a65212a936bedf33fda7be))
* **tools:** SEARCH_MCP_API_KEY for an MCP search server's key ([53b887a](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/53b887a6083dd009aff7e93f8ca9d1f8f377af76))


### Bug Fixes

* **agents:** MemGPT and SWEAgent require the arguments of their actions ([5458d90](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/5458d9085704576801fead51ffaf3936e44747cd))
* **agents:** SWEAgent and MemGPT tell the model what is missing instead of failing quietly ([347cf6c](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/347cf6c131469a7eb3446a0dad54eab91272832c))
* **config:** an empty value in .env means not set ([f20e476](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/f20e476c8d90b72e2a76e4944c5d80305f26c9ab))
* **core:** an unknown architecture option is an error, not ignored ([85b37b9](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/85b37b9388b6e301d9d0cafaa9434645655d3ae4))
* import the LLM factory lazily, which the settings' provider lookup needs ([82979a7](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/82979a7621dcfc75b8472d74edcf8e1248c064d8))
* **llm:** each provider gets its own default model, and Google, Ollama and embeddings work ([401a423](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/401a423cec0a9eb0ef7ac8f2dfd5699655df25cd))
* **llm:** give Ollama a 16k context window by default ([9c807e0](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/9c807e0adefffd05921c41a930a8fa7e0df8824b))
* **llm:** Ollama supports structured output ([21ec15a](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/21ec15a340c23c732c2f92b0a5aee69f6f3c4bdc))
* **llm:** replace the default models providers have retired ([7ec1bf9](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/7ec1bf9ea225fc51df608f6f338998696de1f139))
* **moa:** each layer reads only the layer before it, as in the paper ([d970124](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/d970124702f5332aac790038733017637f8fcaf7))
* **notebooks:** stop hardcoding Nebius models that have been retired ([a7d7d14](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/a7d7d144dc30236f54aac160811665873edb8dc8))


### Documentation

* Mixture of Agents in the listings, keyless search and a free local path in the README ([f727f65](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/f727f65d52b0078981dc097cdc9fc52e5e789be9))
* sync the site's notebook copies with notebooks/ ([18586d9](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/18586d92b8ca782dd27aaadc11c1b3fa87d83e76))
* the November 2026 models, keyless search, OpenRouter and OpenAI-compatible servers ([126b127](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/126b127c19a4e14203badcb7ff6c8cb27a98c3a7))
* the test numbers, and what the real-model sweep showed ([7649699](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/7649699ada388a0c2ee471f4cd8265310280651c))


### Tests

* **integration:** five tests that could not fail or could not pass ([d1692ce](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/d1692ced00f9b7a1e3881793c381de450add17b7))
* **integration:** INTEGRATION_LLM_KWARGS passes extra model arguments to the sweep ([e936899](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/e936899038121cfa8868c93ca5542d461f9a535c))
* **integration:** the CellularAutomata grid is one string literal ([70b8230](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/70b82304d0c2b7ee4de561b8f2dbd0246a858432))
* providers, embeddings fallback, the in-memory store, search backends and MCP tools ([9b8c13f](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/9b8c13fc7997597bb010ac1ebc36e4b7d40f54e8))
* the integration sweep retrieves with the real embeddings model ([56b66b8](https://github.com/FareedKhan-dev/all-agentic-architectures/commit/56b66b8848997d0e6353c0c477a22a0632328500))

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
