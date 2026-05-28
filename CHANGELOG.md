# Changelog

All notable changes to this project will be documented in this file. From `0.2.0` onward this file is maintained automatically by [release-please](https://github.com/googleapis/release-please) from [Conventional Commit](https://www.conventionalcommits.org/) messages.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
