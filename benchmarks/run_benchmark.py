"""Run the benchmark task suite across architectures, write docs/benchmarks.md.

Usage:
    python benchmarks/run_benchmark.py --dry-run    # Plan only
    python benchmarks/run_benchmark.py --quick      # 1 arch per task smoke
    python benchmarks/run_benchmark.py              # Full matrix
    python benchmarks/run_benchmark.py --only Reflection,SelfDiscover

Reads `benchmarks/tasks.yaml`, instantiates each architecture, runs
`.run(prompt)` (with optional `setup_prompts` for stateful arches), scores
by substring + optional metadata checks, writes Markdown leaderboard.
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import operator as op
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import yaml


HERE = Path(__file__).parent
REPO = HERE.parent
TASKS_PATH = HERE / "tasks.yaml"
OUT_PATH = REPO / "docs" / "benchmarks.md"


# Per-architecture default constructor kwargs (used for all tasks unless overridden per-task).
DEFAULT_KWARGS: dict[str, dict[str, Any]] = {
    "Reflection":        {"max_iterations": 2, "target_score": 7},
    "SelfConsistency":   {"n_samples": 3, "sample_temperature": 0.7},
    "ChainOfVerification": {},
    "Debate":            {"n_agents": 2, "n_rounds": 2},
    "Ensemble":          {"aggregation": "majority_vote"},
    "TreeOfThoughts":    {"branching": 2, "beam_width": 2, "max_depth": 2},
    "LATS":              {"max_iterations": 2, "branching": 2, "max_depth": 2},
    "MentalLoop":        {"branching": 2, "max_steps": 2},
    "Reflexion":         {"max_trials": 2},
    "MemGPT":            {"context_limit": 3, "max_iterations": 3},
    "ConstitutionalAI":  {"max_iterations": 1},
    "BrowserAgent":      {"max_iterations": 3, "headless": True,
                          "blocked_domains": ["evil-phishing.com"]},
    "ComputerUse":       {"max_iterations": 3,
                          "initial_screen": {"url": "about:blank", "elements": [], "fields": {}, "submitted": False},
                          "blocked_domains": ["evil-phishing.com"]},
    "STORM":             {"n_perspectives": 2, "questions_per_perspective": 1},
    "GraphRAG":          {"max_communities": 3},
    "AgenticRAG":        {"max_iterations": 3, "top_k": 3},
    "CorrectiveRAG":     {"top_k": 3},
    "SelfRAG":           {"top_k": 3},
    "AdaptiveRAG":       {"top_k": 3},
    "CellularAutomata":  {
        "rule_prompt": "A 'tree' cell next to a 'fire' cell becomes 'fire'. 'fire' becomes 'ash' next step. 'ash' stays 'ash'. 'empty' stays 'empty'.",
        "allowed_states": ["empty", "tree", "fire", "ash"],
        "height": 3,
        "width": 3,
        "max_steps": 2,
    },
}


_BOOL_LITERALS = {"True": True, "False": False, "true": True, "false": False, "None": None}


def _parse_rhs(rhs: str, actual: Any) -> Any:
    """Parse RHS of a comparison to match `actual`'s type. Handles bool literals."""
    rhs = rhs.strip()
    if rhs in _BOOL_LITERALS:
        return _BOOL_LITERALS[rhs]
    if isinstance(actual, bool):  # actual is bool but rhs isn't a bool literal
        return rhs
    try:
        return type(actual)(rhs)
    except (TypeError, ValueError):
        return rhs


def _eval_condition(actual: Any, condition: str) -> bool:
    """Evaluate condition like '>=1', '==False', '==escalate', 'in:a,b'."""
    condition = condition.strip()
    if condition.startswith("in:"):
        choices = [c.strip() for c in condition[3:].split(",")]
        return str(actual) in choices
    # Longest-first so '>=' matches before '>'
    for op_str, op_fn in (("==", op.eq), ("!=", op.ne), (">=", op.ge), ("<=", op.le), (">", op.gt), ("<", op.lt)):
        if condition.startswith(op_str):
            rhs_v = _parse_rhs(condition[len(op_str):], actual)
            try:
                return bool(op_fn(actual, rhs_v))
            except TypeError:
                return False
    return str(actual) == condition


def _patch_kwargs_for_task(arch_name: str, task: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Inject task-specific constructor kwargs (docs for RAG, working_dir for SWE, etc.)."""
    tid = task["id"]
    kwargs = dict(kwargs)
    if arch_name in ("AgenticRAG", "CorrectiveRAG", "SelfRAG", "AdaptiveRAG", "GraphRAG", "GraphMemoryAgent"):
        from agentic_architectures.data import STARDUST_CORPUS
        kwargs.setdefault("documents", list(STARDUST_CORPUS))
    if arch_name == "SWEAgent":
        work = Path(tempfile.mkdtemp(prefix="swe_bench_"))
        (work / "factorial.py").write_text(
            "def factorial(n):\n    return n * factorial(n - 1)\n\n"
            "if __name__ == '__main__':\n    assert factorial(0) == 1\n    print('PASS')\n"
        )
        kwargs["working_dir"] = work
        kwargs.setdefault("max_iterations", 6)
    if arch_name == "ComputerUse":
        if tid == "safety_block_blocked_domain":
            kwargs.setdefault("initial_screen", {"url": "https://example.com", "elements": [], "fields": {}, "submitted": False})
    if arch_name in ("ToolUse", "ReAct"):
        from agentic_architectures.tools import web_search_tool
        kwargs.setdefault("tools", [web_search_tool(max_results=3)])
        if arch_name == "ToolUse":
            kwargs.setdefault("max_rounds", 3)
    if arch_name == "MetaController":
        from agentic_architectures.architectures import Reflection, Planning
        kwargs.setdefault("roster", {
            "planning": Planning(llm=kwargs.get("_llm")),
            "reflection": Reflection(llm=kwargs.get("_llm")),
        })
        kwargs.pop("_llm", None)
    return kwargs


def run_one(arch_name: str, task: dict[str, Any], llm) -> dict[str, Any]:
    """Instantiate the architecture, run setup_prompts (if any) then the main prompt."""
    mod = importlib.import_module("agentic_architectures.architectures")
    cls = getattr(mod, arch_name)
    kwargs = dict(DEFAULT_KWARGS.get(arch_name, {}))
    kwargs["_llm"] = llm   # passed-through hint for MetaController
    kwargs = _patch_kwargs_for_task(arch_name, task, kwargs)

    t0 = time.time()
    output = ""
    metadata: dict[str, Any] = {}
    error: str | None = None
    arch = None
    captured = io.StringIO()
    try:
        arch = cls(llm=llm, **kwargs)
        with redirect_stdout(captured), redirect_stderr(captured):
            # Stateful arches: run setup prompts on same instance
            for sp in task.get("setup_prompts", []) or []:
                arch.run(sp)
            r = arch.run(task["prompt"])
        output = (r.output or "").strip()
        metadata = dict(r.metadata or {})
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        if arch is not None and hasattr(arch, "close"):
            try: arch.close()
            except Exception: pass
    elapsed = time.time() - t0

    out_lower = output.lower()
    contains_hits = [n for n in task.get("expected_contains", []) if n.lower() in out_lower]
    excludes_hits = [b for b in task.get("expected_excludes", []) if b.lower() in out_lower]

    metadata_pass = True
    metadata_failures: list[str] = []
    for key, condition in (task.get("score_metadata") or {}).items():
        actual = metadata.get(key)
        if actual is None or not _eval_condition(actual, condition):
            metadata_pass = False
            metadata_failures.append(f"{key}={actual!r} fails {condition}")

    contains_pass = (
        len(contains_hits) == len(task.get("expected_contains", []))
        and not excludes_hits
    )
    correct = contains_pass and metadata_pass and error is None

    return {
        "architecture": arch_name,
        "task_id": task["id"],
        "task_kind": task["kind"],
        "correct": correct,
        "elapsed_s": round(elapsed, 1),
        "output_excerpt": output[:200],
        "error": error,
        "contains_hits": contains_hits,
        "excludes_hits": excludes_hits,
        "metadata_failures": metadata_failures,
    }


def write_leaderboard(results: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    task_ids = [t["id"] for t in tasks]
    archs = sorted({r["architecture"] for r in results})
    by_arch_task: dict[tuple[str, str], dict[str, Any]] = {
        (r["architecture"], r["task_id"]): r for r in results
    }

    lines: list[str] = []
    lines.append("# Benchmarks\n")
    lines.append("Comparison of every architecture across the task suite. Each cell: `✓ (Xs)` = passed in X seconds, `✗` = ran but failed scoring, `❌ err` = exception, `—` = not applicable.\n")
    lines.append(f"Generated by `benchmarks/run_benchmark.py` over **{len(archs)} architectures** × **{len(tasks)} tasks** = **{len(results)} attempts**.\n")

    # Per-architecture aggregate table
    lines.append("## Leaderboard\n")
    header = "| Architecture | " + " | ".join(task_ids) + " | Score |"
    sep = "|---" * (len(task_ids) + 2) + "|"
    lines.append(header)
    lines.append(sep)
    for arch in archs:
        row = [f"`{arch}`"]
        score = 0
        n_attempted = 0
        for tid in task_ids:
            rec = by_arch_task.get((arch, tid))
            if rec is None:
                row.append("—")
                continue
            n_attempted += 1
            if rec["correct"]:
                score += 1
                row.append(f"✓ ({rec['elapsed_s']}s)")
            elif rec["error"]:
                row.append(f"❌ err")
            else:
                row.append(f"✗ ({rec['elapsed_s']}s)")
        row.append(f"**{score}/{n_attempted}**" if n_attempted else "—")
        lines.append("| " + " | ".join(row) + " |")

    # Per-task answer excerpts
    lines.append("\n## Per-task results\n")
    for t in tasks:
        lines.append(f"### `{t['id']}` ({t['kind']})\n")
        lines.append(f"> {t['prompt'].strip()[:280]}\n")
        if t.get("setup_prompts"):
            lines.append(f"Setup prompts: {len(t['setup_prompts'])} (called before main prompt on same arch instance)\n")
        if t.get("expected_contains"):
            lines.append(f"Expected contains: `{t['expected_contains']}`\n")
        if t.get("score_metadata"):
            lines.append(f"Expected metadata: `{t['score_metadata']}`\n")
        lines.append("| Arch | Result | Excerpt |")
        lines.append("|---|---|---|")
        for arch in archs:
            rec = by_arch_task.get((arch, t["id"]))
            if rec is None:
                continue
            v = "✓" if rec["correct"] else ("❌" if rec["error"] else "✗")
            ex = rec["output_excerpt"].replace("|", "\\|").replace("\n", " ")[:120]
            err = f" [{rec['error']}]" if rec["error"] else ""
            mf = f" [meta fails: {rec['metadata_failures']}]" if rec["metadata_failures"] else ""
            lines.append(f"| `{arch}` | {v} ({rec['elapsed_s']}s) | {ex}{err}{mf} |")
        lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote leaderboard → {OUT_PATH.relative_to(REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quick", action="store_true", help="Run only 1 arch per task")
    ap.add_argument("--only", default="", help="Comma-separated arch names to include")
    ap.add_argument("--provider", default="nebius")
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    args = ap.parse_args()

    cfg = yaml.safe_load(TASKS_PATH.read_text(encoding="utf-8"))
    tasks = cfg["tasks"]
    only = set(s.strip() for s in args.only.split(",") if s.strip()) if args.only else None

    # Coverage check
    covered_arches = set()
    matrix: list[tuple[str, dict[str, Any]]] = []
    for t in tasks:
        archs = t["architectures"]
        if args.quick:
            archs = archs[:1]
        for arch in archs:
            if only and arch not in only:
                continue
            covered_arches.add(arch)
            matrix.append((arch, t))

    print(f"Benchmark plan: {len(matrix)} attempts over {len(tasks)} tasks; {len(covered_arches)} unique architectures covered")
    # Coverage report: which architectures NOT in any task
    from agentic_architectures.architectures import __all__ as all_arch_names
    arch_classes = {n for n in all_arch_names if n not in ("Architecture", "ArchitectureResult")}
    missing = sorted(arch_classes - covered_arches)
    if missing:
        print(f"  ⚠️  not covered by any task: {missing}")
    else:
        print(f"  ✅ every architecture covered")

    if args.dry_run:
        for arch, t in matrix:
            print(f"  → {arch:25s}  on {t['id']}")
        print("\n(dry run — no LLM calls made)")
        return

    print(f"\nUsing {args.provider} · {args.model}")
    from agentic_architectures import get_llm
    llm = get_llm(provider=args.provider, model=args.model, temperature=0.2)

    results: list[dict[str, Any]] = []
    for i, (arch, t) in enumerate(matrix, 1):
        print(f"\n[{i}/{len(matrix)}] {arch:25s} on {t['id']:25s} …", end="", flush=True)
        rec = run_one(arch, t, llm)
        v = "✓" if rec["correct"] else ("ERR" if rec["error"] else "✗")
        print(f" {v} ({rec['elapsed_s']}s)")
        if rec["error"]:
            print(f"     error: {rec['error']}")
        if rec["metadata_failures"]:
            print(f"     metadata: {rec['metadata_failures']}")
        results.append(rec)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    (OUT_PATH.parent / "benchmarks_raw.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    write_leaderboard(results, tasks)

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    errored = sum(1 for r in results if r["error"])
    print(f"\nSummary: {correct}/{total} correct · {errored} errored")


if __name__ == "__main__":
    main()
