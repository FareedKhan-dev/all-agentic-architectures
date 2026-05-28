"""Benchmark harness — run every architecture against a task suite, produce a leaderboard.

The trending-hook of the repo: readers love comparative tables, and being able
to point at a benchmarks page is a strong signal of "this is a serious project".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic_architectures.architectures.base import Architecture


@dataclass
class BenchmarkTask:
    """A single task in the benchmark suite."""

    id: str
    prompt: str
    rubric: str = "Score the answer for correctness and completeness on a 1-5 scale."
    expected_keywords: list[str] = field(default_factory=list)


@dataclass
class BenchmarkRun:
    """The outcome of one (architecture × task) cell."""

    architecture: str
    task_id: str
    output: str
    latency_s: float
    score: float | None = None
    error: str | None = None


def run_benchmark(
    architectures: list[Architecture],
    tasks: list[BenchmarkTask],
    judge_schema: type | None = None,
) -> list[BenchmarkRun]:
    """Run every architecture against every task. Optionally score via LLMJudge."""
    runs: list[BenchmarkRun] = []
    judge = None
    if judge_schema is not None:
        from agentic_architectures.evaluators.judge import LLMJudge

        judge = LLMJudge(
            schema=judge_schema,
            rubric="Score the answer for correctness and completeness on a 1-5 scale.",
        )

    for arch in architectures:
        for task in tasks:
            t0 = time.perf_counter()
            try:
                result = arch.run(task.prompt)
                latency = time.perf_counter() - t0
                score: float | None = None
                if judge is not None:
                    evaluation: Any = judge.evaluate(result.output, context={"task": task.prompt})
                    # Average of all int fields named like "correctness" etc.
                    nums = [v for v in evaluation.model_dump().values() if isinstance(v, (int, float))]
                    score = sum(nums) / len(nums) if nums else None
                runs.append(
                    BenchmarkRun(
                        architecture=arch.name,
                        task_id=task.id,
                        output=result.output,
                        latency_s=latency,
                        score=score,
                    )
                )
            except Exception as e:  # noqa: BLE001 -- benchmarks must keep going
                runs.append(
                    BenchmarkRun(
                        architecture=arch.name,
                        task_id=task.id,
                        output="",
                        latency_s=time.perf_counter() - t0,
                        error=f"{type(e).__name__}: {e}",
                    )
                )
    return runs


def to_markdown_table(runs: list[BenchmarkRun]) -> str:
    """Render benchmark runs as a markdown leaderboard."""
    headers = ["Architecture", "Task", "Score", "Latency (s)", "Error"]
    rows = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in runs:
        rows.append(
            "| "
            + " | ".join(
                [
                    r.architecture,
                    r.task_id,
                    f"{r.score:.2f}" if r.score is not None else "—",
                    f"{r.latency_s:.2f}",
                    (r.error or "").replace("|", "\\|")[:60],
                ]
            )
            + " |"
        )
    return "\n".join(rows)
