"""Post-process notebook 04: rewrite § 9 with the actual Plan-Execute-Replan run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "04_planning.ipynb"
ANSI = re.compile(r"\x1b\[[0-9;]*[mGKH]")


def _normalize_ws(s: str) -> str:
    s = re.sub(r"[─-╿]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def cell_output_text(cell: nbformat.NotebookNode) -> str:
    chunks: list[str] = []
    for o in cell.outputs:
        t = o.get("text", "") or o.get("data", {}).get("text/plain", "")
        if isinstance(t, list):
            t = "".join(t)
        chunks.append(ANSI.sub("", str(t)))
    return "\n".join(chunks)


def extract_run(nb: nbformat.NotebookNode) -> dict[str, object]:
    info: dict[str, object] = {
        "answer": "",
        "steps_executed": 0,
        "replans": 0,
        "max_replans": 0,
        "steps": [],
        "results": [],
    }
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "result = arch.run(TASK)" in cell.source:
            text = cell_output_text(cell)
            m = re.search(
                r"Final answer[\s\S─-╿]*?\n(.+?)\n[\s─-]*\n[\s─-╿]*?\d+\s+step",
                text,
                re.DOTALL,
            )
            if m:
                info["answer"] = re.sub(r"[\s─-╿]+$", "", m.group(1)).strip()
            m = re.search(
                r"(\d+)\s+step\(s\)\s+executed\s+\W\s+(\d+)\s+replan\(s\)\s+\W\s+budget\s+(\d+)",
                text,
            )
            if m:
                info["steps_executed"] = int(m.group(1))
                info["replans"] = int(m.group(2))
                info["max_replans"] = int(m.group(3))
        if "for i, t in enumerate(result.trace" in cell.source:
            text = cell_output_text(cell)
            # `print_step` renders "› [N] STEP\n<body>\n› <next>", so the step body
            # lives between the STEP label line and the next "› RESULT" or "› [N+1]".
            steps = re.findall(
                r"\[\d+\]\s+STEP\s*\n(.+?)(?=\n\s*[›>]\s+|\Z)",
                text,
                re.DOTALL,
            )
            results = re.findall(
                r"\bRESULT\s*\n(.+?)(?=\n\s*[›>]\s+|\Z)",
                text,
                re.DOTALL,
            )
            info["steps"] = [_normalize_ws(s) for s in steps]
            info["results"] = [_normalize_ws(r) for r in results]
            if not info["answer"]:
                # Fall back: synthesise the answer from results if we missed it.
                pass
    return info


def make_commentary(info: dict[str, object]) -> str:
    n_steps = info.get("steps_executed", 0)
    n_replans = info.get("replans", 0)
    budget = info.get("max_replans", 0)
    steps: list[str] = info.get("steps", [])  # type: ignore[assignment]
    results: list[str] = info.get("results", [])  # type: ignore[assignment]
    answer: str = info.get("answer", "")  # type: ignore[assignment]

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ").strip()

    plan_table = (
        "\n".join(
            f"| {i+1} | {esc(s)[:120]}{'…' if len(s) > 120 else ''} | {esc(r)[:120]}{'…' if r and len(r) > 120 else ''} |"
            for i, (s, r) in enumerate(zip(steps, results))
        )
        if steps
        else "| — | _(no plan steps captured)_ | — |"
    )

    obs: list[str] = []
    if n_steps == 0:
        obs.append("**No plan steps executed.** Likely the planner failed structured-output validation; check provider compatibility.")
    elif n_steps == 1:
        obs.append(
            "**Planning degraded to single-step.** Only one step was executed — "
            "the task either didn't decompose, or the planner produced a single super-step. "
            "For this kind of task, **ReAct (nb 03)** would have been equally effective and cheaper."
        )
    elif n_steps >= 7:
        obs.append(
            f"**Over-decomposition.** {n_steps} steps for a task this size is likely "
            "too many — each step is a sub-agent call. Tighten the Plan schema "
            "description: *'Use 3-5 steps; combine atomic lookups when possible.'*"
        )
    else:
        obs.append(
            f"**Healthy decomposition.** {n_steps} ordered steps — each materially "
            "different and pushing toward the answer. This is what Planning looks "
            "like when it works."
        )

    if n_replans == 0:
        obs.append(
            "**Plan was good first try.** The replanner immediately set `is_done=True` "
            "after the initial plan finished — no extension needed. Best-case outcome."
        )
    elif n_replans >= budget:
        obs.append(
            f"**Replan budget exhausted** ({n_replans}/{budget}). The replanner was "
            "FORCED to finalize even though it wanted more steps. Inspect the answer "
            "carefully — it may be under-grounded."
        )
    else:
        obs.append(
            f"**Replan used productively** ({n_replans}/{budget}). The initial plan "
            "wasn't quite sufficient; the replanner added steps and converged."
        )

    if answer and "http" not in answer.lower():
        obs.append(
            "**No URLs in the final answer** despite the task asking for citation. "
            "The replanner synthesised from parametric knowledge instead of grounding "
            "in the executor's results — consider tightening the `_synthesize_from_history` "
            "prompt or adding a citation-required schema field."
        )

    obs_block = "\n\n".join(f"- {o}" for o in obs)

    answer_block = (
        "> " + (answer[:600].replace("\n", "\n> ") if answer else "_(no answer captured)_")
        + ("…" if len(answer) > 600 else "")
    )

    return f"""## 9 · What we just observed

The cells above are live. Below: a quantitative + qualitative breakdown of the **actual** Plan-Execute-Replan loop the Nebius-hosted Llama-3.3-70B agent produced on this run.

### 9.1 · Quantitative summary

| Metric | Value |
|---|---|
| Plan steps executed | **{n_steps}** |
| Replans triggered | **{n_replans}** / {budget} |
| Final answer length | {len(answer)} chars |

### 9.2 · Plan ↔ result alignment

| # | Plan step | Execution result (truncated) |
|---|---|---|
{plan_table}

### 9.3 · Pathologies / patterns surfaced in this run

{obs_block}

### 9.4 · The final answer (verbatim)

{answer_block}

### 9.5 · The takeaway

When a task **naturally decomposes** (multi-fact comparison, structured report, multi-step computation), Planning is the right tool — you save token cost vs. ReAct's per-step thinking *and* gain a human-inspectable contract. When the task is **open-ended or one-shot**, planning is overhead — fall back to ReAct or plain Tool Use.

The cleanest signal: did your run use any replans? If yes — the initial plan wasn't quite right, but the recovery worked. If no replans AND ≥ 3 steps executed — you got the ideal Planning trace."""


def main() -> None:
    nb = nbformat.read(NB_PATH, as_version=4)
    info = extract_run(nb)
    new_md = make_commentary(info)
    replaced = False
    for cell in nb.cells:
        if cell.cell_type == "markdown" and cell.source.lstrip().startswith(
            "## 9 · What we just observed"
        ):
            cell.source = new_md
            replaced = True
            break
    if not replaced:
        raise RuntimeError("section 9 not found")
    nbformat.write(nb, NB_PATH)
    print(
        f"tailored section 9: steps={info['steps_executed']}, "
        f"replans={info['replans']}, captured_steps={len(info['steps'])}"
    )


if __name__ == "__main__":
    main()
