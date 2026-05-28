"""Post-process notebook 17: rewrite § 9 against the captured run."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "17_reflexive_metacognitive.ipynb"
ANSI = re.compile(r"\x1b\[[0-9;]*[mGKH]")


def cell_output_text(cell: nbformat.NotebookNode) -> str:
    chunks: list[str] = []
    for o in cell.outputs:
        t = o.get("text", "") or o.get("data", {}).get("text/plain", "")
        if isinstance(t, list):
            t = "".join(t)
        chunks.append(ANSI.sub("", str(t)))
    return "\n".join(chunks)


def extract_run(nb: nbformat.NotebookNode) -> dict[str, object]:
    info: dict[str, object] = {"tasks": []}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "TASKS = [" in cell.source and "for tag, t in TASKS" in cell.source:
            text = cell_output_text(cell)
            # Split by task markers and parse each chunk independently.
            chunks = re.split(r"\n(?=›\s*\[\w+\])", text)
            tasks = []
            for chunk in chunks:
                tag_m = re.search(r"\[(\w+)\]", chunk)
                route_m = re.search(r"final_route:\s+(\w+)\s+\(LLM\s+said:\s+(\w+);\s+override=(\w+)\)", chunk)
                feat_m = re.search(
                    r"capability_match=(\d+),\s+requires_lookup=(\w+),\s+requires_credentials=(\w+)",
                    chunk,
                )
                ans_m = re.search(r"answer:\s+(.+?)$", chunk, re.DOTALL)
                if not (tag_m and route_m and feat_m):
                    continue
                tasks.append({
                    "tag": tag_m.group(1).lower(),
                    "route": route_m.group(1),
                    "llm_route": route_m.group(2),
                    "override": route_m.group(3) == "True",
                    "capability_match": int(feat_m.group(1)),
                    "requires_lookup": feat_m.group(2) == "True",
                    "requires_credentials": feat_m.group(3) == "True",
                    "answer": re.sub(r"\s+", " ", ans_m.group(1) if ans_m else "").strip()[:200],
                })
            info["tasks"] = tasks
    return info


def make_commentary(info: dict[str, object]) -> str:
    tasks: list[dict] = info.get("tasks", [])  # type: ignore[assignment]

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ").strip()

    table = (
        "\n".join(
            f"| {t['tag']} | {t['capability_match']}/5 | {t['requires_credentials']} | {t['llm_route']} | **{t['route']}** | {t['override']} | {esc(t['answer'])[:100]} |"
            for t in tasks
        )
        if tasks
        else "| — | — | — | — | — | — | _(no tasks captured)_ |"
    )

    route_counts = Counter(t["route"] for t in tasks)
    override_count = sum(1 for t in tasks if t["override"])

    obs: list[str] = []

    # Route diversity
    if tasks:
        if len(route_counts) >= 3:
            obs.append(
                f"**Route diversity**: {len(route_counts)} distinct routes used across "
                f"{len(tasks)} tasks ({dict(route_counts)}). The architecture is genuinely "
                "discriminating between task types."
            )
        elif len(route_counts) == 2:
            obs.append(
                f"**Partial route diversity**: only {len(route_counts)} routes used "
                f"({dict(route_counts)}). Try more varied tasks to exercise all 4 branches."
            )
        else:
            obs.append(
                f"**Single route used** ({list(route_counts.keys())[0]}) — the agent is "
                "either over-escalating or over-confident. Inspect the self-model "
                "(too vague? too strict?)."
            )

    # Override-firing check
    if tasks:
        if override_count > 0:
            overridden = [t for t in tasks if t["override"]]
            obs.append(
                f"**Python override fired on {override_count}/{len(tasks)} task(s)** — "
                "specifically for: " + ", ".join(t["tag"] for t in overridden) + ". "
                "These are the cases where the LLM admitted credentials gap or low "
                "capability but its `route` choice would have let it answer anyway. "
                "Python forced `escalate` instead — the deterministic backstop working."
            )
        else:
            obs.append(
                "**Python override NEVER fired** — either the LLM consistently chose "
                "the correct route on its own, OR the high-risk tasks (medical / legal) "
                "didn't trigger `requires_credentials=True`. Inspect the per-task table "
                "to see which."
            )

    # Per-task sanity check against expected routes
    expected = {
        "general": "answer",
        "live": ("use_tool", "escalate"),  # either acceptable
        "medical": "escalate",
        "legal": "escalate",
    }
    correct = 0
    wrong = []
    for t in tasks:
        exp = expected.get(t["tag"])
        if exp is None:
            continue
        exp_set = {exp} if isinstance(exp, str) else set(exp)
        if t["route"] in exp_set:
            correct += 1
        else:
            wrong.append(f"{t['tag']}→{t['route']} (expected {exp_set})")
    if tasks:
        obs.append(
            f"**Routing accuracy vs author expectation**: {correct}/{len(tasks)} tasks "
            "routed as expected. " + ("Mismatches: " + ", ".join(wrong) if wrong else "All correct.")
        )

    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns."

    return f"""## 9 · What we just observed

The cells above ran 4 tasks of varying domain through Reflexive Metacognitive with the default self-model.

### 9.1 · Per-task routing decisions

| Tag | capability_match | requires_credentials | LLM route | **Final route** | Python override? | Answer (truncated) |
|---|---|---|---|---|---|---|
{table}

### 9.2 · Route + override distribution

| Metric | Value |
|---|---|
| Distinct routes used | {len(route_counts)} ({dict(route_counts) or '—'}) |
| Python overrides fired | {override_count} / {len(tasks)} |
| Most common route | {(route_counts.most_common(1) or [('—', 0)])[0][0]} |

### 9.3 · Patterns surfaced in this run

{obs_block}

### 9.4 · The takeaway

A *healthy* Reflexive Metacognitive run has:

1. **Route diversity** — different task domains hit different routes.
2. **Python override fires** on credentialed tasks (medical / legal / fiduciary).
3. **No bluffing** — the answer for `escalate` cases explicitly declines, doesn't hedge.
4. **`requires_credentials` semantic accuracy** — set True for tasks that genuinely require professional credentials.

The deterministic Python override is the safety guarantee here. Even if the LLM is prompt-injected to say `route="answer"` on a medical question, as long as `requires_credentials` ends up True, Python forces escalate. Never ship this architecture without that backstop."""


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
    print(f"tailored section 9: {len(info['tasks'])} tasks captured")


if __name__ == "__main__":
    main()
