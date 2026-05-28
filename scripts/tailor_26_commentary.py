"""Post-process notebook 26: rewrite § 9 against the Adaptive RAG captured run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "26_adaptive_rag.ipynb"
ANSI = re.compile(r"\x1b\[[0-9;]*[mGKH]")


def cell_output_text(cell: nbformat.NotebookNode) -> str:
    chunks: list[str] = []
    for o in cell.outputs:
        t = o.get("text", "") or o.get("data", {}).get("text/plain", "")
        if isinstance(t, list):
            t = "".join(t)
        chunks.append(ANSI.sub("", str(t)))
    return "\n".join(chunks)


TASK = re.compile(
    r"TASK_TAG:\s+(.+?)\s*\n"
    r"\s+TASK:\s+(.+?)\s*\n"
    r"\s+ROUTED_TO:\s+(\w+)\s*\n"
    r"\s+CLASSIFICATION_RATIONALE:\s+(.+?)\s*\n"
    r"\s+RETRIEVAL_COUNT:\s+(\d+)\s*\n"
    r"\s+FINAL_ANSWER:\s+([^\n]+(?:\n(?![ \t]*TASK_TAG:)[^\n]*)*)"
)


def extract_run(nb: nbformat.NotebookNode) -> dict[str, object]:
    info: dict[str, object] = {"tasks": []}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        text = cell_output_text(cell)
        if "TASK_TAG:" in cell.source and "ROUTED_TO" in cell.source:
            tasks = []
            for m in TASK.finditer(text):
                tag, task, route, rationale, n_ret, ans = m.groups()
                tasks.append({
                    "tag": tag.strip(),
                    "task": task.strip(),
                    "route": route.strip(),
                    "rationale": rationale.strip(),
                    "retrieval_count": int(n_ret),
                    "answer": ans.strip(),
                })
            info["tasks"] = tasks
    return info


def _esc(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def make_commentary(info: dict[str, object]) -> str:
    tasks: list[dict] = info.get("tasks", [])  # type: ignore[assignment]
    if tasks:
        rows = "\n".join(
            f"| `{t['tag']}` | `{t['route']}` | {t['retrieval_count']} | "
            f"{_esc(t['rationale'])[:80]} | {_esc(t['answer'])[:80]}{'…' if len(t['answer']) > 80 else ''} |"
            for t in tasks
        )
    else:
        rows = "| — | — | — | — | — |"

    expected_routes = {
        "arithmetic": "no_retrieval",
        "simple_lookup": "single_step",
        "multi_hop": "multi_step",
    }
    obs: list[str] = []
    correct_routes = 0
    for t in tasks:
        exp = expected_routes.get(t["tag"], "?")
        if t["route"] == exp:
            correct_routes += 1
            obs.append(f"**✅ `{t['tag']}` correctly routed to `{t['route']}`** (matches expected).")
        else:
            obs.append(f"**🤔 `{t['tag']}` routed to `{t['route']}`, expected `{exp}`** — classifier mis-bucketed this query.")
    if tasks:
        obs.append(f"**Routing accuracy: {correct_routes}/{len(tasks)}** correct against the expected buckets.")
    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns surfaced."

    return f"""## 9 · What we just observed

The cells above ran Adaptive RAG on 3 task types matched to the 3 routing buckets. The classifier picks the bucket once upfront; Python routes to the matched executor.

### 9.1 · Classification + routing summary

| Tag | Routed to | Retrievals | Classification rationale | Final answer |
|---|---|---|---|---|
{rows}

### 9.2 · Routing accuracy + patterns

{obs_block}

### 9.3 · The takeaway

Adaptive RAG's value is **all in the classifier**. Read §9.1's `Routed to` column against §9.2's expected buckets. A mismatch means the classifier mis-bucketed the query, and the subsequent execution will under-perform (single_step retrieve when multi was needed → incomplete answer; or wasted calls on no_retrieval that wasn't really no-retrieval).

The deterministic-picker pattern is the LLM's categorical `complexity` field plus Python's `if/elif` route. No numeric scoring — the LLM never emits a confidence-of-route score that could flatten."""


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
    print(f"tailored: {len(info['tasks'])} tasks")


if __name__ == "__main__":
    main()
