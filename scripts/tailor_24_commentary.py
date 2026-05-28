"""Post-process notebook 24: rewrite § 9 against the CRAG captured run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "24_corrective_rag.ipynb"
ANSI = re.compile(r"\x1b\[[0-9;]*[mGKH]")


def cell_output_text(cell: nbformat.NotebookNode) -> str:
    chunks: list[str] = []
    for o in cell.outputs:
        t = o.get("text", "") or o.get("data", {}).get("text/plain", "")
        if isinstance(t, list):
            t = "".join(t)
        chunks.append(ANSI.sub("", str(t)))
    return "\n".join(chunks)


TASK_BLOCK = re.compile(
    r"TASK_TAG:\s+(.+?)\s*\n"
    r"\s+TASK:\s+(.+?)\s*\n"
    r"\s+N_RETRIEVED:\s+(\d+)\s*\n"
    r"\s+N_RELEVANT:\s+(\d+)\s*\n"
    r"\s+N_AMBIGUOUS:\s+(\d+)\s*\n"
    r"\s+N_IRRELEVANT:\s+(\d+)\s*\n"
    r"\s+RELEVANCE_FRACTION:\s+([0-9.]+)\s*\n"
    r"\s+ROUTE:\s+(.+?)\s*\n"
    r"\s+N_WEB:\s+(\d+)\s*\n"
    r"\s+FINAL_ANSWER:\s+(.+?)(?=\n\s*TASK_TAG:|\Z)",
    re.DOTALL,
)


def extract_run(nb: nbformat.NotebookNode) -> dict[str, object]:
    info: dict[str, object] = {"tasks": []}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        text = cell_output_text(cell)
        if "TASK_TAG:" in cell.source and "ROUTE:" in cell.source:
            tasks = []
            for m in TASK_BLOCK.finditer(text):
                tag, task, n_ret, n_rel, n_amb, n_irr, rel_frac, route, n_web, ans = m.groups()
                tasks.append({
                    "tag": tag.strip(),
                    "task": task.strip(),
                    "n_retrieved": int(n_ret),
                    "n_relevant": int(n_rel),
                    "n_ambiguous": int(n_amb),
                    "n_irrelevant": int(n_irr),
                    "relevance_fraction": float(rel_frac),
                    "route": route.strip(),
                    "n_web": int(n_web),
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
            f"| `{t['tag']}` | {t['n_retrieved']} | "
            f"{t['n_relevant']}/{t['n_ambiguous']}/{t['n_irrelevant']} | "
            f"{t['relevance_fraction']:.0%} | `{t['route']}` | {t['n_web']} | "
            f"{_esc(t['answer'])[:80]}{'…' if len(t['answer']) > 80 else ''} |"
            for t in tasks
        )
    else:
        rows = "| — | — | — | — | — | — | — |"

    # Per-task expectations
    obs: list[str] = []
    for t in tasks:
        tag = t["tag"]
        rf = t["relevance_fraction"]
        route = t["route"]
        if tag == "in_corpus":
            if route == "use_retrieved" and rf >= 0.5:
                obs.append(f"**✅ `{tag}` task** — high relevance fraction ({rf:.0%}) routed to `use_retrieved` as expected.")
            else:
                obs.append(f"**🤔 `{tag}` task** — relevance fraction {rf:.0%}, route `{route}`. Expected high relevance + use_retrieved. Either the grader is over-strict or the retrieve missed the relevant doc.")
        elif tag == "out_of_corpus":
            if route == "use_web":
                obs.append(f"**✅ `{tag}` task** — corpus correctly graded irrelevant ({rf:.0%}), routed to web fallback.")
            elif route == "use_mixed":
                obs.append(f"**🤔 `{tag}` task** — graded {rf:.0%} relevant, routed `use_mixed`. Some docs surface-matched the query (the grader is being lenient).")
            else:
                obs.append(f"**❌ `{tag}` task** — should have fallen back to web but routed `{route}`. Grader was too lenient on irrelevant docs.")
        elif tag == "mixed":
            if route == "use_mixed":
                obs.append(f"**✅ `{tag}` task** — correctly recognised partial coverage; mixed retrieve + web.")
            else:
                obs.append(f"**🤔 `{tag}` task** — routed `{route}` not `use_mixed`. Hybrid coverage wasn't recognised.")
    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns surfaced."

    return f"""## 9 · What we just observed

The cells above ran CRAG on **3 task types** (in-corpus, out-of-corpus, mixed) to exercise the grade-then-route logic.

### 9.1 · Per-task retrieval, grading, and routing

| Tag | Retrieved | Rel/Amb/Irr | Rel% | Route | Web docs | Final answer |
|---|---|---|---|---|---|---|
{rows}

### 9.2 · Patterns surfaced in this run

{obs_block}

### 9.3 · The takeaway

CRAG's two columns to watch in § 9.1: **`Rel%`** (the deterministic-picker input) and **`Route`** (the Python-composed output). The architecture is working iff:

1. **In-corpus tasks** show high Rel% → `use_retrieved`.
2. **Out-of-corpus tasks** show low Rel% → `use_web`.
3. **Mixed tasks** show moderate Rel% → `use_mixed`.

If routes don't match expectations (§ 9.2 flags), the grader is mis-calibrated — usually too lenient (calls weak-matches `relevant`). Tighten the grader's prompt or add a second-pass confirm step."""


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
