"""Post-process notebook 23: rewrite § 9 against the Agentic RAG captured run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "23_agentic_rag.ipynb"
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
    r"\s+RETRIEVAL_COUNT:\s+(\d+)\s*\n"
    r"\s+ITERATIONS_USED:\s+(\d+)\s*\n"
    r"\s+QUERIES:\s+(\[.*?\])\s*\n"
    r"\s+FINAL_ANSWER:\s+(.+?)(?=\n\s*TASK_TAG:|\n\s*AGGREGATE:|\Z)",
    re.DOTALL,
)
PLAIN_BLOCK = re.compile(
    r"PLAIN_TASK_TAG:\s+(.+?)\s*\n"
    r"\s+PLAIN_ANSWER:\s+(.+?)(?=\n\s*PLAIN_TASK_TAG:|\n\s*PLAIN_RAG_RETRIEVALS:|\Z)",
    re.DOTALL,
)


def extract_run(nb: nbformat.NotebookNode) -> dict[str, object]:
    info: dict[str, object] = {"tasks": [], "plain": [], "plain_total": 0}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        text = cell_output_text(cell)
        if "TASK_TAG:" in cell.source and "AGGREGATE:" in cell.source:
            tasks = []
            for m in TASK_BLOCK.finditer(text):
                tag, task, count, iters, queries, answer = m.groups()
                tasks.append({
                    "tag": tag.strip(),
                    "task": task.strip(),
                    "retrieval_count": int(count),
                    "iterations": int(iters),
                    "queries": queries.strip(),
                    "answer": answer.strip(),
                })
            info["tasks"] = tasks
        if "PLAIN_TASK_TAG:" in cell.source:
            plain = []
            for m in PLAIN_BLOCK.finditer(text):
                tag, ans = m.groups()
                plain.append({"tag": tag.strip(), "answer": ans.strip()})
            info["plain"] = plain
            tm = re.search(r"PLAIN_RAG_RETRIEVALS:\s+(\d+)", text)
            if tm:
                info["plain_total"] = int(tm.group(1))
    return info


def _esc(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def make_commentary(info: dict[str, object]) -> str:
    tasks: list[dict] = info.get("tasks", [])  # type: ignore[assignment]
    plain: list[dict] = info.get("plain", [])  # type: ignore[assignment]
    plain_total = info.get("plain_total", 0)

    if tasks:
        rows = "\n".join(
            f"| `{t['tag']}` | {t['retrieval_count']} | {t['iterations']} | "
            f"{_esc(t['answer'])[:80]}{'…' if len(t['answer']) > 80 else ''} |"
            for t in tasks
        )
        total_retrievals = sum(t["retrieval_count"] for t in tasks)
        summary = (
            f"- **Total retrievals across {len(tasks)} tasks**: {total_retrievals}\n"
            f"- **Plain-RAG baseline retrievals** (1 per task): {plain_total}\n"
            f"- **Net savings vs always-retrieve**: {plain_total - total_retrievals} fewer "
            "retrieval calls" + (" (Agentic RAG used FEWER)" if total_retrievals < plain_total else
            " (Agentic RAG used MORE — sign of over-retrieval pathology)")
        )
    else:
        rows = "| — | — | — | — |"
        summary = "_(no tasks captured)_"

    if plain:
        comp_rows = []
        for t in tasks:
            pa = next((p for p in plain if p["tag"] == t["tag"]), None)
            comp_rows.append(
                f"| `{t['tag']}` | {_esc(t['answer'])[:60]} | "
                f"{_esc(pa['answer'])[:60] if pa else '—'} |"
            )
        comp_table = "\n".join(comp_rows)
    else:
        comp_table = "| — | — | — |"

    obs: list[str] = []
    if tasks:
        # Per-task expected behaviour
        for t in tasks:
            tag = t["tag"]
            rc = t["retrieval_count"]
            if tag == "arithmetic" and rc > 0:
                obs.append(
                    f"**⚠️  Over-retrieval on `{tag}` task** ({rc} retrieval call(s)) — "
                    "this task is pure arithmetic and shouldn't have needed retrieval. "
                    "Strengthen the DECIDE prompt's 'do not retrieve redundantly' rule."
                )
            elif tag == "arithmetic" and rc == 0:
                obs.append(
                    f"**✅ Correctly skipped retrieval on `{tag}` task** — agent recognised "
                    "this as parametric, didn't waste a call."
                )
            elif tag == "general" and rc > 0:
                obs.append(
                    f"**⚠️  Over-retrieval on `{tag}` task** ({rc} call(s)) — general "
                    "knowledge shouldn't need retrieval. Likely the corpus's presence in the "
                    "prompt anchored the agent toward retrieving."
                )
            elif tag == "general" and rc == 0:
                obs.append(
                    f"**✅ Correctly skipped retrieval on `{tag}` task** — answered from "
                    "parametric memory."
                )
            elif tag in ("payload", "multi-fact") and rc == 0:
                obs.append(
                    f"**❌ Failed to retrieve on `{tag}` task** — agent committed to an answer "
                    "without consulting the corpus. Likely hallucinated."
                )
            elif tag == "multi-fact" and rc >= 2:
                obs.append(
                    f"**✅ Multi-hop retrieval on `{tag}` task** ({rc} sequential retrievals) — "
                    "agent recognised the answer needed more than one query."
                )

        if plain and tasks:
            total_r = sum(t["retrieval_count"] for t in tasks)
            if total_r < plain_total:
                obs.append(
                    f"**✅ Agentic RAG saved {plain_total - total_r} retrieval call(s) vs "
                    f"plain RAG** ({total_r} vs {plain_total}). The agent's discrimination "
                    "between retrieval-needing and parametric tasks paid off."
                )
            else:
                obs.append(
                    f"**🤔 Agentic RAG used {total_r - plain_total} MORE retrievals than "
                    "plain RAG** — multi-hop tasks pushed the count above the always-one baseline. "
                    "Net cost is higher but multi-hop answers should be more complete."
                )

    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns surfaced."

    return f"""## 9 · What we just observed

The cells above ran Agentic RAG on **4 task types** (single-fact retrieval needed, arithmetic, multi-hop, out-of-corpus general knowledge) and compared to a plain-RAG baseline that always retrieves once.

### 9.1 · Per-task retrieval behaviour

| Tag | Retrievals | Iterations | Final answer |
|---|---|---|---|
{rows}

{summary}

### 9.2 · Agentic RAG vs plain RAG

| Tag | Agentic answer | Plain-RAG answer |
|---|---|---|
{comp_table}

### 9.3 · Patterns surfaced in this run

{obs_block}

### 9.4 · The takeaway

Agentic RAG's value lives in two columns of § 9.1: **`Retrievals`** and **`Final answer`**. The architecture earns its keep when:

1. **Retrieval count varies across tasks** (0 for parametric, 1 for single-fact, ≥2 for multi-hop). Flat retrieval count → degenerated to plain RAG.
2. **Zero-retrieval answers are still correct** (agent's parametric-memory judgement is accurate).
3. **Multi-hop retrievals produce complete answers** (not just the first-query's fact).

Read § 9.3 for the specific patterns this run surfaced — over-retrieval on arithmetic, under-retrieval on multi-fact, etc. — and use them to tune the DECIDE prompt for your task distribution."""


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
        f"tailored: {len(info['tasks'])} tasks, plain_total={info['plain_total']}, "
        f"agentic_total={sum(t['retrieval_count'] for t in info['tasks'])}"
    )


if __name__ == "__main__":
    main()
