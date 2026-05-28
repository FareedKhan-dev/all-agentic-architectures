"""Post-process notebook 27: rewrite § 9 against the GraphRAG captured run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "27_graph_rag.ipynb"
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
    r"TASK_TAG:\s+(\w+)\s*\n"
    r"\s+TASK:\s+(.+?)\s*\n"
    r"\s+SCOPE:\s+(\w+)\s*\n"
    r"\s+TARGET_ENTITIES:\s+(\[.*?\])\s*\n"
    r"\s+CONTEXT_CHARS:\s+(\d+)\s*\n"
    r"\s+N_COMMUNITIES:\s+(\d+)\s*\n"
    r"\s+COMMUNITY_SIZES:\s+(\[[^\]]*\])\s*\n"
    r"\s+FINAL_ANSWER:\s+([^\n]+(?:\n(?![ \t]*TASK_TAG:)[^\n]*)*)"
)
BUILD = re.compile(
    r"BUILD_ELAPSED:\s+([0-9.]+)s.*?"
    r"N_COMMUNITIES:\s+(\d+).*?"
    r"COMMUNITY_SIZES:\s+(\[[^\]]*\])",
    re.DOTALL,
)


def extract_run(nb: nbformat.NotebookNode) -> dict[str, object]:
    info: dict[str, object] = {"build": None, "tasks": []}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        text = cell_output_text(cell)
        if "BUILD_ELAPSED" in text:
            m = BUILD.search(text)
            if m:
                info["build"] = {
                    "elapsed": float(m.group(1)),
                    "n_communities": int(m.group(2)),
                    "community_sizes": m.group(3),
                }
        if "TASK_TAG:" in cell.source and "SCOPE" in cell.source:
            tasks = []
            for m in TASK.finditer(text):
                tag, task, scope, ents, ctx_chars, n_comm, comm_sizes, ans = m.groups()
                tasks.append({
                    "tag": tag.strip(),
                    "task": task.strip(),
                    "scope": scope.strip(),
                    "target_entities": ents.strip(),
                    "context_chars": int(ctx_chars),
                    "n_communities": int(n_comm),
                    "community_sizes": comm_sizes.strip(),
                    "answer": ans.strip(),
                })
            info["tasks"] = tasks
    return info


def _esc(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def make_commentary(info: dict[str, object]) -> str:
    build = info.get("build") or {}
    tasks: list[dict] = info.get("tasks", [])  # type: ignore[assignment]

    if build:
        build_block = (
            f"- **Build elapsed**: {build['elapsed']:.1f}s\n"
            f"- **Communities detected**: {build['n_communities']}\n"
            f"- **Community sizes**: {build['community_sizes']}"
        )
    else:
        build_block = "_(no build info captured)_"

    if tasks:
        rows = "\n".join(
            f"| `{t['tag']}` | `{t['scope']}` | {t['target_entities']} | "
            f"{t['context_chars']} | {_esc(t['answer'])[:120]}{'…' if len(t['answer']) > 120 else ''} |"
            for t in tasks
        )
    else:
        rows = "| — | — | — | — | — |"

    obs: list[str] = []
    expected = {"local": "local", "global": "global"}
    for t in tasks:
        exp = expected.get(t["tag"], "?")
        if t["scope"] == exp:
            obs.append(f"**✅ `{t['tag']}` correctly classified as `{t['scope']}`**.")
        else:
            obs.append(f"**🤔 `{t['tag']}` classified as `{t['scope']}`, expected `{exp}`**.")
    if build:
        if build["n_communities"] > 1:
            obs.append(f"**✅ Graph clustering produced {build['n_communities']} distinct communities** — community-summary path has real material to feed global queries.")
        else:
            obs.append("**⚠️  Only 1 (or 0) communities** — corpus too sparse/dense for meaningful clustering. Global queries collapse to one summary.")
    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns surfaced."

    return f"""## 9 · What we just observed

The cells above built a knowledge graph from the Stardust corpus, detected entity communities, summarised each, then ran local vs global queries.

### 9.1 · Build summary

{build_block}

### 9.2 · Per-query routing + context

| Tag | Scope | Target entities | Context chars | Final answer |
|---|---|---|---|---|
{rows}

### 9.3 · Patterns surfaced in this run

{obs_block}

### 9.4 · The takeaway

GraphRAG's value over plain vector RAG comes from two sources:

1. **Global question answers** — the community summaries (built once, used many times) let the system answer "what are the themes?" without rerunning extraction per query. Plain RAG can't answer this — themes aren't in any single retrieved snippet.
2. **Local question precision** — entity-anchored subgraph traversal gives exactly the facts about an entity, no irrelevant similar-but-unrelated snippets.

The cost is the build phase (1 LLM call per doc + 1 per community). Once built, queries are cheap. Cache the graph + summaries; rebuild only when the corpus changes."""


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
    print(f"tailored: build={info['build']} tasks={len(info['tasks'])}")


if __name__ == "__main__":
    main()
