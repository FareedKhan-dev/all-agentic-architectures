"""Post-process notebook 25: rewrite § 9 against the Self-RAG captured run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "25_self_rag.ipynb"
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
    r"\s+NEEDS_RETRIEVAL:\s+(True|False)\s*\n"
    r"\s+N_RETRIEVED:\s+(\d+)\s*\n"
    r"\s+N_KEPT:\s+(\d+)\s*\n"
    r"\s+KEPT_INDICES:\s+(\[.*?\])\s*\n"
    r"\s+N_FULLY_RELEVANT:\s+(\d+)\s*\n"
    r"\s+N_NO_SUPPORT:\s+(\d+)\s*\n"
    r"\s+N_VERY_USEFUL:\s+(\d+)\s*\n"
    r"((?:[ \t]+doc\[\d+\]:[^\n]+\n)*)"
    r"[ \t]+FINAL_ANSWER:[ \t]+([^\n]+(?:\n(?![ \t]*TASK_TAG:)[^\n]*)*)",
)


def extract_run(nb: nbformat.NotebookNode) -> dict[str, object]:
    info: dict[str, object] = {"tasks": []}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        text = cell_output_text(cell)
        if "TASK_TAG:" in cell.source and "NEEDS_RETRIEVAL" in cell.source:
            tasks = []
            for m in TASK.finditer(text):
                tag, q, need_ret, n_ret, n_kept, kept_idx, n_rel, n_no_sup, n_use, doc_block, ans = m.groups()
                docs = [
                    {k: v for k, v in re.findall(r"(\w+)=(\w+)", line)}
                    for line in doc_block.strip().splitlines() if line.strip()
                ]
                tasks.append({
                    "tag": tag.strip(),
                    "task": q.strip(),
                    "needs_retrieval": need_ret == "True",
                    "n_retrieved": int(n_ret),
                    "n_kept": int(n_kept),
                    "kept_indices": kept_idx.strip(),
                    "n_fully_relevant": int(n_rel),
                    "n_no_support": int(n_no_sup),
                    "n_very_useful": int(n_use),
                    "docs": docs,
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
            f"| `{t['tag']}` | {t['needs_retrieval']} | {t['n_retrieved']} | {t['n_kept']} | "
            f"{t['n_fully_relevant']} | {t['n_no_support']} | {t['n_very_useful']} | "
            f"{_esc(t['answer'])[:80]}{'…' if len(t['answer']) > 80 else ''} |"
            for t in tasks
        )
    else:
        rows = "| — | — | — | — | — | — | — | — |"

    obs: list[str] = []
    for t in tasks:
        tag = t["tag"]
        if tag == "parametric":
            if not t["needs_retrieval"]:
                obs.append(f"**✅ `{tag}` task** — correctly skipped retrieval (needs_retrieval=False).")
            else:
                obs.append(f"**⚠️ `{tag}` task** — agent retrieved despite parametric question. Decider mis-calibrated.")
        elif tag == "direct":
            if t["n_kept"] >= 1 and t["needs_retrieval"]:
                obs.append(f"**✅ `{tag}` task** — retrieved + kept ≥1 doc ({t['n_kept']}/{t['n_retrieved']}). Reflection passed real-evidence docs through.")
            else:
                obs.append(f"**🤔 `{tag}` task** — kept {t['n_kept']}/{t['n_retrieved']} docs. Possibly over-strict reflector.")
        elif tag == "mismatch":
            if t["n_kept"] == 0 and t["needs_retrieval"]:
                obs.append(f"**✅ `{tag}` task** — reflector correctly dropped ALL docs (0/{t['n_retrieved']}). Architecture correctly admits gap.")
            elif t["n_kept"] > 0:
                obs.append(f"**🤔 `{tag}` task** — kept {t['n_kept']}/{t['n_retrieved']} docs despite the corpus not containing the answer. Reflector was too lenient — likely kept docs that surface-mention nitrogen / boiling but don't actually contain the answer.")
    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns surfaced."

    return f"""## 9 · What we just observed

The cells above ran Self-RAG on 3 task types (direct/parametric/mismatch) to exercise the per-doc reflection-token gate.

### 9.1 · Per-task reflection summary

| Tag | Needs retrieval? | Retrieved | Kept | Fully-relevant docs | No-support docs | Very-useful docs | Final answer |
|---|---|---|---|---|---|---|---|
{rows}

### 9.2 · Patterns surfaced in this run

{obs_block}

### 9.3 · The takeaway

Self-RAG's value lives in the **Kept / Retrieved** ratio in § 9.1. The architecture is working when:

1. **Direct tasks**: Kept > 0 (real evidence let through).
2. **Parametric tasks**: NEEDS_RETRIEVAL=False (no reflection burned).
3. **Mismatch tasks**: Kept = 0 (architecture admits the gap rather than hallucinating).

The deterministic-picker is in `_compose_keep`: pure-Python `is_relevant != 'not_relevant' AND is_supported != 'no_support'`. No LLM emits a numeric per-doc quality score — the gate is decided by two categorical commitments per doc."""


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
