"""Post-process notebook 09: rewrite § 9 against the ToT captured run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "09_tree_of_thoughts.ipynb"
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
        "tree_size": 0,
        "max_depth": 0,
        "best_score": 0,
        "tree_lines": [],
        "llama_scores": [],
    }
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "result = arch.run(TASK)" in cell.source:
            text = cell_output_text(cell)
            m = re.search(
                r"Final answer[\s\S─-╿]*?\n(.+?)\n[\s─-]*\n[\s─-╿]*?tree size:",
                text,
                re.DOTALL,
            )
            if m:
                info["answer"] = re.sub(r"[\s─-╿]+$", "", m.group(1)).strip()
            m = re.search(
                r"tree size:\s+(\d+).*?max depth reached:\s+(\d+).*?best leaf score:\s+(\d+)/5",
                text,
                re.DOTALL,
            )
            if m:
                info["tree_size"] = int(m.group(1))
                info["max_depth"] = int(m.group(2))
                info["best_score"] = int(m.group(3))
        if "render_tree(" in cell.source:
            text = cell_output_text(cell)
            lines = re.findall(r"[⭐\s]+\[d=(\d+)\s+s=(\d+)/5\s+id=(\d+)\]\s+(.+?)(?=\n|\Z)", text)
            info["tree_lines"] = [
                {
                    "depth": int(d),
                    "score": int(s),
                    "id": int(i),
                    "content": _normalize_ws(c)[:140],
                }
                for d, s, i, c in lines
            ]
        if "llama_result = llama_arch.run" in cell.source:
            text = cell_output_text(cell)
            m = re.search(r"Score distribution \(Llama,[^)]+\):\s+\[([\d, ]+)\]", text)
            if m:
                info["llama_scores"] = [int(x) for x in m.group(1).split(",") if x.strip()]
    # Also derive reasoning-model scores from the tree
    info["scores_dist"] = sorted(
        [t["score"] for t in info["tree_lines"] if t["depth"] > 0], reverse=True
    )
    return info


def make_commentary(info: dict[str, object]) -> str:
    tree_size = info.get("tree_size", 0)
    max_d = info.get("max_depth", 0)
    best = info.get("best_score", 0)
    lines: list[dict] = info.get("tree_lines", [])  # type: ignore[assignment]
    answer: str = info.get("answer", "")  # type: ignore[assignment]
    scores: list[int] = info.get("scores_dist", [])  # type: ignore[assignment]
    llama_scores: list[int] = info.get("llama_scores", [])  # type: ignore[assignment]

    # Distribution analysis
    from collections import Counter

    sc_counts = Counter(scores)
    sc_table = "\n".join(f"| {s}/5 | {n} |" for s, n in sorted(sc_counts.items(), reverse=True))

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ").strip()

    sample_table = (
        "\n".join(
            f"| {l['depth']} | {l['score']}/5 | {l['id']} | {esc(l['content'])[:120]}{'…' if len(l['content']) > 120 else ''} |"
            for l in lines[:12]
        )
        if lines
        else "| — | — | — | _(no tree lines captured)_ |"
    )

    obs: list[str] = []
    if scores and len(set(scores)) == 1:
        obs.append(
            f"**Evaluator was lenient** — every thought scored **{scores[0]}/5**. "
            "Beam search has no signal to prune on; the tree degenerates into "
            "exhaustive expansion. Mitigation: tighten the `_ThoughtScore` rubric "
            "to *'reserve 5/5 for genuinely excellent thoughts'*, or use a "
            "different model in the evaluator seat."
        )
    elif scores:
        spread = max(scores) - min(scores)
        if spread >= 2:
            obs.append(
                f"**Healthy score spread** ({min(scores)}-{max(scores)}/5). "
                "The evaluator is genuinely discriminating between branches, "
                "which means beam search is doing real work."
            )
        else:
            obs.append(
                f"**Narrow score spread** ({min(scores)}-{max(scores)}/5). "
                "Some discrimination but not much. A stricter rubric would help."
            )

    if tree_size > 0 and max_d > 0:
        ideal = 1 + 3 * 2 * max_d  # root + branching=3 × beam=2 × depth
        if tree_size < ideal * 0.5:
            obs.append(
                f"**Tree pruned aggressively**: {tree_size} thoughts captured for "
                f"max_depth={max_d}. Beam search killed off most branches early."
            )

    # Compare reasoning vs non-reasoning evaluator
    if llama_scores and scores:
        reasoning_spread = max(scores) - min(scores) if scores else 0
        llama_spread = max(llama_scores) - min(llama_scores) if llama_scores else 0
        if reasoning_spread > llama_spread:
            obs.append(
                f"**Reasoning model gave more nuanced scores** "
                f"({len(set(scores))} distinct values) than Llama "
                f"({len(set(llama_scores))} distinct values). One reason the "
                "repo defaults ToT to a reasoning model."
            )
        elif llama_spread > reasoning_spread:
            obs.append(
                f"**Llama gave more spread scores** than the reasoning model on "
                "this run — surprising. May be noise; re-run to verify."
            )

    if not obs:
        obs.append("- No notable patterns. Re-run to see if it's repeatable.")

    obs_block = "\n\n".join(f"- {o}" for o in obs)

    answer_block = (
        "> " + (answer[:600].replace("\n", "\n> ") if answer else "_(no answer captured)_")
        + ("…" if len(answer) > 600 else "")
    )

    return f"""## 9 · What we just observed

The cells above ran a 3-deep, 3-wide beam search with `beam_width=2` against **Llama 3.3** on the **Game of 24** puzzle (objective scoring forces real discrimination).

### 9.1 · Quantitative summary

| Metric | Value |
|---|---|
| Tree size | **{tree_size}** thoughts |
| Max depth reached | **{max_d}** / 3 |
| Best leaf score | **{best}**/5 |
| Score distribution (non-root) | {scores} |
| Distinct score values | {len(set(scores)) if scores else 0} |

### 9.2 · Score distribution table

| Score | Count |
|---|---|
{sc_table}

### 9.3 · A sample of captured thoughts

| Depth | Score | id | Content snippet |
|---|---|---|---|
{sample_table}

### 9.4 · Patterns surfaced in this run

{obs_block}

### 9.5 · Final answer (verbatim)

{answer_block}

### 9.6 · The takeaway

A *healthy* ToT run has:

1. **A spread of scores** across thoughts (2-5 range, not all 5/5).
2. **The tree actually pruned** — at least one low-scoring branch killed off, not just exhaustive expansion.
3. **The best-leaf score visibly higher** than the average score.
4. **A final answer that obviously synthesizes the winning path**, not just paraphrases the task.

When the evaluator is lenient (everything 5/5), the search reduces to brute-force expansion at high cost — see § 11.1 for the mitigation. The reasoning-model default helps but doesn't solve this entirely."""


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
        f"tailored section 9: tree_size={info['tree_size']}, "
        f"max_depth={info['max_depth']}, scores_dist={info['scores_dist']}, "
        f"llama_scores={info.get('llama_scores')}"
    )


if __name__ == "__main__":
    main()
