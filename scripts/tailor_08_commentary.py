"""Post-process notebook 08: rewrite § 9 against the Episodic+Semantic run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "08_episodic_semantic_memory.ipynb"
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
        "onboarding_turns": [],
        "recall_question": "",
        "recall_answer": "",
        "facts_recalled": 0,
        "episodes_recalled": 0,
        "triples": [],
        "episode_snippets": [],
    }
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "ONBOARDING = [" in cell.source:
            text = cell_output_text(cell)
            # Onboarding turns: capture facts extracted / total entities
            turns = re.findall(
                r"\[Turn (\d+)\] USER\s*\n(.+?)\[Turn \d+\] ASSISTANT:\s*(.+?)\n.*?facts extracted=(\d+).*?total entities stored=(\d+).*?total episodes=(\d+)",
                text,
                re.DOTALL,
            )
            info["onboarding_turns"] = [
                {
                    "turn": int(t),
                    "user": _normalize_ws(u),
                    "assistant": _normalize_ws(a),
                    "facts_extracted": int(f),
                    "entities_stored": int(e),
                    "episodes_stored": int(ep),
                }
                for t, u, a, f, e, ep in turns
            ]
            # Recall Q/A — print_md renders Markdown so `**Q:**` becomes bold-formatted
            # plain `Q:` in the captured stdout (the ** markers are stripped by the renderer).
            m = re.search(
                r"\b(?:\*\*)?Q:\s*(?:\*\*)?\s*(.+?)\n\s*\n\s*(?:\*\*)?A:\s*(?:\*\*)?\s*(.+?)(?=\n\s*episodes\s+recalled|\n\s*[─-╿]|\Z)",
                text, re.DOTALL,
            )
            if m:
                info["recall_question"] = _normalize_ws(m.group(1))
                info["recall_answer"] = _normalize_ws(m.group(2))
            m = re.search(r"episodes recalled:\s*(\d+).*?facts recalled:\s*(\d+)", text, re.DOTALL)
            if m:
                info["episodes_recalled"] = int(m.group(1))
                info["facts_recalled"] = int(m.group(2))
        if "All triples in semantic memory" in cell.source:
            text = cell_output_text(cell)
            triples = re.findall(r"\((.+?)\)\s+--\[(.+?)\]-->\s+\((.+?)\)", text)
            info["triples"] = [{"s": s.strip(), "p": p.strip(), "o": o.strip()} for s, p, o in triples]
            eps = re.findall(r"\[\d+\]\s+(.+?)(?=\n|\Z)", text)
            info["episode_snippets"] = [_normalize_ws(e)[:120] for e in eps]
    return info


def make_commentary(info: dict[str, object]) -> str:
    turns: list[dict] = info.get("onboarding_turns", [])  # type: ignore[assignment]
    triples: list[dict] = info.get("triples", [])  # type: ignore[assignment]
    eps: list[str] = info.get("episode_snippets", [])  # type: ignore[assignment]
    recall_q: str = info.get("recall_question", "")  # type: ignore[assignment]
    recall_a: str = info.get("recall_answer", "")  # type: ignore[assignment]
    facts_recalled = info.get("facts_recalled", 0)
    episodes_recalled = info.get("episodes_recalled", 0)

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ").strip()

    growth_table = (
        "\n".join(
            f"| {t['turn']} | {esc(t['user'])[:80]}{'…' if len(t['user']) > 80 else ''} | {t['facts_extracted']} | {t['entities_stored']} | {t['episodes_stored']} |"
            for t in turns
        )
        if turns
        else "| — | — | — | — | — |"
    )

    triples_table = (
        "\n".join(f"| {i+1} | {t['s']} | {t['p']} | {t['o']} |" for i, t in enumerate(triples[:20]))
        if triples
        else "| — | _(no triples extracted)_ | — | — |"
    )

    obs: list[str] = []

    # Check growth
    if turns:
        final_entities = turns[-1]["entities_stored"]
        if final_entities == 0:
            obs.append(
                "**Fact-extractor produced zero entities** across all onboarding turns. "
                "The structured-output extractor failed silently — check the schema and "
                "Field descriptions in `_ExtractedFacts`."
            )
        elif final_entities < len(turns) * 2:
            obs.append(
                f"**Light extraction**: only {final_entities} entities total after {len(turns)} "
                "onboarding turns. The extractor is being too conservative — typical onboarding "
                "should yield 2-4 entities per turn."
            )
        else:
            obs.append(
                f"**Healthy extraction**: {final_entities} entities stored after "
                f"{len(turns)} onboarding turns "
                f"({final_entities / len(turns):.1f} entities/turn average)."
            )

    # Check recall worked
    if facts_recalled > 0:
        obs.append(
            f"**Semantic recall worked**: the recall query retrieved {facts_recalled} fact(s) "
            "from the graph. This is the *single most important signal* that the dual-memory "
            "design is doing its job — the answer is grounded in stored facts, not hallucination."
        )
    else:
        obs.append(
            "**Semantic recall FAILED**: 0 facts retrieved for the recall query. Either "
            "(a) no facts were stored, (b) the entity-match heuristic missed the matches "
            "(case sensitivity, synonyms), or (c) the query was too generic. The fallback "
            "'include all entities if query is short' kicked in but found nothing useful."
        )

    if episodes_recalled > 0:
        obs.append(
            f"**Episodic recall worked**: vector similarity returned {episodes_recalled} "
            "past episode(s) most similar to the recall query."
        )

    # Recall answer grounding check
    if recall_a:
        recall_lower = recall_a.lower()
        triple_keywords = {t["o"].lower() for t in triples if len(t["o"]) > 2}
        matches = sum(1 for kw in triple_keywords if kw in recall_lower)
        if triple_keywords:
            if matches / len(triple_keywords) >= 0.5:
                obs.append(
                    f"**Recall answer is well-grounded**: {matches}/{len(triple_keywords)} "
                    "of the stored fact-objects appear in the recall answer. The agent "
                    "is genuinely answering from memory."
                )
            else:
                obs.append(
                    f"**Recall answer is partially grounded**: only {matches}/{len(triple_keywords)} "
                    "of the stored facts appear in the recall answer. Either the answer is "
                    "filtering / summarising, or it's drifting toward parametric knowledge."
                )

    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns surfaced."

    recall_block = (
        f"**Q:** {recall_q}\n\n"
        f"**A:** {recall_a}" if recall_q else "_(recall not captured)_"
    )

    return f"""## 9 · What we just observed

The cells above ran **4 sequential calls on the same `arch` instance** — 3 onboarding turns that populated memory, then a recall turn that queried it.

### 9.1 · Memory growth across onboarding turns

| Turn | User message (truncated) | Facts extracted | Entities stored | Episodes stored |
|---|---|---|---|---|
{growth_table}

### 9.2 · Final state of semantic memory (all triples)

| # | Subject | Predicate | Object |
|---|---|---|---|
{triples_table}

### 9.3 · Recall test

{recall_block}

- **`episodes_recalled`** = {episodes_recalled} (vector similarity hits over the {len(eps)} stored episodes)
- **`facts_recalled`** = {facts_recalled} (graph entity-match hits)

### 9.4 · Patterns surfaced in this run

{obs_block}

### 9.5 · The takeaway

A *healthy* dual-memory run looks like:

1. **Linear growth** of entities + episodes across onboarding turns.
2. **Recall query retrieves both** episodes (vector) and facts (graph).
3. **Recall answer mentions specific stored objects** — proof the agent is answering from memory, not training data.
4. **No duplicated triples** in the final graph (entity-match wasn't fooled by synonyms).

Compare those four signals to what you see above to judge memory quality. The single most important production metric is **fact-recall precision**: did the agent recall facts that are *actually* relevant to the query?"""


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
        f"tailored section 9: {len(info['onboarding_turns'])} onboarding turns, "
        f"{len(info['triples'])} triples, "
        f"{info['facts_recalled']} facts recalled, "
        f"{info['episodes_recalled']} episodes recalled"
    )


if __name__ == "__main__":
    main()
