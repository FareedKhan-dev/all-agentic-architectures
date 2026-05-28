"""Post-process notebook 12: rewrite § 9 against the Graph Memory captured run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "12_graph_memory.ipynb"
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
        "triples_extracted": 0,
        "qa_pairs": [],
        "nodes": 0,
        "edges": 0,
        "all_triples": [],
    }
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "CORPUS = " in cell.source:
            text = cell_output_text(cell)
            m = re.search(r"extracted:\s+(\d+)\s+triples", text)
            if m:
                info["triples_extracted"] = int(m.group(1))
            # Per-question blocks
            qa = re.findall(
                r"Q:\s+(.+?)\s*\n.*?A:\s+(.+?)(?=\n\s+\(matched|\Z)",
                text,
                re.DOTALL,
            )
            stats = re.findall(
                r"\(matched\s+(\d+)\s+entity/ies,\s+used\s+(\d+)\s+fact\(s\)\)",
                text,
            )
            pairs = []
            for i, (q, a) in enumerate(qa):
                matched = int(stats[i][0]) if i < len(stats) else 0
                used = int(stats[i][1]) if i < len(stats) else 0
                pairs.append({
                    "q": _normalize_ws(q),
                    "a": _normalize_ws(a),
                    "matched": matched,
                    "facts": used,
                })
            info["qa_pairs"] = pairs
        if "All triples in the knowledge graph" in cell.source:
            text = cell_output_text(cell)
            m = re.search(r"Nodes:\s+(\d+)", text)
            if m:
                info["nodes"] = int(m.group(1))
            m = re.search(r"Edges:\s+(\d+)", text)
            if m:
                info["edges"] = int(m.group(1))
            triples = re.findall(r"\((.+?)\)\s+--\[(.+?)\]-->\s+\((.+?)\)", text)
            info["all_triples"] = [{"s": s.strip(), "p": p.strip(), "o": o.strip()} for s, p, o in triples]
    return info


def make_commentary(info: dict[str, object]) -> str:
    pairs: list[dict] = info.get("qa_pairs", [])  # type: ignore[assignment]
    triples: list[dict] = info.get("all_triples", [])  # type: ignore[assignment]
    n_triples = info.get("triples_extracted", 0)
    nodes = info.get("nodes", 0)
    edges = info.get("edges", 0)

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ").strip()

    qa_table = (
        "\n".join(
            f"| {i+1} | {esc(p['q'])[:60]} | {p['matched']} | {p['facts']} | {esc(p['a'])[:200]}{'…' if len(p['a']) > 200 else ''} |"
            for i, p in enumerate(pairs)
        )
        if pairs
        else "| — | _(no Q&A captured)_ | — | — | — |"
    )

    triples_sample = (
        "\n".join(f"| {t['s']} | {t['p']} | {t['o']} |" for t in triples[:15])
        if triples
        else "| — | — | _(no triples captured)_ |"
    )

    # Predicate vocabulary check
    predicates = [t["p"] for t in triples]
    pred_set = set(predicates)
    fragmented_pairs = []
    for p1 in pred_set:
        for p2 in pred_set:
            if p1 < p2 and (p1 in p2 or p2 in p1 or _are_synonyms(p1, p2)):
                fragmented_pairs.append((p1, p2))

    obs: list[str] = []
    if n_triples == 0:
        obs.append("**Zero triples extracted** — extractor failed silently. Check schema validation.")
    elif n_triples < 5:
        obs.append(f"**Sparse extraction** ({n_triples} triples). Probably too conservative; tighten the schema description to be more exhaustive.")
    else:
        obs.append(f"**Healthy extraction**: {n_triples} triples from the ~10-sentence corpus ({n_triples / 10:.1f} triples/sentence).")

    # Q&A success
    q_with_facts = sum(1 for p in pairs if p["facts"] > 0)
    if pairs:
        obs.append(
            f"**Entity-match hit rate**: {q_with_facts}/{len(pairs)} questions matched ≥1 entity. "
            + ("This is solid — most questions found their referent in the graph." if q_with_facts == len(pairs)
               else "Some questions failed entity match. See § 11.1 for the brittle-matching pathology.")
        )

    if fragmented_pairs:
        obs.append(
            f"**Predicate fragmentation detected**: {len(fragmented_pairs)} predicate pair(s) look "
            f"like synonyms (e.g. `{fragmented_pairs[0][0]}` ↔ `{fragmented_pairs[0][1]}`). "
            "In a large corpus this fragments the graph. Production version needs predicate normalisation."
        )

    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns."

    return f"""## 9 · What we just observed

The cells above ran ingest + 5 query rounds against the Anthropic / OpenAI corpus.

### 9.1 · Graph statistics

| Metric | Value |
|---|---|
| Triples extracted from corpus | **{n_triples}** |
| Total nodes in graph | **{nodes}** |
| Total edges in graph | **{edges}** |
| Distinct predicates | **{len(pred_set)}** |

### 9.2 · Per-question results

| # | Question | Entities matched | Facts used | Answer (truncated) |
|---|---|---|---|---|
{qa_table}

### 9.3 · Sample of stored triples

| Subject | Predicate | Object |
|---|---|---|
{triples_sample}

### 9.4 · Patterns surfaced in this run

{obs_block}

### 9.5 · The takeaway

A *healthy* Graph Memory run has:

1. **Triples-per-sentence ≥ 1** during ingest.
2. **Every question matches ≥1 entity** during query.
3. **Answers cite specific triples** (not "I think" / "probably").
4. **Distinct predicates is small** (<25 for a small corpus) — predicate fragmentation kept under control.

When entity match fails, the answer either says "no information" (good) or hallucinates from parametric knowledge (bad). The synthesis prompt explicitly forbids the latter — see the verbatim answers above to verify."""


def _are_synonyms(p1: str, p2: str) -> bool:
    """Crude synonym check for predicate fragmentation detection."""
    s1 = set(p1.replace("_", " ").split())
    s2 = set(p2.replace("_", " ").split())
    return bool(s1 & s2) and (s1 != s2)


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
        f"tailored section 9: {info['triples_extracted']} triples, "
        f"{len(info['qa_pairs'])} Q&A pairs, {info['nodes']} nodes"
    )


if __name__ == "__main__":
    main()
