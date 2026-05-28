"""Post-process notebook 15: rewrite § 9 against the RLHF captured run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "15_rlhf_self_improvement.ipynb"
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
        "tasks": [],
        "composite_scores": [],
        "llm_scores": [],
        "composite_spread": 0,
        "llm_spread": 0,
    }
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "TASKS = [" in cell.source and "TASK_TAG" in cell.source:
            text = cell_output_text(cell)
            blocks = re.findall(
                r"TASK_TAG:\s+(\w+)\s*\n"
                r"\s+COMPOSITE_SCORE.*?:\s+(\d+)/10\s*\n"
                r"\s+LLM_OVERALL_RAW:\s+(\d+)/10\s*\n"
                r"\s+features:\s+(.+?)\n"
                r"\s+archived=(\w+),\s+archive_size=(\d+)\s*\n"
                r"\s+output:\s+(.+?)(?=\n\s*TASK_TAG:|\nCOMPOSITE_SCORES|\Z)",
                text,
                re.DOTALL,
            )
            tasks = []
            for tag, composite, llm_raw, features, archived, archive_size, output in blocks:
                feat_dict: dict[str, str] = {}
                for kv in features.split(","):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        feat_dict[k.strip()] = v.strip()
                tasks.append({
                    "tag": tag,
                    "composite": int(composite),
                    "llm_raw": int(llm_raw),
                    "features": feat_dict,
                    "archived": archived == "True",
                    "archive_size": int(archive_size),
                    "output": _normalize_ws(output)[:200],
                })
            info["tasks"] = tasks
            # Aggregate spread summary line
            m = re.search(r"COMPOSITE_SCORES_PY:\s+\[([\d, ]+)\].*?spread=(\d+)", text)
            if m:
                info["composite_scores"] = [int(x) for x in m.group(1).split(",") if x.strip()]
                info["composite_spread"] = int(m.group(2))
            m = re.search(r"LLM_OVERALL_RAW:\s+\[([\d, ]+)\].*?spread=(\d+)", text)
            if m:
                info["llm_scores"] = [int(x) for x in m.group(1).split(",") if x.strip()]
                info["llm_spread"] = int(m.group(2))
    return info


def make_commentary(info: dict[str, object]) -> str:
    tasks: list[dict] = info.get("tasks", [])  # type: ignore[assignment]
    composite_scores: list[int] = info.get("composite_scores", [])  # type: ignore[assignment]
    llm_scores: list[int] = info.get("llm_scores", [])  # type: ignore[assignment]
    composite_spread = info.get("composite_spread", 0)
    llm_spread = info.get("llm_spread", 0)

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ").strip()

    def _feat_summary(f: dict[str, str]) -> str:
        order = ["is_on_brief", "word_count", "has_concrete_imagery", "avoids_cliches", "is_engaging"]
        return ", ".join(f"{k}={f.get(k, '?')}" for k in order if k in f)

    tasks_table = (
        "\n".join(
            f"| {t['tag']} | **{t['composite']}**/10 | {t['llm_raw']}/10 | "
            f"{'✓' if t['archived'] else '✗'} | {esc(_feat_summary(t['features']))} |"
            for t in tasks
        )
        if tasks
        else "| — | — | — | — | _(no tasks captured)_ |"
    )

    obs: list[str] = []

    if composite_scores and llm_scores:
        obs.append(
            f"**Python composite scores: {composite_scores} (spread {composite_spread})** "
            f"vs **LLM raw `overall_score`: {llm_scores} (spread {llm_spread})**. "
            + (
                f"Python's composite has WIDER spread than the LLM's raw score — "
                "the multi-dimensional decomposition produced more discrimination "
                "than the LLM was willing to commit to in its single `overall_score` field. "
                "This is the deterministic-scoring fix working as designed."
                if composite_spread > llm_spread
                else (
                    "Both scores are similarly spread on this task set. The "
                    "improvement isn't visible in score-spread because the tasks "
                    "happen to be objectively similar quality — but the score is "
                    "now **transparent** (you can see exactly which features earned "
                    "or lost points)."
                    if composite_spread == llm_spread
                    else "LLM raw score happens to spread wider than the composite "
                    "on this particular run — unusual; the composite is usually "
                    "wider because each feature is an independent commitment."
                )
            )
        )

    # Feature divergence check
    if tasks and tasks[0].get("features"):
        feature_sets = []
        for t in tasks:
            feats = t["features"]
            feature_sets.append(
                tuple(feats.get(k, "?") for k in
                      ("is_on_brief", "has_concrete_imagery", "avoids_cliches", "is_engaging"))
            )
        if len(set(feature_sets)) > 1:
            obs.append(
                f"**Feature divergence across tasks**: {len(set(feature_sets))} distinct feature "
                "patterns out of {len(tasks)}. Different tasks scored on different criteria — "
                "the multi-dim decomposition is doing real work."
            )
        elif len(set(feature_sets)) == 1:
            obs.append(
                f"**All {len(tasks)} tasks got the SAME feature pattern** — Llama gave "
                "identical booleans across the three tasks. This is the same flat-scoring "
                "pathology resurfacing at the feature level. The output is still "
                "explainable (you can see which features contributed) but the architecture "
                "isn't actually distinguishing the tasks. Use genuinely different "
                "task shapes (e.g., easy vs hard constraints) to force divergence."
            )

    # Archive correctness
    if tasks:
        archived_count = sum(1 for t in tasks if t["archived"])
        if archived_count == 0:
            obs.append("**Archive is empty** — no task crossed the threshold. Lower target_score, or improve outputs.")
        elif archived_count == len(tasks):
            obs.append(
                f"**All {len(tasks)} tasks archived** — happy path, but watch for "
                "sycophantic-editor pathology. If every output passes the bar regardless "
                "of obvious quality differences, raise target_score."
            )
        else:
            obs.append(
                f"**Selective archiving** ({archived_count}/{len(tasks)} accepted) — "
                "the threshold + composite signal genuinely discriminated."
            )

    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns."

    return f"""## 9 · What we just observed

The cells above ran **3 tasks of varying difficulty** through ONE `RLHFSelfImprovement` instance, with the **multi-dimensional deterministic-scoring fix** applied (see § 3.0).

### 9.1 · Per-task feature decomposition

| Tag | Python COMPOSITE | LLM `overall_score` | Archived? | Editor feature commitments |
|---|---|---|---|---|
{tasks_table}

### 9.2 · Score-spread comparison

| Source | Values | Spread (max−min) |
|---|---|---|
| **Python composite** (the deciding signal) | {composite_scores} | **{composite_spread}** |
| LLM raw `overall_score` (preserved, unused) | {llm_scores} | {llm_spread} |

### 9.3 · Patterns surfaced in this run

{obs_block}

### 9.4 · The takeaway

The multi-dimensional fix has three properties worth checking:

1. **Transparency** — every score has an explicit Python-side decomposition you can read.
2. **More spread than single-score** — usually, because 5 independent booleans diverge more than 1 numeric commitment compresses.
3. **Honest residual** — even with multi-dim, identical tasks get identical features. When that happens, the architecture is admitting "I can't distinguish these" rather than papering it over with a fake 9/10 vs 8/10."""


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
        f"tailored section 9: {len(info['tasks'])} tasks, "
        f"composite_scores={info['composite_scores']}, llm_scores={info['llm_scores']}"
    )


if __name__ == "__main__":
    main()
