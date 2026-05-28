"""Post-process notebook 16: rewrite § 9 against the Cellular Automata captured run."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "16_cellular_automata.ipynb"
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
    info: dict[str, object] = {"per_step_counts": [], "history_grids": []}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "INITIAL_GRID" in cell.source and "result = arch.run" in cell.source:
            text = cell_output_text(cell)
            # Extract grids — "Step N:" header, then consecutive lines containing `|`.
            # Rich may add leading/trailing whitespace per row.
            grid_blocks = re.findall(
                r"Step\s+(\d+):\s*\n((?:[ \t]*[a-z|]+[ \t]*\n){2,})",
                text,
                re.IGNORECASE,
            )
            info["history_grids"] = [
                {
                    "step": int(n),
                    "grid": [line.strip() for line in g.strip().split("\n") if "|" in line],
                }
                for n, g in grid_blocks
            ]
            # Per-step counts: "step N: {'tree': 12, ...}"
            count_lines = re.findall(r"step\s+(\d+):\s+(\{[^}]+\})", text)
            counts = []
            for n, dict_str in count_lines:
                try:
                    d = ast.literal_eval(dict_str)
                    counts.append({"step": int(n), "counts": d})
                except (SyntaxError, ValueError):
                    pass
            info["per_step_counts"] = counts
    return info


def make_commentary(info: dict[str, object]) -> str:
    history: list[dict] = info.get("history_grids", [])  # type: ignore[assignment]
    counts: list[dict] = info.get("per_step_counts", [])  # type: ignore[assignment]

    grids_block = (
        "\n".join(
            f"**Step {h['step']}**:\n```\n" + "\n".join(h["grid"]) + "\n```"
            for h in history
        )
        if history
        else "_(no history captured)_"
    )

    counts_table = (
        "\n".join(
            f"| {c['step']} | {c['counts'].get('tree', 0)} | {c['counts'].get('fire', 0)} | {c['counts'].get('ash', 0)} | {c['counts'].get('empty', 0)} |"
            for c in counts
        )
        if counts
        else "| — | — | — | — | — |"
    )

    obs: list[str] = []

    if counts:
        # Monotonicity checks for forest-fire rule
        trees = [c["counts"].get("tree", 0) for c in counts]
        ashes = [c["counts"].get("ash", 0) for c in counts]
        fires = [c["counts"].get("fire", 0) for c in counts]

        # Trees: should be monotone non-increasing
        if trees == sorted(trees, reverse=True):
            obs.append(
                f"**Tree count is monotone non-increasing**: {trees}. The rule "
                "'trees only convert to fire' was followed faithfully — no LLM "
                "hallucinated a fire-back-to-tree transition."
            )
        else:
            obs.append(
                f"**Tree count is NOT monotone non-increasing**: {trees}. The "
                "LLM hallucinated a fire-or-ash-back-to-tree transition somewhere. "
                "Rule violation — see § 11.1."
            )

        # Ash: should be monotone non-decreasing
        if ashes == sorted(ashes):
            obs.append(
                f"**Ash count is monotone non-decreasing**: {ashes}. Ash never "
                "reverts, as the rule specifies. Good signal that the rule was followed."
            )
        else:
            obs.append(
                f"**Ash count is NOT monotone non-decreasing**: {ashes}. Ash "
                "shouldn't decrease under the forest-fire rule. Rule violation."
            )

        # Fire: should rise then fall (peak)
        if len(fires) >= 3:
            peak = max(fires)
            peak_idx = fires.index(peak)
            if 0 < peak_idx < len(fires) - 1 or fires[-1] < peak:
                obs.append(
                    f"**Fire count peaks then declines**: {fires}. Classic forest-fire "
                    "emergent dynamic — fire spreads radially, then burns out as cells "
                    "convert to ash."
                )
            elif fires[-1] >= peak:
                obs.append(
                    f"**Fire count still rising at end**: {fires}. The simulation "
                    "needs more steps to reach burn-out. Try `max_steps=5`."
                )

    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns."

    return f"""## 9 · What we just observed

The cells above ran a 4×4 forest-fire CA for 3 steps — each step makes 16 LLM calls (one per cell), so this run cost ~48 calls total.

### 9.1 · Per-step state counts

| Step | tree | fire | ash | empty |
|---|---|---|---|---|
{counts_table}

### 9.2 · Grid evolution

{grids_block}

### 9.3 · Rule-violation checks

{obs_block}

### 9.4 · The takeaway

A *correctly* running forest-fire CA shows three signatures:

1. **Tree count monotone non-increasing** — trees only convert to fire, never the reverse.
2. **Ash count monotone non-decreasing** — ash is the absorbing state.
3. **Fire count rises then falls** — the wavefront moves outward, then burns through.

When the LLM hallucinates a rule violation (e.g., ash → tree), the macro counts violate one of these signatures. This is **a free correctness check** baked into the dynamics — no separate verifier needed.

For production CA simulations, validate the rule on a tiny grid first via these monotonicity checks, then compile the rule to hard-coded Python once you trust it."""


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
        f"tailored section 9: {len(info['per_step_counts'])} step-count records, "
        f"{len(info['history_grids'])} grids captured"
    )


if __name__ == "__main__":
    main()
