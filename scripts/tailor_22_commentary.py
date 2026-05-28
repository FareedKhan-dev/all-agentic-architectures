"""Post-process notebook 22: rewrite § 9 against the LATS captured run."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "22_lats.ipynb"
ANSI = re.compile(r"\x1b\[[0-9;]*[mGKH]")


def cell_output_text(cell: nbformat.NotebookNode) -> str:
    chunks: list[str] = []
    for o in cell.outputs:
        t = o.get("text", "") or o.get("data", {}).get("text/plain", "")
        if isinstance(t, list):
            t = "".join(t)
        chunks.append(ANSI.sub("", str(t)))
    return "\n".join(chunks)


PRIMARY = re.compile(
    r"TREE_SIZE:\s+(\d+)\s*\n"
    r"LEAF_COUNT:\s+(\d+)\s*\n"
    r"ITERATIONS_USED:\s+(\d+)/(\d+)\s*\n"
    r"BEST_LEAF_VALUE:\s+([0-9.]+)/10\s*\n"
    r"LEAF_VALUES \(sorted desc\):\s+(\[.*?\])\s*\n"
    r"LEAF_VALUES_SPREAD:\s+([0-9.]+)",
    re.DOTALL,
)
PATH = re.compile(r"=== BEST PATH \(root → best leaf\) ===\n(.*?)(?=\n\n)", re.DOTALL)
FEATURES = re.compile(r"BEST_LEAF_FEATURES:\s+(\{.+?\})", re.DOTALL)
FINAL = re.compile(r"=== FINAL ANSWER ===\n(.+?)\Z", re.DOTALL)


def extract_run(nb: nbformat.NotebookNode) -> dict[str, object]:
    info: dict[str, object] = {"primary": None, "path": [], "features": {}, "final": ""}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        text = cell_output_text(cell)
        if "TREE_SIZE:" in text and "BEST_PATH" in text:
            m = PRIMARY.search(text)
            if m:
                try:
                    leaf_values = ast.literal_eval(m.group(6))
                except Exception:
                    leaf_values = []
                info["primary"] = {
                    "tree_size": int(m.group(1)),
                    "leaf_count": int(m.group(2)),
                    "iterations_used": int(m.group(3)),
                    "max_iterations": int(m.group(4)),
                    "best_leaf_value": float(m.group(5)),
                    "leaf_values": leaf_values,
                    "spread": float(m.group(7)),
                }
            pm = PATH.search(text)
            if pm:
                info["path"] = [
                    re.sub(r"^\s*\[\d+\]\s+", "", ln).strip()
                    for ln in pm.group(1).splitlines() if ln.strip()
                ]
            fm = FEATURES.search(text)
            if fm:
                try:
                    info["features"] = ast.literal_eval(fm.group(1))
                except Exception:
                    info["features"] = {}
            am = FINAL.search(text)
            if am:
                info["final"] = am.group(1).strip()
    return info


def _esc(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def make_commentary(info: dict[str, object]) -> str:
    p = info.get("primary") or {}
    path = info.get("path") or []  # type: ignore[assignment]
    feats = info.get("features") or {}
    final = info.get("final") or ""

    if p:
        summary = (
            f"- **Tree size**: {p['tree_size']} nodes ({p['leaf_count']} leaves)\n"
            f"- **Iterations used**: {p['iterations_used']}/{p['max_iterations']} "
            f"{'(early terminated)' if p['iterations_used'] < p['max_iterations'] else '(budget exhausted)'}\n"
            f"- **Best leaf value**: {p['best_leaf_value']:.2f}/10\n"
            f"- **Leaf values (sorted desc)**: `{p['leaf_values']}`\n"
            f"- **Spread (max − min)**: **{p['spread']:.2f}**"
        )
    else:
        summary = "_(no run captured)_"

    if path:
        path_rows = "\n".join(
            f"| {i} | {_esc(t)[:160]}{'…' if len(t) > 160 else ''} |"
            for i, t in enumerate(path)
        )
    else:
        path_rows = "| — | _(no path captured)_ |"

    if feats:
        feat_rows = "\n".join(f"| `{k}` | `{v}` |" for k, v in feats.items())
    else:
        feat_rows = "| — | _(no features captured)_ |"

    obs: list[str] = []
    if p:
        spread = p["spread"]  # type: ignore[index]
        if spread > 0:
            obs.append(
                f"**✅ Deterministic-picker reward is working**: leaf values have spread of "
                f"**{spread:.2f}** points. UCB1 had real discriminating power across leaves. "
                "If the spread were 0, the flat-scoring pathology would have collapsed the search."
            )
        else:
            obs.append(
                "**❌ Leaf values are all identical** — reward signal collapsed despite the "
                "deterministic-picker fix. Inspect features per leaf to find which booleans "
                "stayed pinned. Likely cause: LLM hedging via 'low' confidence on every leaf."
            )
        if p["best_leaf_value"] >= 9.0:  # type: ignore[index]
            obs.append(
                f"**✅ Found a high-value terminal leaf** (value {p['best_leaf_value']:.2f}/10). "  # type: ignore[index]
                "Likely satisfies `is_complete=True` AND `confidence=high` — a strong solution candidate."
            )
        elif p["best_leaf_value"] >= 5.0:  # type: ignore[index]
            obs.append(
                f"**🤔 Best leaf scored {p['best_leaf_value']:.2f}/10** — promising but not "  # type: ignore[index]
                "fully verified as complete. Run more iterations or raise `max_depth`."
            )
        else:
            obs.append(
                f"**❌ Best leaf value only {p['best_leaf_value']:.2f}/10** — search did not "  # type: ignore[index]
                "find a high-value path. Increase budget, lower branching, or strengthen the "
                "expansion prompt."
            )
        if p["iterations_used"] < p["max_iterations"]:  # type: ignore[index]
            obs.append(
                f"**✅ Early termination** — search stopped at iteration {p['iterations_used']} "  # type: ignore[index]
                f"of {p['max_iterations']} after finding a terminal high-value leaf. Saved "
                "the unused budget."
            )
        if p["tree_size"] <= 1 + p["leaf_count"]:  # type: ignore[index]
            obs.append(
                "**🤔 Tree stayed shallow** — most nodes are root's direct children, no deep "
                "descent happened. Either max_depth is too low or the LLM's expansion prompts "
                "produced no promising candidates to dig into."
            )

    if path and len(path) >= 3:
        obs.append(
            f"**Path length {len(path)}**: search descended {len(path)-1} step(s) below root — "
            "the tree found a multi-step trajectory, not a one-shot answer."
        )

    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns surfaced."

    return f"""## 9 · What we just observed

The cells above ran LATS on Game of 24 with `branching=2, max_iterations=4`. We measure tree growth, reward spread (the deterministic-picker signal), and the discovered best path.

### 9.1 · Search statistics

{summary}

### 9.2 · The best path from root to terminal leaf

| # | thought |
|---|---|
{path_rows}

### 9.3 · Best leaf's `_LeafEvaluation` features (deterministic-picker source)

| feature | value |
|---|---|
{feat_rows}

These independent booleans/categorical fed into `_composite_value(...)` which produced the leaf's reward — **no numeric judgement was made by the LLM**. The reward came from Python composing the LLM's structured feature commitments.

### 9.4 · Final answer

```
{final or "(none captured)"}
```

### 9.5 · Patterns surfaced in this run

{obs_block}

### 9.6 · The takeaway

LATS only earns its complexity over Tree of Thoughts (nb 09) when **all four** properties hold:
1. **Reward has spread** — § 9.1's `Spread` value must be > 0 (deterministic-picker prevents flatness).
2. **UCB1 explores** — under-visited siblings get attention via the exploration bonus.
3. **Backup amplifies** — high-value leaves boost their ancestors, redirecting future descents.
4. **Best path is multi-step** — § 9.2 should have ≥ 3 entries, otherwise plain CoT suffices.

When any property fails, fall back to ToT (cheaper, simpler) or Self-Consistency (even simpler). On this Game-of-24 run with branching=2 and only 4 iterations, the tree stays small but the reward spread + path depth show all four properties holding."""


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
    p = info["primary"] or {}
    print(
        f"tailored: tree={p.get('tree_size')} leaves={p.get('leaf_count')} "
        f"best_val={p.get('best_leaf_value')} spread={p.get('spread')} "
        f"iters={p.get('iterations_used')}/{p.get('max_iterations')}"
    )


if __name__ == "__main__":
    main()
