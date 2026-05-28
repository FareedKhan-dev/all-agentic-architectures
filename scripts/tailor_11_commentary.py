"""Post-process notebook 11: rewrite § 9 against the Meta-Controller captured run."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "11_meta_controller.ipynb"
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
    info: dict[str, object] = {"routes": []}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "TASKS = [" in cell.source:
            text = cell_output_text(cell)
            # Pattern: "[Qn] → routed to: ARCH" body, "reason BODY", "output BODY"
            blocks = re.findall(
                r"\[(Q\d+)\]\s+→\s+routed\s+to:\s+(\w+)\s*\n(.+?)(?=\n\s*[›>]\s+\[Q\d+\]|\nRouting distribution|\Z)",
                text,
                re.DOTALL,
            )
            routes = []
            for tag, arch_name, body in blocks:
                reason_m = re.search(r"reason\s*\n(.+?)(?=\n\s*[›>])", body, re.DOTALL)
                reason = _normalize_ws(reason_m.group(1)) if reason_m else ""
                output_m = re.search(r"output \(truncated\)\s*\n(.+?)(?=\n\s*[›>]|\Z)", body, re.DOTALL)
                output = _normalize_ws(output_m.group(1)) if output_m else ""
                # Original task text by tag — pull from source
                task_m = re.search(rf'\("{tag}",\s*"([^"]+)"\)', cell.source)
                task = task_m.group(1) if task_m else ""
                routes.append({
                    "tag": tag,
                    "task": task[:120],
                    "arch": arch_name.lower(),
                    "reason": reason[:200],
                    "output": output[:200],
                })
            info["routes"] = routes
    return info


def make_commentary(info: dict[str, object]) -> str:
    routes: list[dict] = info.get("routes", [])  # type: ignore[assignment]

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ").strip()

    routes_table = (
        "\n".join(
            f"| {r['tag']} | `{r['arch']}` | {esc(r['task'])[:80]} | {esc(r['reason'])[:150]} |"
            for r in routes
        )
        if routes
        else "| — | — | _(no routes captured)_ | — |"
    )

    arch_counts = Counter(r["arch"] for r in routes)
    dist_table = "\n".join(f"| `{a}` | {n} |" for a, n in arch_counts.most_common())

    obs: list[str] = []
    n_distinct = len(arch_counts)
    if n_distinct == 1:
        winner = list(arch_counts.keys())[0]
        obs.append(
            f"**Severe preference bias** — all {len(routes)} tasks routed to "
            f"`{winner}`. The router is essentially a hardcoded `{winner}` "
            "with extra cost. Either the architecture descriptions are too "
            "similar, or Llama has a strong default toward this architecture. "
            "Mitigation: tighten architecture descriptions to be mutually exclusive."
        )
    elif n_distinct == 2:
        obs.append(
            f"**Partial routing diversity** — {len(routes)} tasks split across "
            f"{n_distinct} architectures. Some variety but not full discrimination. "
            "Re-run with more diverse tasks to confirm."
        )
    elif n_distinct >= 3:
        obs.append(
            f"**Healthy routing diversity** — {len(routes)} tasks split across "
            f"{n_distinct} distinct architectures. The router genuinely "
            "discriminates between task shapes."
        )

    # Per-route specific checks
    expected_routes = {
        "Q1": "tool_use",   # one-shot lookup
        "Q2": "reflection", # code with quality emphasis
        "Q3": "planning",   # multi-aspect comparison
        "Q4": "react",      # multi-hop chain
    }
    correct = 0
    wrong_routes = []
    for r in routes:
        expected = expected_routes.get(r["tag"])
        if expected and r["arch"] == expected:
            correct += 1
        elif expected:
            wrong_routes.append(f"{r['tag']} → `{r['arch']}` (expected `{expected}`)")
    if routes and expected_routes:
        obs.append(
            f"**Routing accuracy against author's expectations**: "
            f"{correct}/{len(routes)} tasks routed as the author would have. "
            + (f"Mismatches: {', '.join(wrong_routes)}." if wrong_routes else "")
        )

    obs_block = "\n\n".join(f"- {o}" for o in obs)

    return f"""## 9 · What we just observed

The cells above ran Meta-Controller against **4 deliberately diverse tasks** to see whether the LLM router actually discriminates between architecture shapes.

### 9.1 · Routing decisions captured live

| Task | Routed to | Task preview | Router reason |
|---|---|---|---|
{routes_table}

### 9.2 · Architecture-choice distribution

| Architecture | Times chosen |
|---|---|
{dist_table}

### 9.3 · Patterns surfaced in this run

{obs_block}

### 9.4 · The takeaway

A *healthy* Meta-Controller run has:

1. **Routing diversity** — different task shapes go to different architectures.
2. **Specific reasons** — the router justifies each choice with task-specific language, not generic praise.
3. **Output quality matches the route** — Reflection routes produce polished code; ToolUse routes produce concise factual answers; Planning routes produce structured multi-part outputs.

When all 4 tasks route to the same architecture, Meta-Controller's overhead is wasted — just call that architecture directly. The router's value is **only** realized when traffic is diverse and routing distributes work."""


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
    print(f"tailored section 9: {len(info['routes'])} routes captured")


if __name__ == "__main__":
    main()
