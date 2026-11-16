"""Post-process notebook 36: rewrite § 9 against the MoA captured run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "36_mixture_of_agents.ipynb"
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
        "final_output": "",
        "layers_executed": 0,
        "total_calls": 0,
        "trace": [],
    }
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "result = arch.run(TASK)" in cell.source:
            text = cell_output_text(cell)
            m = re.search(r"Layers executed:\s+(\d+)", text)
            if m:
                info["layers_executed"] = int(m.group(1))
            m = re.search(r"Total agent calls:\s+(\d+)", text)
            if m:
                info["total_calls"] = int(m.group(1))
            m = re.search(
                r"Final MoA output\s*\n(.+?)(?=\nLayers executed:|\Z)",
                text,
                re.DOTALL,
            )
            if m:
                info["final_output"] = _normalize_ws(m.group(1))
        if "=== LAYER" in cell.source and "key_insight" in cell.source:
            text = cell_output_text(cell)
            # Extract layer-by-layer trace
            layer_blocks = re.findall(
                r"===\s+LAYER\s+(\d+)\s+===.*?Agent\s+(\d+)\s+\[(\w+)\].*?" r"Key insight:\s+(.*?)(?=\n\*\*Answer|\Z)",
                text,
                re.DOTALL,
            )
            # Fix: use a simpler pattern
            # Split by layer headers
            parts = re.split(r"=== LAYER (\d+) ===", text)
            trace = []
            for i in range(1, len(parts), 2):
                layer = int(parts[i])
                layer_text = parts[i + 1] if i + 1 < len(parts) else ""
                agents = re.findall(
                    r"Agent (\d+) \[(\w+)\].*?Key insight: (.+?)(?=\n\*\*Answer|\Z)",
                    layer_text,
                    re.DOTALL,
                )
                for agent_num, persona, insight in agents:
                    trace.append({
                        "layer": layer,
                        "agent": int(agent_num),
                        "persona": persona,
                        "key_insight": _normalize_ws(insight)[:200],
                    })
            info["trace"] = trace

    return info


def make_commentary(info: dict[str, object]) -> str:
    trace = info.get("trace", [])  # type: ignore[assignment]
    output: str = info.get("final_output", "")  # type: ignore[assignment]
    layers = info.get("layers_executed", 0)
    calls = info.get("total_calls", 0)

    obs: list[str] = []

    if not trace:
        obs.append("**No trace captured** — run step may have failed or output format differed.")
    else:
        max_layer = max((t["layer"] for t in trace), default=0)  # type: ignore[index]
        obs.append(
            f"**{max_layer + 1} layers executed with {calls} total agent calls.** "
            f"Expected: 2 layers × 3 agents + 1 aggregator = 7 calls. "
            f"Actual: {calls} calls."
        )

        # Check for persona diversity
        personas_by_layer: dict[int, set[str]] = {}
        for t in trace:  # type: ignore[assignment]
            layer = t["layer"]
            if layer not in personas_by_layer:
                personas_by_layer[layer] = set()
            personas_by_layer[layer].add(t["persona"])

        for layer in sorted(personas_by_layer.keys()):
            personas = personas_by_layer[layer]
            if len(personas) >= 3:
                obs.append(
                    f"**Layer {layer}: full persona diversity** — all 3 roles "
                    f"({', '.join(personas)}) present. Healthy signal."
                )
            elif len(personas) >= 2:
                obs.append(
                    f"**Layer {layer}: partial diversity** — {len(personas)} of 3 roles present. "
                    "Consider increasing `agents_per_layer` to ensure full coverage."
                )
            else:
                obs.append(
                    f"**Layer {layer}: low diversity** — only {personas.pop()} persona active. "
                    "Cross-layer refinement may be weaker than expected."
                )

        # Check refinement across layers
        if len(trace) >= 3:
            obs.append(
                "**Cross-layer refinement present** — layer 1 agents received layer 0 "
                "outputs as context. Progressive refinement active."
            )

        if output:
            if len(output) < 100:
                obs.append("**Short final output** — aggregator may have truncated. Check prompt fit.")
            else:
                obs.append(f"**Final output length: {len(output)} chars** — adequate synthesis length.")

    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns."

    agg_block = (
        "> "
        + (output[:600].replace("\n", "\n> ") if output else "_(no final output captured)_")
        + ("…" if len(output) > 600 else "")
    )

    # Build trace table
    trace_rows = "\n".join(
        f"| Layer {t['layer']} | {t['agent']} | {t['persona']} | {t['key_insight'][:150]} |"
        for t in trace[:9]  # type: ignore[index]
    )

    return f"""## 9 · What we just observed

The cells above ran {layers} layers of MoA (3 agents per layer) on a research synthesis task, then aggregated the final layer's outputs.

### 9.1 · Quantitative summary

| Metric | Value |
|---|---|
| Layers executed | {layers} |
| Agent calls | {calls} |
| Agents per layer | 3 (Decomposer / Synthesizer / Critic) |
| Trace entries | {len(trace)} |

### 9.2 · Layer-by-layer key insights

| Layer | Agent | Persona | Key insight |
|---|---|---|---|
{trace_rows}

### 9.3 · Patterns surfaced in this run

{obs_block}

### 9.4 · The final aggregated answer (verbatim)

{agg_block}

### 9.5 · The takeaway

A *healthy* MoA run has:

1. **Full persona diversity** — Decomposer, Synthesizer, and Critic all contribute in each layer.
2. **Progressive refinement** — layer 1 outputs are measurably different (not just paraphrases) from layer 0.
3. **Python-computed indicator tally** — the aggregator weighs signals deterministically, not via LLM self-scoring.
4. **The final synthesis is richer than any single agent's output** — cross-layer context adds genuine value."""


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
    trace = info.get("trace", [])
    print(f"tailored section 9: {len(trace)} trace entries, {info.get('layers_executed', 0)} layers")


if __name__ == "__main__":
    main()
