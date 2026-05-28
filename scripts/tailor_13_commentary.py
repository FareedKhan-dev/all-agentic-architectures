"""Post-process notebook 13: rewrite § 9 against the Ensemble captured run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "13_ensemble.ipynb"
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
        "aggregated": "",
        "voters": [],
        "confidences": [],
        "spread": 0,
        "opinions": [],
    }
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "result = arch.run(TASK)" in cell.source:
            text = cell_output_text(cell)
            m = re.search(
                r"Aggregated answer.*?\n(.+?)\nVOTERS:",
                text,
                re.DOTALL,
            )
            if m:
                info["aggregated"] = _normalize_ws(m.group(1))
            m = re.search(r"VOTERS:\s+\[([^\]]+)\]", text)
            if m:
                info["voters"] = [v.strip().strip("'\"") for v in m.group(1).split(",") if v.strip()]
            m = re.search(r"CONFIDENCES:\s+\[([\d, ]+)\]", text)
            if m:
                info["confidences"] = [int(x) for x in m.group(1).split(",") if x.strip()]
            m = re.search(r"CONFIDENCE_SPREAD:\s+(\d+)", text)
            if m:
                info["spread"] = int(m.group(1))
        if "for t in result.trace" in cell.source and "voter" in cell.source:
            text = cell_output_text(cell)
            blocks = re.findall(
                r"===\s+(\w+)\s+\(confidence\s+(\d+)/5\)\s+===\s*\n(.+?)(?=\n\s*[›>]\s+===|\Z)",
                text,
                re.DOTALL,
            )
            ops = []
            for voter, conf, body in blocks:
                bl_m = re.search(r"BOTTOM LINE:\s+(.+?)(?=\n\s*[›>]|\Z)", body, re.DOTALL)
                ops.append({
                    "voter": voter.lower(),
                    "confidence": int(conf),
                    "bottom_line": _normalize_ws(bl_m.group(1)) if bl_m else "",
                })
            info["opinions"] = ops
    return info


def make_commentary(info: dict[str, object]) -> str:
    ops: list[dict] = info.get("opinions", [])  # type: ignore[assignment]
    voters: list[str] = info.get("voters", [])  # type: ignore[assignment]
    confs: list[int] = info.get("confidences", [])  # type: ignore[assignment]
    spread = info.get("spread", 0)
    aggregated: str = info.get("aggregated", "")  # type: ignore[assignment]

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ").strip()

    ops_table = (
        "\n".join(
            f"| {o['voter']} | {o['confidence']}/5 | {esc(o['bottom_line'])[:200]}{'…' if len(o['bottom_line']) > 200 else ''} |"
            for o in ops
        )
        if ops
        else "| — | — | _(no opinions captured)_ |"
    )

    # Detect genuine disagreement by looking for YES/NO contrast
    yes_no_words = {"yes", "no", "likely", "unlikely", "doubt", "support", "skeptic"}
    bottoms = [o["bottom_line"].lower() for o in ops]
    has_yes = any("yes" in b or "likely" in b or "will" in b for b in bottoms)
    has_no = any("no" in b or "doubt" in b or "unlikely" in b or "won't" in b for b in bottoms)
    genuine_disagreement = has_yes and has_no

    obs: list[str] = []
    if not ops:
        obs.append("**No voters captured** — vote step failed.")
    else:
        if spread == 0 and len(confs) > 1:
            obs.append(
                f"**Flat confidence scores** — all {len(confs)} voters reported "
                f"{confs[0]}/5 confidence. The familiar Llama-as-Scorer pathology "
                "(see Mental Loop nb 10 §9). Self-reported confidence is unreliable "
                "as a signal — focus on the CONTENT of each bottom-line instead."
            )
        elif spread > 0:
            obs.append(
                f"**Confidence spread {spread}** ({min(confs)}-{max(confs)}/5) — "
                "voters actually disagreed about how sure they were. Less common "
                "on instruction-tuned models — this is a healthy signal."
            )

        if genuine_disagreement:
            obs.append(
                "**Genuine perspective disagreement detected.** At least one "
                "voter said YES/likely AND another said NO/doubt. This is what "
                "Ensemble is FOR — different perspectives producing different "
                "directional answers on the same question."
            )
        elif len(ops) >= 2:
            unique_bottoms = len({o["bottom_line"] for o in ops})
            if unique_bottoms == 1:
                obs.append(
                    "**Hidden conformity** — all voters produced essentially the "
                    "same bottom line. Either the question wasn't really contested, "
                    "or the perspective prompts didn't activate distinct framings."
                )
            else:
                obs.append(
                    f"**Soft disagreement** — voters phrased the answer differently "
                    f"but didn't take opposing directional positions ({unique_bottoms} distinct bottom lines)."
                )

    # Aggregator quality check
    if aggregated and ops:
        agg_lower = aggregated.lower()
        preserves_dissent = any(
            phrase in agg_lower
            for phrase in ("however", "but", "on the other hand", "skeptic", "disagree", "uncertainty")
        )
        if preserves_dissent:
            obs.append(
                "**Aggregator preserved nuance** — used hedging language "
                "('however', 'uncertainty', etc.) in the synthesis, suggesting "
                "minority views weren't washed out."
            )
        else:
            obs.append(
                "**Aggregator washout suspected** — no hedging language in the "
                "synthesis. The minority voice may have been flattened — see § 11.1."
            )

    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns."

    agg_block = (
        "> " + (aggregated[:600].replace("\n", "\n> ") if aggregated else "_(no aggregated answer captured)_")
        + ("…" if len(aggregated) > 600 else "")
    )

    return f"""## 9 · What we just observed

The cells above ran 3 voters (analytical / skeptical / pragmatic) against the same contested forecasting question, then aggregated their opinions.

### 9.1 · Quantitative summary

| Metric | Value |
|---|---|
| Voters run | {len(ops)} |
| Confidence values | {confs} |
| Confidence spread | {spread} |
| Voters who answered YES/likely | {sum(1 for b in bottoms if 'yes' in b or 'likely' in b or 'will' in b)} |
| Voters who answered NO/doubt | {sum(1 for b in bottoms if 'no' in b or 'doubt' in b or 'unlikely' in b)} |

### 9.2 · Per-voter bottom-line answers

| Voter | Confidence | Bottom line |
|---|---|---|
{ops_table}

### 9.3 · Patterns surfaced in this run

{obs_block}

### 9.4 · The aggregated answer (verbatim)

{agg_block}

### 9.5 · The takeaway

A *healthy* Ensemble run has:

1. **Genuine disagreement** — at least 2 of K voters produce different directional answers.
2. **Aggregator preserves nuance** — hedging language ('however', 'on the other hand') in the synthesis.
3. **Confidence values are NOT the signal** — they're noisy. The bottom-line CONTENT discrimination matters.
4. **The synthesis is shorter than the sum of inputs** — extracts insight, doesn't just concatenate."""


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
    print(f"tailored section 9: {len(info['opinions'])} voter opinions, confidences {info['confidences']}")


if __name__ == "__main__":
    main()
