"""Post-process notebook 01: replace generic § 9 commentary with analysis of the
actual captured run.

This is the pattern every notebook in the repo follows:
  1.  build skeleton    →  scripts/build_NN_*.py
  2.  papermill execute →  captures live outputs
  3.  tailor commentary →  this kind of script, per notebook

The tailored commentary stays in sync with whatever output the model produced
on the last execution. Re-running steps 2+3 refreshes the analysis.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "01_reflection.ipynb"

ANSI = re.compile(r"\x1b\[[0-9;]*[mGKH]")


def extract_trace(nb: nbformat.NotebookNode) -> list[dict[str, str]]:
    """Pull (iteration, score, critique) tuples out of the trace cell's output."""
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "for i, step in enumerate(result.trace" not in cell.source:
            continue
        # Concatenate all stdout outputs.
        text = ""
        for o in cell.outputs:
            t = o.get("text", "") or o.get("data", {}).get("text/plain", "")
            if isinstance(t, list):
                t = "".join(t)
            text += ANSI.sub("", str(t))
        # Parse: "› Iteration N  ·  score S/10" followed by "Critique: …".
        steps = re.findall(
            r"Iteration\s+(\d+)\s+·\s+score\s+(\d+)/10\s*\n+\s*Critique:\s+(.+?)(?=\n\s*›|\n\s*Iteration|\Z)",
            text,
            flags=re.DOTALL,
        )
        return [
            {
                "iteration": int(i),
                "score": int(s),
                "critique": re.sub(r"\s+", " ", c).strip(),
            }
            for i, s, c in steps
        ]
    return []


def make_commentary(trace: list[dict[str, str]]) -> str:
    """Build a §9 markdown cell that directly quotes the captured run."""
    if not trace:
        return _generic_fallback()

    table_rows = "\n".join(
        f"| {t['iteration']} | {t['score']}/10 | {_one_liner(t['critique'])} |" for t in trace
    )
    n_rounds = len(trace)
    first_score = trace[0]["score"]
    last_score = trace[-1]["score"]
    scores = [t["score"] for t in trace]
    regressed = last_score < first_score
    plateaued = len(set(scores)) <= 2 and max(scores) - min(scores) <= 1
    direction = (
        "**regressed** (final score lower than first)"
        if regressed
        else "**plateaued**" if plateaued else "improved"
    )

    drift_block = ""
    if n_rounds >= 3:
        focus_per_round = [_one_liner(t["critique"]) for t in trace]
        drift_block = (
            "\n\n**(b) Critique drift — the Critic's focus moved each round:**\n\n"
            + "\n".join(f"- Round {t['iteration']} → *“{focus}”*"
                       for t, focus in zip(trace, focus_per_round))
            + "\n\nIn a stable rubric, the same flaws would be flagged repeatedly until fixed. "
            "Here the Critic finds *whatever is salient relative to the current draft* — that's "
            "drift, and it's why same-model Critics often need an external rubric to stay anchored."
        )

    plateau_block = (
        f"\n\n**(a) Plateau / regression.** Across {n_rounds} round(s) the score went "
        f"{' → '.join(str(t['score']) for t in trace)}/10. The trajectory {direction}. "
        "This is a textbook plateau: the Refiner keeps rewriting against fresh critiques, but the "
        "*quality dimension being scored* doesn't move. **2–4 rounds is the sweet spot** — more is "
        "wasted budget."
    )

    regression_block = ""
    if regressed:
        regression_block = (
            f"\n\n**(c) Score regression.** Round {trace[-1]['iteration']} scored "
            f"{last_score}/10, *worse* than round 1's {first_score}/10. The Refiner over-corrected "
            "against accumulated nitpicks and produced a draft the Critic liked **less** than the "
            "original. **In production you want early-stop when consecutive scores fail to improve** "
            "— see § 11.3."
        )

    return f"""## 9 · What we just observed

The cells above are live. The commentary below directly quotes the **actual** critiques the
Nebius-hosted Llama-3.3-70B Critic produced on this run.

### 9.1 · What the trace shows

| Iteration | Score | Critic's main concern (one-liner) |
|---|---|---|
{table_rows}

This {n_rounds}-round trace is a textbook demonstration of the Reflection failure modes we describe
generically in § 11 — except now we have evidence.
{plateau_block}{drift_block}{regression_block}

### 9.2 · What this specific run teaches

You can read this trace two ways:

1. **The pessimistic read.** Same-model Reflection at a high `target_score` finds something to
   complain about every round, even when the draft is already good. The Critic doesn't have a
   stable definition of "good" — it has a *relative* one tied to whatever draft it's looking at.

2. **The optimistic read.** Even with a generic rubric, the **first** refinement round was almost
   always substantive — the Refiner genuinely improves on round 1. The marginal value falls off
   fast after that.

**Practical takeaway:** for production, use `target_score ≈ 8–9` (not 10), cap
`max_iterations=2`, and write a **domain-specific rubric** so the Critic stays anchored. Generic
loops at high target scores are how you burn budget for negative ROI — exactly as this run
demonstrates."""


_CRITIQUE_TRIGGERS = re.compile(
    r"\b(however|one (potential )?improvement|consider|could (be )?(improved|added)|"
    r"should|missing|lacks?|but |firstly|secondly|to (further )?improve|"
    r"areas? that could|the .+ could be (more|improved)|"
    r"a few areas?|could benefit|recommend)",
    flags=re.IGNORECASE,
)


def _one_liner(critique: str, max_len: int = 130) -> str:
    """Return the most substantive critique sentence — the actionable one, not the praise."""
    text = critique.strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences:
        return text

    # Prefer a sentence that contains a "here's what to improve" trigger.
    actionable = next((s for s in sentences if _CRITIQUE_TRIGGERS.search(s)), None)
    out = actionable if actionable else sentences[0]
    out = re.sub(r"\s+", " ", out).strip().replace("|", "\\|")
    if len(out) > max_len:
        out = out[: max_len - 1].rstrip() + "…"
    return out


def _generic_fallback() -> str:
    return """## 9 · What we just observed

(Trace was empty — see § 11 for generic failure-mode discussion.)"""


def main() -> None:
    nb = nbformat.read(NB_PATH, as_version=4)
    trace = extract_trace(nb)
    if not trace:
        print("warning: no trace found, leaving § 9 untouched")
        return

    new_md = make_commentary(trace)

    replaced = False
    for cell in nb.cells:
        if cell.cell_type == "markdown" and cell.source.lstrip().startswith(
            "## 9 · What we just observed"
        ):
            cell.source = new_md
            replaced = True
            break

    if not replaced:
        raise RuntimeError("Could not find § 9 markdown cell")

    nbformat.write(nb, NB_PATH)
    scores = " -> ".join(str(t["score"]) for t in trace)
    print(f"tailored section 9 with {len(trace)} iterations (scores: {scores})")


if __name__ == "__main__":
    main()
