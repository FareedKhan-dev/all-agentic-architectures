"""Post-process notebook 21: rewrite § 9 against the Self-Consistency captured run."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "21_self_consistency.ipynb"
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
    r"FINAL_ANSWER:\s+(.+?)\n"
    r"EXPECTED:\s+(.+?)\n"
    r"MATCH:\s+(True|False)\s*\n"
    r"\s*\n"
    r"N_SAMPLES:\s+(\d+)\s*\n"
    r"UNIQUE_ANSWERS:\s+(\d+)\s*\n"
    r"WINNER_COUNT:\s+(\d+)/(\d+)\s*\n"
    r"AGREEMENT_FRACTION:\s+([0-9.]+)\s*\n"
    r"TALLY:\s+(\{.*?\})"
)
SINGLE = re.compile(
    r"SINGLE_SAMPLE_TALLY:\s+(\{.*?\})\s*\n"
    r"SINGLE_SAMPLE_CORRECT:\s+(\d+)/(\d+)\s+trials landed on\s+(.+?)\n"
    r"SINGLE_SAMPLE_ERROR_RATE:\s+([0-9.]+)"
)


def extract_run(nb: nbformat.NotebookNode) -> dict[str, object]:
    info: dict[str, object] = {"primary": None, "single": None, "samples": []}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        text = cell_output_text(cell)
        if "FINAL_ANSWER:" in text and "TALLY:" in text and "SINGLE_SAMPLE" not in cell.source:
            m = PRIMARY.search(text)
            if m:
                try:
                    tally = ast.literal_eval(m.group(9))
                except Exception:
                    tally = {}
                info["primary"] = {
                    "final_answer": m.group(1).strip().strip("'\""),
                    "expected": m.group(2).strip().strip("'\""),
                    "match": m.group(3) == "True",
                    "n_samples": int(m.group(4)),
                    "unique_answers": int(m.group(5)),
                    "winner_count": int(m.group(6)),
                    "agreement_fraction": float(m.group(8)),
                    "tally": tally,
                }
        if "SINGLE_SAMPLE_TALLY" in text:
            m = SINGLE.search(text)
            if m:
                try:
                    s_tally = ast.literal_eval(m.group(1))
                except Exception:
                    s_tally = {}
                info["single"] = {
                    "tally": s_tally,
                    "correct": int(m.group(2)),
                    "n_trials": int(m.group(3)),
                    "expected": m.group(4).strip().strip("'\""),
                    "error_rate": float(m.group(5)),
                }
        if "sample" in cell.source and "answer=" in text and "---" in text:
            # 8.1 sample dump — extract per-sample answers
            samples = re.findall(r"sample\s+(\d+)\s+--\s+answer=([^\s]+)", text)
            if samples:
                info["samples"] = [{"i": int(i), "answer": a.strip().strip("'\"")} for i, a in samples]
    return info


def make_commentary(info: dict[str, object]) -> str:
    p = info.get("primary") or {}
    s = info.get("single") or {}
    samples = info.get("samples") or []  # type: ignore[assignment]

    # ---- 9.1 vote summary ----
    if p:
        tally_rows = "\n".join(
            f"| `{ans}` | {ct} | {ct / p['n_samples']:.0%} |"  # type: ignore[index]
            for ans, ct in sorted(p["tally"].items(), key=lambda kv: -kv[1])  # type: ignore[index]
        )
        winner_status = (
            f"**✅ Correct** (matches expected `{p['expected']}`)"
            if p["match"] else
            f"**❌ Wrong** (expected `{p['expected']}`, got `{p['final_answer']}`)"
        )
        summary = (
            f"- **Winner**: `{p['final_answer']}` — {winner_status}\n"
            f"- **Agreement**: {p['winner_count']}/{p['n_samples']} samples = {p['agreement_fraction']:.0%}\n"
            f"- **Unique answers across samples**: {p['unique_answers']}"
        )
    else:
        tally_rows = "| — | — | — |"
        summary = "_(no run captured)_"

    # ---- 9.2 sample-by-sample table ----
    if samples:
        sample_rows = "\n".join(
            f"| {smp['i']} | `{smp['answer']}` |"
            for smp in samples
        )
    else:
        sample_rows = "| — | _(no samples captured)_ |"

    # ---- 9.3 vs single-sample contrast ----
    if p and s:
        n_trials = s["n_trials"]  # type: ignore[index]
        correct = s["correct"]  # type: ignore[index]
        sc_tally = ", ".join(f"`{k}` ×{v}" for k, v in sorted(s["tally"].items(), key=lambda kv: -kv[1]))  # type: ignore[index]
        single_block = (
            f"| Strategy | Correct trials | Error rate |\n"
            f"|---|---|---|\n"
            f"| **Self-Consistency (modal of N={p['n_samples']})** | {1 if p['match'] else 0}/1 | {0 if p['match'] else 100:.0%} |\n"
            f"| **Single-sample baseline** | {correct}/{n_trials} | {s['error_rate']:.0%} |\n\n"
            f"Single-sample tally over {n_trials} independent runs: {sc_tally}."
        )
    else:
        single_block = "_(single-sample contrast not captured)_"

    # ---- 9.4 auto-flags ----
    obs: list[str] = []
    if p:
        if p["agreement_fraction"] >= 0.9 and p["match"]:  # type: ignore[index]
            obs.append(
                f"**✅ Strong agreement on the right answer** ({p['winner_count']}/{p['n_samples']}). "  # type: ignore[index]
                "If you'd run a single-sample CoT you'd almost certainly have landed on the same answer. "
                "Self-Consistency added little lift here — it would matter more on harder tasks."
            )
        elif 0.5 <= p["agreement_fraction"] < 0.9 and p["match"]:  # type: ignore[index]
            wrong = p["n_samples"] - p["winner_count"]  # type: ignore[index]
            obs.append(
                f"**✅ Modal vote rescued the right answer.** {wrong} of {p['n_samples']} samples "  # type: ignore[index]
                "landed on a wrong answer — exactly the case Self-Consistency exists to handle. "
                "A single-sample CoT had a non-trivial chance of being wrong."
            )
        elif p["agreement_fraction"] < 0.5:  # type: ignore[index]
            obs.append(
                f"**⚠️  Modal answer wins with <50% support** ({p['winner_count']}/{p['n_samples']}). "  # type: ignore[index]
                "The task is too hard or temperature too high. Treat the answer as low-confidence; "
                "raise N, lower temperature, or pair with verification."
            )
        if not p["match"]:  # type: ignore[index]
            obs.append(
                f"**❌ Modal vote landed on the wrong answer** (`{p['final_answer']}` vs expected `{p['expected']}`). "  # type: ignore[index]
                "Self-Consistency can't fix systematic model bias. Pair with CoVe (nb 20) or RAG."
            )
        if p["unique_answers"] == 1:  # type: ignore[index]
            obs.append(
                "**🟰 All samples agreed**. Either the task is easy or temperature was too low. "
                "If correct, you saved 0 by using Self-Consistency. If wrong, you wasted N× cost on identical wrong answers."
            )

    if p and s:
        sc_correct = 1 if p["match"] else 0  # type: ignore[index]
        ss_rate = s["correct"] / s["n_trials"]  # type: ignore[index]
        if sc_correct == 1 and ss_rate < 1:
            obs.append(
                f"**✅ Self-Consistency outperformed single-sample baseline** "
                f"on this run: single-sample was right {s['correct']}/{s['n_trials']} = "  # type: ignore[index]
                f"{ss_rate:.0%} of the time; modal vote got it right."
            )
        elif sc_correct == 0 and ss_rate > 0:
            obs.append(
                f"**🤔 Self-Consistency was wrong but some single-sample trials got it right.** "
                "Either the modal vote got unlucky on this draw, or the wrong answer happens to be "
                "more *common* than the right one (which Self-Consistency can't fix). "
                "Re-run a few times to distinguish noise from bias."
            )

    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns surfaced."

    return f"""## 9 · What we just observed

The cells above ran Self-Consistency on a perspective-taking trick (the Sally-siblings problem) where some chain-of-thought paths are expected to slip but the modal answer should be correct.

### 9.1 · Vote tally + winner

{summary}

| Answer | Count | Share |
|---|---|---|
{tally_rows}

### 9.2 · Per-sample answers

| Sample | Answer |
|---|---|
{sample_rows}

### 9.3 · Self-Consistency vs single-sample CoT

{single_block}

### 9.4 · Patterns surfaced in this run

{obs_block}

### 9.5 · The takeaway

Self-Consistency is the simplest deterministic-picker architecture in the catalogue: every sample is the same model, every vote is one ballot, the picker is `Counter.most_common(1)`. Its lift comes from one place — when *some* paths are wrong but the *modal* path is right. Read the § 9.1 tally:

- **All votes identical** → architecture spent N× cost for nothing.
- **Modal answer wins with a clear majority** → the lift you paid for.
- **Modal answer wins with a thin plurality (<50%)** → treat as low confidence; the architecture is *honest* about uncertainty but the answer might still be wrong.

The single-sample comparison in § 9.3 makes the lift concrete by re-running the same task one-shot N times — the gap between single-shot accuracy and Self-Consistency accuracy is the architecture's value on this task."""


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
    s = info["single"] or {}
    print(
        f"tailored: primary tally={p.get('tally')} winner={p.get('final_answer')!r} match={p.get('match')}; "
        f"single correct={s.get('correct')}/{s.get('n_trials')}"
    )


if __name__ == "__main__":
    main()
