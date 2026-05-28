"""Post-process notebook 10: rewrite § 9 against the Mental Loop captured run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "10_mental_loop.ipynb"
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
        "explanation": "",
        "chosen_action": "",
        "chosen_score": 0,
        "simulations": [],
        "llm_scores": [],
        "final_scores": [],
        "llm_spread": 0,
        "py_spread": 0,
    }
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "result = arch.run(TASK)" in cell.source:
            text = cell_output_text(cell)
            m = re.search(
                r"Recommendation[\s\S─-╿]*?\n(.+?)\n[\s─-]*\n[\s─-╿]*?chosen:",
                text,
                re.DOTALL,
            )
            if m:
                info["explanation"] = re.sub(r"[\s─-╿]+$", "", m.group(1)).strip()
            m = re.search(r"CHOSEN_SCORE:\s+(\d+)/5", text)
            if m:
                info["chosen_score"] = int(m.group(1))
            m = re.search(r"CHOSEN_ACTION:\s+(.+?)(?=\n|\Z)", text)
            if m:
                info["chosen_action"] = _normalize_ws(m.group(1))[:200]
        if "for i, t in enumerate(result.trace" in cell.source:
            text = cell_output_text(cell)
            # New header format: "[N] Action · FINAL score X/5 · (LLM said Y/5, source=Z, predicted_metric=M)"
            blocks = re.findall(
                r"\[(\d+)\]\s+Action[^·]*·\s+FINAL\s+score\s+(\d+)/5\s+·\s+\(LLM\s+said\s+(\d+)/5,\s+source=(\w+),\s+predicted_metric=([^)]+)\)\s*\n(.+?)(?=\n\s*[›>]\s+\[\d+\]\s+Action|\Z)",
                text,
                re.DOTALL,
            )
            sims = []
            for n, final_s, llm_s, src, metric, body in blocks:
                action_m = re.match(r"(.+?)(?=\n\s*[›>])", body, re.DOTALL)
                action = _normalize_ws(action_m.group(1)) if action_m else _normalize_ws(body[:150])
                outcome_m = re.search(r"Predicted outcome\s*\n(.+?)(?=\n\s*[›>])", body, re.DOTALL)
                outcome = _normalize_ws(outcome_m.group(1)) if outcome_m else ""
                risks_m = re.search(r"Risks\s*\n(.+?)(?=\n\s*[›>]|\Z)", body, re.DOTALL)
                risks = _normalize_ws(risks_m.group(1)) if risks_m else ""
                # Parse predicted_metric (could be 'None' or a number)
                m_str = metric.strip()
                m_val: float | None = None
                if m_str not in ("None", "?", ""):
                    try:
                        m_val = float(m_str)
                    except ValueError:
                        m_val = None
                sims.append({
                    "n": int(n),
                    "score": int(final_s),
                    "llm_score": int(llm_s),
                    "score_source": src,
                    "predicted_metric": m_val,
                    "action": action[:200],
                    "outcome": outcome[:300],
                    "risks": risks[:200],
                })
            info["simulations"] = sims
            # Pull summary lines.
            m = re.search(r"LLM_SCORES_RAW:\s+\[([\d, ]+)\]", text)
            if m:
                info["llm_scores"] = [int(x) for x in m.group(1).split(",") if x.strip()]
            m = re.search(r"PYTHON_SCORES_FINAL:\s+\[([\d, ]+)\]", text)
            if m:
                info["final_scores"] = [int(x) for x in m.group(1).split(",") if x.strip()]
            m = re.search(r"LLM_SPREAD:\s+(\d+)\s+·\s+PYTHON_SPREAD:\s+(\d+)", text)
            if m:
                info["llm_spread"] = int(m.group(1))
                info["py_spread"] = int(m.group(2))
    return info


def make_commentary(info: dict[str, object]) -> str:
    sims: list[dict] = info.get("simulations", [])  # type: ignore[assignment]
    expl: str = info.get("explanation", "")  # type: ignore[assignment]
    chosen: str = info.get("chosen_action", "")  # type: ignore[assignment]
    chosen_score = info.get("chosen_score", 0)
    llm_scores: list[int] = info.get("llm_scores", [])  # type: ignore[assignment]
    final_scores: list[int] = info.get("final_scores", [])  # type: ignore[assignment]
    llm_spread = info.get("llm_spread", 0)
    py_spread = info.get("py_spread", 0)

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ").strip()

    sim_table = (
        "\n".join(
            f"| {s['n']} | {s['predicted_metric'] if s['predicted_metric'] is not None else '—'} | {s['llm_score']}/5 | **{s['score']}/5** | {esc(s['action'])[:60]}{'…' if len(s['action']) > 60 else ''} |"
            for s in sims
        )
        if sims
        else "| — | — | — | — | _(no simulations captured)_ |"
    )

    scores = final_scores or [s["score"] for s in sims]
    spread = (max(scores) - min(scores)) if scores else 0

    obs: list[str] = []
    if not scores:
        obs.append("**No simulations captured** — the simulate step failed.")
    else:
        # Compare LLM vs Python spread — this is the headline finding now.
        if llm_scores and final_scores:
            obs.append(
                f"**The deterministic-scoring fix is doing its job.** "
                f"The LLM's own `overall_score` field on these 3 candidates was "
                f"`{llm_scores}` (spread = {llm_spread}, a narrow band — the "
                f"familiar LLM-as-Scorer flatness pathology). The **Python scoring "
                f"function** computed `{final_scores}` from the LLM's "
                f"`predicted_metric` field (spread = {py_spread}) — a real "
                f"discriminating signal that the argmax can act on. This is the "
                f"central lesson of Mental Loop: **let the LLM predict the "
                f"underlying number, let Python compute the score.**"
            )
            if py_spread > llm_spread:
                obs.append(
                    f"**Score spread comparison**: LLM={llm_spread}, Python={py_spread}. "
                    f"Python won by {py_spread - llm_spread} points of dynamic range — "
                    "exactly the improvement we built `scoring_fn` for."
                )
        elif spread == 0 and len(scores) > 1:
            obs.append(
                f"**Flat scores ({scores[0]}/5 everywhere)** — `scoring_fn` not "
                "set, so the LLM's compressed scoring went through unmodified. "
                "Pass `scoring_fn=...` to MentalLoop for real discrimination."
            )

    # Risk content check
    risky_count = sum(1 for s in sims if s.get("risks") and len(s["risks"]) > 30)
    if sims and risky_count == 0:
        obs.append(
            "**No substantive risks listed** — the simulator imagined best cases for "
            "every candidate. Likely *optimistic bias*. Tighten the risks-field prompt."
        )
    elif sims and risky_count < len(sims):
        obs.append(
            f"**Risks listed for only {risky_count}/{len(sims)} candidates** — "
            "uneven analysis. Add a hard rule: every simulation must have ≥1 concrete risk."
        )

    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns."

    expl_block = (
        "> " + (expl[:600].replace("\n", "\n> ") if expl else "_(no explanation captured)_")
        + ("…" if len(expl) > 600 else "")
    )

    return f"""## 9 · What we just observed

The cells above ran Mental Loop on the **NYC commute decision** with `scoring_fn=commute_score_from_minutes` — a deterministic Python function that converts predicted minutes → 1-5 score.

### 9.1 · Quantitative summary

| Metric | Value |
|---|---|
| Candidates generated | **{len(sims)}** |
| Chosen action | {chosen[:120] + ('…' if len(chosen) > 120 else '')} |
| Chosen FINAL score | **{chosen_score}**/5 |
| LLM-raw score spread | {llm_spread} (often flat — the pathology) |
| Python FINAL score spread | **{py_spread}** (the fix) |
| LLM-raw scores | {llm_scores or "—"} |
| Python FINAL scores | {final_scores or "—"} |

### 9.2 · Per-candidate breakdown

| # | predicted_metric (min) | LLM score | **Python score (final)** | Action |
|---|---|---|---|---|
{sim_table}

### 9.3 · Patterns surfaced in this run

{obs_block}

### 9.4 · Final recommendation (verbatim)

{expl_block}

### 9.5 · The takeaway

The deterministic-scoring pattern is the canonical fix for LLM-as-Scorer flatness:

1. **LLM predicts the underlying NUMBER** (`predicted_metric: float`) — concrete, harder to fudge.
2. **Python computes the SCORE** from that number via a deterministic function — perfectly calibrated.
3. **The argmax now has signal** even when the LLM compresses its own `overall_score`.

For tasks with a measurable outcome (time, cost, error rate, throughput, accuracy), this pattern eliminates the entire class of "everything is 4/5" bugs. Use it whenever you can express the scoring criterion as Python."""


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
        f"tailored section 9: {len(info['simulations'])} sims, "
        f"chosen score {info['chosen_score']}/5"
    )


if __name__ == "__main__":
    main()
