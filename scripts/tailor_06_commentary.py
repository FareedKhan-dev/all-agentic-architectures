"""Post-process notebook 06: rewrite § 9 against the PEV captured run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "06_pev.ipynb"
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
        "answer": "",
        "steps_total": 0,
        "steps_passed": 0,
        "steps_fail_accepted": 0,
        "total_attempts": 0,
        "step_records": [],
    }
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "result = arch.run(TASK)" in cell.source:
            text = cell_output_text(cell)
            m = re.search(
                r"Final answer[\s\S─-╿]*?\n(.+?)\n[\s─-]*\n[\s─-╿]*?steps:",
                text,
                re.DOTALL,
            )
            if m:
                info["answer"] = re.sub(r"[\s─-╿]+$", "", m.group(1)).strip()
            m = re.search(
                r"steps:\s+(\d+).*?pass:\s+(\d+).*?fail-accepted:\s+(\d+).*?total attempts:\s+(\d+)",
                text,
                re.DOTALL,
            )
            if m:
                info["steps_total"] = int(m.group(1))
                info["steps_passed"] = int(m.group(2))
                info["steps_fail_accepted"] = int(m.group(3))
                info["total_attempts"] = int(m.group(4))
        if "for i, t in enumerate(result.trace" in cell.source:
            text = cell_output_text(cell)
            # Each step record block: "[N] ✓/✗ verdict (attempts=K, confidence=C/5)"
            records = re.findall(
                r"\[\d+\]\s+[✓✗]?\s*(\w[\w-]*)\s+\(attempts=(\d+),\s*confidence=([^\s)]+)/?5?\)\s*\n\s*step:\s+(.+?)(?=\n\s*[›>]\s+|\Z)",
                text,
                re.DOTALL,
            )
            info["step_records"] = [
                {
                    "verdict": v,
                    "attempts": int(a),
                    "confidence": c,
                    "step": _normalize_ws(s),
                }
                for v, a, c, s in records
            ]
    return info


def make_commentary(info: dict[str, object]) -> str:
    n_total = info.get("steps_total", 0)
    n_pass = info.get("steps_passed", 0)
    n_fail = info.get("steps_fail_accepted", 0)
    total_att = info.get("total_attempts", 0)
    records: list[dict] = info.get("step_records", [])  # type: ignore[assignment]
    answer: str = info.get("answer", "")  # type: ignore[assignment]

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ").strip()

    rec_table = (
        "\n".join(
            f"| {i+1} | {r['verdict']} | {r['attempts']} | {r['confidence']}/5 | {esc(r['step'])[:100]} |"
            for i, r in enumerate(records)
        )
        if records
        else "| — | — | — | — | _(no step records captured)_ |"
    )

    retry_count = total_att - n_total

    obs: list[str] = []
    if n_total == 0:
        obs.append("**No steps executed.** Likely planning failure — check provider compatibility.")
    elif n_fail == 0 and n_pass == n_total:
        if retry_count == 0:
            obs.append(
                "**Clean run — every step passed first try.** Either the task was "
                "straightforward and the Executor nailed it, or the Verifier was too "
                "lenient. Cross-check the answer's grounding against the per-step results."
            )
        else:
            obs.append(
                f"**All steps eventually passed**, but with {retry_count} retry-round(s). "
                "The Verifier was doing real work — it caught issues on the first attempt "
                "of some step(s) and the retry recovered."
            )
    elif n_fail > 0 and n_pass > 0:
        obs.append(
            f"**Partial success: {n_pass}/{n_total} steps passed, {n_fail} fail-accepted.** "
            "This is the honest PEV signal — the Verifier rejected some step(s) until "
            "retries ran out. Inspect the `last_critique` field on fail-accepted steps "
            "to see what the Verifier kept flagging."
        )
    elif n_fail == n_total:
        obs.append(
            f"**Total failure: every step was fail-accepted.** The Verifier never accepted "
            "anything. Either the rubric is too strict or the Executor is genuinely "
            "incapable of meeting the bar. Switch the Verifier model or relax the rubric."
        )

    # Retry effectiveness check
    retries_used = [r for r in records if r["attempts"] > 1]
    retries_recovered = [r for r in retries_used if r["verdict"] == "pass"]
    if retries_used and len(retries_recovered) < len(retries_used):
        obs.append(
            f"**Retries were partially effective**: {len(retries_recovered)} of "
            f"{len(retries_used)} retried step(s) recovered to `pass`; the rest stayed "
            "failed. When retry doesn't help, the step is likely genuinely impossible — "
            "force-accept and synthesize honestly."
        )

    if answer:
        has_partial_signal = any(
            phrase in answer.lower()
            for phrase in ("conflicting", "approximate", "partial", "not available", "could not")
        )
        if n_fail > 0 and not has_partial_signal:
            obs.append(
                "**Synthesis is silent about the failure(s).** Even though "
                f"{n_fail} step(s) were `fail-accepted`, the final answer reads "
                "confidently with no hedging. The `_finalize` prompt asks for "
                "honesty about failures but the LLM glossed over it. Add a hard "
                "rule: *'If any step has verdict fail-accepted, prefix the answer "
                "with PARTIAL: and list the missing data point.'*"
            )

    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable pathologies — Verifier and Executor cooperated cleanly."

    answer_block = (
        "> " + (answer[:600].replace("\n", "\n> ") if answer else "_(no answer captured)_")
        + ("…" if len(answer) > 600 else "")
    )

    return f"""## 9 · What we just observed

The cells above are live. Below: a quantitative + qualitative breakdown of the **actual** Plan-Execute-Verify trace the Nebius-hosted Llama-3.3-70B produced on this run.

### 9.1 · Quantitative summary

| Metric | Value |
|---|---|
| Steps executed | **{n_total}** |
| Steps passed | **{n_pass}** / {n_total} |
| Steps `fail-accepted` | **{n_fail}** |
| Total attempts (incl. retries) | **{total_att}** |
| Retry rounds | {retry_count} |
| Pass rate | {(n_pass / n_total * 100) if n_total else 0:.0f}% |

### 9.2 · Per-step verdicts

| # | Verdict | Attempts | Confidence | Step |
|---|---|---|---|---|
{rec_table}

### 9.3 · Patterns surfaced in this run

{obs_block}

### 9.4 · The final answer (verbatim)

{answer_block}

### 9.5 · The takeaway

The pass-rate metric is what makes PEV worth its extra cost over plain Planning: you have an **honest quality signal per task**. A run with 100% pass-rate and 0 retries either means the task was easy or the Verifier was lazy — check the per-step confidence. A run with `fail-accepted` steps is *useful information*: the agent reached the end of its plan but knows the answer is incomplete, and the final synthesis (if the prompt is doing its job) hedges accordingly."""


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
        f"tailored section 9: total={info['steps_total']}, "
        f"pass={info['steps_passed']}, fail={info['steps_fail_accepted']}, "
        f"captured_records={len(info['step_records'])}"
    )


if __name__ == "__main__":
    main()
