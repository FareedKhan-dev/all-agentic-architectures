"""Post-process notebook 20: rewrite § 9 against the CoVe captured run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "20_chain_of_verification.ipynb"
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
    r"BASELINE_LINES:\s+(\d+).*?"
    r"BASELINE_REAL_WINNERS_FOUND:\s+\[(.*?)\].*?"
    r"VERIFICATION_QUESTION_COUNT:\s+(\d+).*?"
    r"LOW_CONFIDENCE_COUNT:\s+(\d+).*?"
    r"REVISED_LINES:\s+(\d+).*?"
    r"REVISED_REAL_WINNERS_FOUND:\s+\[(.*?)\].*?"
    r"CHANGES_MADE_COUNT:\s+(\d+)",
    re.DOTALL,
)
LLAMA = re.compile(
    r"LLAMA_BASELINE_LINES:\s+(\d+).*?"
    r"LLAMA_BASELINE_REAL:\s+\[(.*?)\].*?"
    r"LLAMA_REVISED_LINES:\s+(\d+).*?"
    r"LLAMA_REVISED_REAL:\s+\[(.*?)\].*?"
    r"LLAMA_CHANGES_COUNT:\s+(\d+)",
    re.DOTALL,
)


def _parse_list(s: str) -> list[str]:
    return [x.strip().strip("'\"") for x in s.split(",") if x.strip()]


def extract_run(nb: nbformat.NotebookNode) -> dict[str, object]:
    info: dict[str, object] = {"primary": None, "llama": None, "baseline_text": "", "revised_text": "", "questions": [], "answers": []}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        text = cell_output_text(cell)
        if "GROUND_TRUTH:" in text and "BASELINE_LINES" in text:
            m = PRIMARY.search(text)
            if m:
                info["primary"] = {
                    "baseline_lines": int(m.group(1)),
                    "baseline_real": _parse_list(m.group(2)),
                    "question_count": int(m.group(3)),
                    "low_confidence": int(m.group(4)),
                    "revised_lines": int(m.group(5)),
                    "revised_real": _parse_list(m.group(6)),
                    "changes_count": int(m.group(7)),
                }
        if "=== BASELINE" in text and "=== VERIFICATION QUESTIONS" in text:
            # 8.1 inspection cell
            bm = re.search(r"=== BASELINE.*?===\n(.*?)\n===", text, re.DOTALL)
            if bm:
                info["baseline_text"] = bm.group(1).strip()
            rm = re.search(r"=== REVISED ANSWER ===\n(.*?)\Z", text, re.DOTALL)
            if rm:
                info["revised_text"] = rm.group(1).strip()
            qm = re.findall(r"\s+\d+\.\s+(.+?)\n", text.split("=== VERIFICATION ANSWERS")[0])
            info["questions"] = qm
            am = re.findall(
                r"\[\d+\]\s+Q:\s+(.+?)\n\s+A:\s+(.+?)\n\s+confidence:\s+(\w+)",
                text,
            )
            info["answers"] = [{"q": q, "a": a, "conf": c} for q, a, c in am]
        if "LLAMA_BASELINE_LINES" in text:
            m = LLAMA.search(text)
            if m:
                info["llama"] = {
                    "baseline_lines": int(m.group(1)),
                    "baseline_real": _parse_list(m.group(2)),
                    "revised_lines": int(m.group(3)),
                    "revised_real": _parse_list(m.group(4)),
                    "changes_count": int(m.group(5)),
                }
    return info


def _esc(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def make_commentary(info: dict[str, object]) -> str:
    p = info.get("primary") or {}
    l = info.get("llama") or {}
    answers = info.get("answers", [])  # type: ignore[assignment]

    # ---- 9.1 summary ----
    if p:
        baseline_hallucinations = p["baseline_lines"] - len(p["baseline_real"])  # type: ignore[index]
        revised_hallucinations = p["revised_lines"] - len(p["revised_real"])  # type: ignore[index]
        summary_table = (
            f"| BASELINE lines | {p['baseline_lines']} |\n"
            f"| BASELINE real winners (of 2 possible) | {len(p['baseline_real'])} — {p['baseline_real']} |\n"
            f"| BASELINE hallucinated lines (lines − real) | **{baseline_hallucinations}** |\n"
            f"| Verification questions generated | {p['question_count']} |\n"
            f"| Low-confidence verification answers | {p['low_confidence']} |\n"
            f"| REVISED lines | {p['revised_lines']} |\n"
            f"| REVISED real winners | {len(p['revised_real'])} — {p['revised_real']} |\n"
            f"| REVISED hallucinated lines | **{revised_hallucinations}** |\n"
            f"| Changes made (REVISE bullet count) | {p['changes_count']} |"
        )
    else:
        summary_table = "| — | _(no run captured)_ |"

    # ---- 9.2 comparison ----
    if p and l:
        comp_table = (
            f"| Reasoning (Qwen3-Thinking) | {p['baseline_lines']} → {p['revised_lines']} | "
            f"{len(p['baseline_real'])}/{p['baseline_lines']} → {len(p['revised_real'])}/{p['revised_lines']} | "
            f"{p['changes_count']} |\n"
            f"| Plain (Llama-3.3-70B) | {l['baseline_lines']} → {l['revised_lines']} | "
            f"{len(l['baseline_real'])}/{l['baseline_lines']} → {len(l['revised_real'])}/{l['revised_lines']} | "
            f"{l['changes_count']} |"
        )
    else:
        comp_table = "| — | — | — | — |"

    # ---- 9.3 auto-flags ----
    obs: list[str] = []
    if p:
        bh = p["baseline_lines"] - len(p["baseline_real"])  # type: ignore[index]
        rh = p["revised_lines"] - len(p["revised_real"])  # type: ignore[index]
        if bh > 0 and rh < bh:
            obs.append(
                f"**✅ CoVe reduced hallucinations**: baseline had **{bh}** hallucinated line(s); "
                f"revised has **{rh}**. The pipeline caught {bh - rh} invention(s)."
            )
        elif bh > 0 and rh >= bh:
            obs.append(
                f"**❌ CoVe did NOT reduce hallucinations**: baseline had **{bh}** hallucinated line(s); "
                f"revised has **{rh}**. Likely the verification answers confirmed the inventions "
                "(same-model hallucination loop). Use a stronger model for EXECUTE, or pair with RAG."
            )
        elif bh == 0:
            obs.append(
                f"**🟰 Baseline was already correct** — no hallucinations to catch. Either the model "
                "knew the answer (Qwen-Thinking has strong factual recall) or it hedged. The "
                "interesting comparison is § 9.2 below: did Llama hallucinate where Qwen didn't?"
            )

        if p["changes_count"] == 0 and bh > 0:  # type: ignore[index]
            obs.append(
                "**⚠️  REVISE made 0 changes despite baseline hallucinations** — the verification "
                "answers must have agreed with the (wrong) baseline claims. This is the same-model "
                "consistency-bias trap CoVe is supposed to break. Mitigation: use different LLMs "
                "for BASELINE and EXECUTE."
            )
        if p["low_confidence"] > 0:  # type: ignore[index]
            obs.append(
                f"**ℹ️  {p['low_confidence']} verification answer(s) were low-confidence** — the model "
                "explicitly flagged uncertainty rather than guessing. That's healthy calibration; "
                "REVISE should treat those as untrusted."
            )

    if p and l:
        lp_revised_real = len(l.get("revised_real", []))
        lp_revised_lines = l.get("revised_lines", 0)
        p_revised_real = len(p.get("revised_real", []))
        p_revised_lines = p.get("revised_lines", 0)
        l_hallu = lp_revised_lines - lp_revised_real
        p_hallu = p_revised_lines - p_revised_real
        if l_hallu > p_hallu:
            obs.append(
                f"**⚠️  Llama's revised answer has {l_hallu} hallucinations vs Qwen-Thinking's {p_hallu}.** "
                "Same architecture, weaker model — CoVe helps but the lift depends on the underlying "
                "model's verification accuracy."
            )
        elif l_hallu == p_hallu:
            obs.append(
                f"**🟰 Both models ended with {p_hallu} hallucinations in the revised answer.** "
                "On this task CoVe + either LLM did equivalent work."
            )

    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns surfaced."

    # ---- 9.4 verbatim ----
    baseline_text = info.get("baseline_text", "") or "_(not captured)_"
    revised_text = info.get("revised_text", "") or "_(not captured)_"

    # ---- 9.5 verification answers table ----
    if answers:
        a_rows = "\n".join(
            f"| {i+1} | {_esc(a['q'])[:80]}{'…' if len(a['q'])>80 else ''} | {_esc(a['a'])[:120]}{'…' if len(a['a'])>120 else ''} | {a['conf']} |"
            for i, a in enumerate(answers)
        )
    else:
        a_rows = "| — | — | — | — |"

    return f"""## 9 · What we just observed

The cells above ran CoVe on a hallucination trap: ask for **5** Le Guin Hugo Best Novel wins when only **2** exist. We measure whether the BASELINE invented fillers and whether REVISE dropped them.

### 9.1 · Hallucination-reduction summary (reasoning model)

| Metric | Value |
|---|---|
{summary_table}

### 9.2 · Reasoning vs non-reasoning LLM

| Model | Lines (baseline → revised) | Real winners present (baseline → revised) | Changes made |
|---|---|---|---|
{comp_table}

### 9.3 · Patterns surfaced in this run

{obs_block}

### 9.4 · Verbatim BEFORE → AFTER (reasoning model)

**Baseline (before verification):**

```
{baseline_text}
```

**Revised (after CoVe):**

```
{revised_text}
```

### 9.5 · The verification Q&A (executed independently of the baseline)

| # | Verification question | Verification answer | Confidence |
|---|---|---|---|
{a_rows}

### 9.6 · The takeaway

CoVe's pedagogical value lives in two cells of § 9.1: **`BASELINE hallucinated lines`** and **`REVISED hallucinated lines`**. If the second is smaller than the first, the architecture worked. If they're equal (or both zero), either the task was too easy or the same-model consistency-bias trap (§ 9.3) defeated the verification — mitigation: use a different / stronger model in the EXECUTE stage, or compose with RAG (§ 11.3 extension #1)."""


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
    l = info["llama"] or {}
    print(
        f"tailored section 9: primary={p.get('baseline_lines')}→{p.get('revised_lines')} lines, "
        f"changes={p.get('changes_count')}; llama={l.get('baseline_lines')}→{l.get('revised_lines')}, "
        f"changes={l.get('changes_count')}"
    )


if __name__ == "__main__":
    main()
