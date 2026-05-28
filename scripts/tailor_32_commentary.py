"""Post-process notebook 32: rewrite § 9 against Constitutional AI run."""
from __future__ import annotations
import re
from pathlib import Path
import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "32_constitutional_ai.ipynb"
ANSI = re.compile(r"\x1b\[[0-9;]*[mGKH]")


def cell_output_text(cell):
    chunks = []
    for o in cell.outputs:
        t = o.get("text", "") or o.get("data", {}).get("text/plain", "")
        if isinstance(t, list): t = "".join(t)
        chunks.append(ANSI.sub("", str(t)))
    return "\n".join(chunks)


def extract(nb):
    info = {"iterations": 0, "all_passed": False, "n_pass": 0, "n_rules": 0,
            "n_fail": 0, "verdicts": [], "failures": [], "final_chars": 0, "final_preview": ""}
    for cell in nb.cells:
        if cell.cell_type != "code": continue
        text = cell_output_text(cell)
        if "ITERATIONS:" not in text or "ALL_PASSED:" not in text: continue
        m = re.search(r"ITERATIONS:\s+(\d+)", text); info["iterations"] = int(m.group(1)) if m else 0
        m = re.search(r"ALL_PASSED:\s+(True|False)", text); info["all_passed"] = (m.group(1) == "True") if m else False
        m = re.search(r"N_PASS:\s+(\d+)/(\d+)", text)
        if m: info["n_pass"], info["n_rules"] = int(m.group(1)), int(m.group(2))
        m = re.search(r"N_FAIL:\s+(\d+)", text); info["n_fail"] = int(m.group(1)) if m else 0
        for vm in re.finditer(r"\[(\d+)\]\s+[✓✗]\s+(pass|fail):\s+(.+)", text):
            info["verdicts"].append({"idx": int(vm.group(1)), "verdict": vm.group(2), "rationale": vm.group(3).strip()})
        for fm in re.finditer(r"^\s+-\s+(.+)$", text, re.MULTILINE):
            info["failures"].append(fm.group(1).strip())
        m = re.search(r"=== FINAL ANSWER \((\d+) chars\) ===\n(.+?)\Z", text, re.DOTALL)
        if m:
            info["final_chars"] = int(m.group(1))
            info["final_preview"] = m.group(2).strip()[:300]
    return info


def _esc(s): return s.replace("|", "\\|").strip()


def make_commentary(info):
    verdicts = info["verdicts"]
    if verdicts:
        rows = "\n".join(
            f"| [{v['idx']}] | {'✅ pass' if v['verdict'] == 'pass' else '❌ fail'} | {_esc(v['rationale'])[:120]}{'…' if len(v['rationale']) > 120 else ''} |"
            for v in verdicts
        )
    else:
        rows = "| — | — | _(no verdicts captured)_ |"
    obs = []
    if info["all_passed"]:
        obs.append(f"**✅ All {info['n_rules']} rules passed** after {info['iterations']} iteration(s). The revision loop terminated cleanly.")
    else:
        obs.append(f"**⚠️ Not all rules passed** ({info['n_pass']}/{info['n_rules']}) — exhausted max_iterations. Surface failures for review.")
    if info["iterations"] > 0:
        obs.append(f"**Revision iterations: {info['iterations']}** — baseline needed correction.")
    else:
        obs.append("**Baseline passed all rules** — no revisions needed.")
    obs_block = "\n\n".join(f"- {o}" for o in obs)

    return f"""## 9 · What we just observed

The cells above ran Constitutional AI on a deliberately-provocative prompt (asks for opinionated, uncited, verbose content — likely to violate rules 0, 1, 2).

### 9.1 · Per-rule verdicts (after final iteration)

| Rule | Verdict | Rationale |
|---|---|---|
{rows}

### 9.2 · Summary

- **Iterations**: {info['iterations']}
- **All passed**: {info['all_passed']}
- **Pass count**: {info['n_pass']}/{info['n_rules']}
- **Final answer length**: {info['final_chars']} chars

### 9.3 · Patterns surfaced

{obs_block}

### 9.4 · The takeaway

CAI's value is **all in the constitution**. A precise rule list ("under 200 words", "no opinions") gives the critique LLM concrete things to commit to per-rule; a vague constitution ("be helpful, harmless, honest") gives the LLM room to flat-pass everything. The deterministic-picker is `all(v == 'pass')` — Python only — so the revise loop is driven by the per-rule LLM commitments, never a numeric overall score."""


def main():
    nb = nbformat.read(NB_PATH, as_version=4)
    info = extract(nb)
    new_md = make_commentary(info)
    replaced = False
    for c in nb.cells:
        if c.cell_type == "markdown" and c.source.lstrip().startswith("## 9 · What we just observed"):
            c.source = new_md; replaced = True; break
    if not replaced: raise RuntimeError("§9 not found")
    nbformat.write(nb, NB_PATH)
    print(f"tailored: iter={info['iterations']} all_passed={info['all_passed']} pass={info['n_pass']}/{info['n_rules']}")

if __name__ == "__main__": main()
