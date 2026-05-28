"""Post-process notebook 30: rewrite § 9 against STORM run."""

from __future__ import annotations
import re
from pathlib import Path
import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "30_storm.ipynb"
ANSI = re.compile(r"\x1b\[[0-9;]*[mGKH]")


def cell_output_text(cell):
    chunks = []
    for o in cell.outputs:
        t = o.get("text", "") or o.get("data", {}).get("text/plain", "")
        if isinstance(t, list): t = "".join(t)
        chunks.append(ANSI.sub("", str(t)))
    return "\n".join(chunks)


def extract(nb):
    info = {"n_persp": 0, "n_q": 0, "n_sec": 0, "chars": 0,
            "perspectives": [], "questions": [], "outline": [], "article": ""}
    for cell in nb.cells:
        if cell.cell_type != "code": continue
        text = cell_output_text(cell)
        if "N_PERSPECTIVES" not in text or "ARTICLE" not in text: continue
        m = re.search(r"N_PERSPECTIVES:\s+(\d+)", text);  info["n_persp"] = int(m.group(1)) if m else 0
        m = re.search(r"N_QUESTIONS:\s+(\d+)", text);     info["n_q"] = int(m.group(1)) if m else 0
        m = re.search(r"N_SECTIONS:\s+(\d+)", text);      info["n_sec"] = int(m.group(1)) if m else 0
        m = re.search(r"ARTICLE_CHARS:\s+(\d+)", text);   info["chars"] = int(m.group(1)) if m else 0
        # Perspectives
        for pm in re.finditer(r"=== PERSPECTIVES ===\n((?:\s+\[\d+\][^\n]+\n)+)", text):
            info["perspectives"] = [re.sub(r"^\s+\[\d+\]\s+", "", l).strip() for l in pm.group(1).splitlines() if l.strip()]
        # Questions
        for qm in re.finditer(r"=== QUESTIONS ===\n((?:\s+\[\d+\][^\n]+\n)+)", text):
            info["questions"] = [re.sub(r"^\s+\[\d+\]\s+", "", l).strip() for l in qm.group(1).splitlines() if l.strip()]
        # Outline
        om = re.search(r"=== OUTLINE ===\n(.*?)=== ARTICLE", text, re.DOTALL)
        if om:
            sections = []
            for sm in re.finditer(r"\s+##\s+([^\n]+)\n((?:\s+-\s+[^\n]+\n)*)", om.group(1)):
                title = sm.group(1).strip()
                points = [re.sub(r"^\s+-\s+", "", l).strip() for l in sm.group(2).splitlines() if l.strip()]
                sections.append({"title": title, "points": points})
            info["outline"] = sections
        am = re.search(r"=== ARTICLE \(first \d+ chars\) ===\n(.+?)\Z", text, re.DOTALL)
        if am:
            info["article"] = am.group(1).strip()
    return info


def _esc(s): return s.replace("|", "\\|").strip()


def make_commentary(info):
    persp_rows = "\n".join(f"| {i+1} | {_esc(p)[:140]}{'…' if len(p) > 140 else ''} |" for i, p in enumerate(info["perspectives"])) or "| — | _(none)_ |"
    outline_rows = "\n".join(f"| `{s['title']}` | {len(s['points'])} | {_esc(', '.join(s['points'][:3]))[:150]}{'…' if len(s['points']) > 3 else ''} |" for s in info["outline"]) or "| — | — | _(none)_ |"
    obs = []
    if info["n_persp"] >= 3: obs.append(f"**✅ {info['n_persp']} distinct perspectives** generated.")
    else: obs.append(f"**⚠️ Only {info['n_persp']} perspectives** — below default n_perspectives. Generator collapsed.")
    if info["n_q"] >= info["n_persp"] * 2 - 1: obs.append(f"**✅ {info['n_q']} questions** ≈ {info['n_persp']} × questions_per_perspective.")
    else: obs.append(f"**🤔 Only {info['n_q']} questions** for {info['n_persp']} perspectives — some perspective Q-gen failed.")
    if info["chars"] > 1000: obs.append(f"**✅ Article assembled ({info['chars']} chars)** across {info['n_sec']} sections.")
    else: obs.append(f"**⚠️ Article very short ({info['chars']} chars)** — writer collapsed sections or LLM hedged.")
    obs_block = "\n\n".join(f"- {o}" for o in obs)

    return f"""## 9 · What we just observed

The cells above ran STORM's 5-stage pipeline on the agentic-AI 2024 topic.

### 9.1 · Pipeline statistics

- **Perspectives generated**: {info['n_persp']}
- **Questions generated**: {info['n_q']}
- **Outline sections**: {info['n_sec']}
- **Final article**: {info['chars']} chars

### 9.2 · The perspectives the agent chose

| # | Perspective |
|---|---|
{persp_rows}

### 9.3 · The outline that emerged

| Title | # key points | First few points |
|---|---|---|
{outline_rows}

### 9.4 · Patterns surfaced in this run

{obs_block}

### 9.5 · The takeaway

STORM's value is in **diversity-by-construction**: a single "write about X" prompt collapses to one voice; STORM forces N perspectives → N×K questions → answers from many angles → an outline that reflects that breadth. Read § 9.2 to gauge whether the perspectives really diverge (paraphrases = pipeline broken) and § 9.3 to confirm the outline draws from multiple perspective threads."""


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
    print(f"tailored: n_persp={info['n_persp']} n_q={info['n_q']} n_sec={info['n_sec']} chars={info['chars']}")

if __name__ == "__main__": main()
