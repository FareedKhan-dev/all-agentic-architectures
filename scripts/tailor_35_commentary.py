"""Post-process notebook 35: rewrite § 9 against AWM run."""
from __future__ import annotations
import re
from pathlib import Path
import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "35_agent_workflow_memory.ipynb"
ANSI = re.compile(r"\x1b\[[0-9;]*[mGKH]")


def cell_output_text(cell):
    chunks = []
    for o in cell.outputs:
        t = o.get("text", "") or o.get("data", {}).get("text/plain", "")
        if isinstance(t, list): t = "".join(t)
        chunks.append(ANSI.sub("", str(t)))
    return "\n".join(chunks)


TASK = re.compile(
    r"TASK_(\d+):\s+(.+?)\s*\n"
    r"\s+USED_RETRIEVED:\s+(True|False)\s*\n"
    r"\s+RETRIEVED_TYPE:\s+'([^']*)'\s*\n"
    r"\s+EXTRACTED_TYPE:\s+'([^']*)'\s*\n"
    r"\s+LIBRARY:\s+(\d+)\s+->\s+(\d+)\s*\n"
    r"\s+ANSWER:\s+([^\n]+(?:\n(?![ \t]*TASK_|\s*FINAL_LIBRARY)[^\n]*)*)"
)


def extract(nb):
    info = {"tasks": [], "lib_size": 0, "workflows": []}
    for cell in nb.cells:
        if cell.cell_type != "code": continue
        text = cell_output_text(cell)
        if "TASK_1:" not in text: continue
        for m in TASK.finditer(text):
            n, q, used, rt, et, lb, la, ans = m.groups()
            info["tasks"].append({
                "n": int(n), "task": q.strip(), "used": used == "True",
                "ret_type": rt, "ext_type": et, "lib_b": int(lb), "lib_a": int(la),
                "answer": ans.strip(),
            })
        m = re.search(r"FINAL_LIBRARY_SIZE:\s+(\d+)", text)
        if m: info["lib_size"] = int(m.group(1))
        for wm in re.finditer(r"-\s+workflow\s+`([^`]+)`:\n((?:\s+•\s+[^\n]+\n)+)", text):
            wt = wm.group(1)
            steps = [s.strip().removeprefix("•").strip() for s in wm.group(2).splitlines() if s.strip()]
            info["workflows"].append({"type": wt, "steps": steps})
    return info


def _esc(s): return s.replace("|", "\\|").strip()


def make_commentary(info):
    rows = "\n".join(
        f"| {t['n']} | {'✅' if t['used'] else '❌'} | `{t['ret_type'] or '—'}` | `{t['ext_type']}` | {t['lib_b']}→{t['lib_a']} | {_esc(t['answer'])[:80]} |"
        for t in info["tasks"]
    ) or "| — | — | — | — | — | — |"
    wf_block = "\n\n".join(
        f"**`{w['type']}`**\n" + "\n".join(f"  - {s}" for s in w["steps"])
        for w in info["workflows"]
    ) or "_(no workflows captured)_"
    obs = []
    if len(info["tasks"]) >= 2 and info["tasks"][1]["used"]:
        obs.append(f"**✅ Task 2 reused task 1's workflow** (`{info['tasks'][1]['ret_type']}`). Memory pipeline alive.")
    elif len(info["tasks"]) >= 2:
        obs.append("**🤔 Task 2 did NOT reuse a workflow** — either retrieval missed or no library entry yet.")
    if len(info["tasks"]) >= 3 and info["tasks"][2]["used"]:
        obs.append(f"**✅ Task 3 reused a workflow** (`{info['tasks'][2]['ret_type']}`).")
    obs.append(f"**Final library size**: {info['lib_size']} workflow(s) extracted across {len(info['tasks'])} task(s).")
    obs_block = "\n\n".join(f"- {o}" for o in obs)
    return f"""## 9 · What we just observed

3 sequential summarise-and-categorise tasks. Task 1 extracts a workflow; tasks 2 and 3 should retrieve and follow it.

### 9.1 · Per-task workflow usage

| # | Used retrieved? | Retrieved type | Extracted type | Library | Answer |
|---|---|---|---|---|---|
{rows}

### 9.2 · The workflow library that built up

{wf_block}

### 9.3 · Patterns surfaced

{obs_block}

### 9.4 · The takeaway

AWM's value is in the **`Used retrieved?`** column of § 9.1. The first task always extracts (no library yet); subsequent structurally-similar tasks should *retrieve and use* the prior workflow. If the column reads `✅, ❌, ❌` across 3 similar tasks, vector retrieval is broken; if `✅, ✅, ✅` it's working — the library is amortising strategy cost across calls."""


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
    print(f"tailored: tasks={len(info['tasks'])} lib_size={info['lib_size']} workflows={[w['type'] for w in info['workflows']]}")

if __name__ == "__main__": main()
