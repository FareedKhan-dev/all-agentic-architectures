"""Post-process notebook 29: rewrite § 9 against the Voyager captured run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "29_voyager.ipynb"
ANSI = re.compile(r"\x1b\[[0-9;]*[mGKH]")


def cell_output_text(cell):
    chunks = []
    for o in cell.outputs:
        t = o.get("text", "") or o.get("data", {}).get("text/plain", "")
        if isinstance(t, list): t = "".join(t)
        chunks.append(ANSI.sub("", str(t)))
    return "\n".join(chunks)


TASK = re.compile(
    r"TASK_TAG:\s+(\w+)\s*\n"
    r"\s+TASK:\s+([^\n]+)\s*\n"
    r"\s+DECISION:\s+(\w+)\s*\n"
    r"\s+SKILL_NAME:\s+([^\n]+)\s*\n"
    r"\s+INVOCATION:\s+([^\n]+)\s*\n"
    r"\s+LIBRARY_SIZE:\s+(\d+)\s+->\s+(\d+)\s*\n"
    r"\s+EXECUTION_OK:\s+(True|False)\s*\n"
    r"\s+EXECUTED_STDOUT:\s+([^\n]+)\s*\n"
    r"\s+LLM_PREDICTED:\s+([^\n]+)\s*\n"
    r"\s+ANSWER:\s+([^\n]+)\s*\n"
    r"\s+EXPECTED:\s+([^\n]+)\s*\n"
    r"\s+MATCH:\s+(True|False)"
)
LIB = re.compile(r"FINAL_LIBRARY_SIZE:\s+(\d+)((?:\s+-\s+skill\s+`[^`]+`:.+?\n)+)", re.DOTALL)


def extract_run(nb):
    info = {"tasks": [], "final_lib_size": 0, "skills": []}
    for cell in nb.cells:
        if cell.cell_type != "code": continue
        text = cell_output_text(cell)
        if "TASK_TAG:" in cell.source and "FINAL_LIBRARY_SIZE" in cell.source:
            tasks = []
            for m in TASK.finditer(text):
                tag, q, dec, name, inv, lib_b, lib_a, ok, stdout, pred, ans, exp, mt = m.groups()
                tasks.append({
                    "tag": tag.strip(), "task": q.strip(), "decision": dec.strip(),
                    "skill": name.strip(), "invocation": inv.strip(),
                    "lib_before": int(lib_b), "lib_after": int(lib_a),
                    "execution_ok": ok == "True",
                    "stdout": stdout.strip().strip("'\""),
                    "llm_predicted": pred.strip().strip("'\""),
                    "answer": ans.strip(), "expected": exp.strip(), "match": mt == "True",
                })
            info["tasks"] = tasks
            m = LIB.search(text)
            if m:
                info["final_lib_size"] = int(m.group(1))
                info["skills"] = [
                    {"name": sm.group(1), "desc": sm.group(2).strip()}
                    for sm in re.finditer(r"-\s+skill\s+`([^`]+)`:\s+(.+)", m.group(2))
                ]
    return info


def _esc(s): return s.replace("|", "\\|").strip()


def make_commentary(info):
    tasks = info["tasks"]
    skills = info["skills"]
    if tasks:
        rows = "\n".join(
            f"| `{t['tag']}` | `{t['decision']}` | `{t['skill']}` | `{t['invocation']}` | "
            f"{t['lib_before']}→{t['lib_after']} | {'✅' if t['execution_ok'] else '❌'} | "
            f"`{t['stdout']}` | `{t['llm_predicted']}` | {'✅' if t['match'] else '❌'} |"
            for t in tasks
        )
    else:
        rows = "| — | — | — | — | — | — | — | — | — |"
    skill_block = "\n".join(f"- **`{s['name']}`** — {_esc(s['desc'])}" for s in skills) or "_(no skills captured)_"
    obs = []
    if len(tasks) >= 2 and tasks[1]["decision"] == "reuse" and tasks[1]["skill"] == tasks[0]["skill"]:
        obs.append(f"**✅ Skill reuse worked**: task 1 wrote `{tasks[0]['skill']}`, task 2 retrieved + reused it.")
    elif len(tasks) >= 2 and tasks[1]["decision"] == "write_new":
        obs.append(f"**🤔 Task 2 didn't reuse task 1's skill** — decider chose write_new. Possibly vector retrieval missed the right skill, or decider was overly strict.")
    if len(tasks) >= 3 and tasks[2]["decision"] == "write_new":
        obs.append(f"**✅ Task 3 correctly wrote a new skill** (`{tasks[2]['skill']}`) — different task type from prior 2.")

    # Real-execution observations
    n_executed = sum(1 for t in tasks if t["execution_ok"])
    obs.append(f"**Real subprocess execution**: {n_executed}/{len(tasks)} skills ran successfully via `subprocess.run([sys.executable, '-I', '-c', script], timeout=5)`. The `Executed stdout` column shows the *actual* subprocess output, not an LLM prediction.")
    mismatches = [t for t in tasks if t["execution_ok"] and t["llm_predicted"] and t["stdout"] != t["llm_predicted"]]
    if mismatches:
        for t in mismatches:
            obs.append(f"**🔍 Prediction vs reality mismatch on `{t['tag']}`**: LLM predicted `{t['llm_predicted']}`, real exec returned `{t['stdout']}`. Real execution catches this; pure-LLM prediction would have shipped the wrong answer.")

    correct = sum(1 for t in tasks if t["match"])
    obs.append(f"**Accuracy: {correct}/{len(tasks)}** tasks answered correctly via the skill library.")
    obs_block = "\n\n".join(f"- {o}" for o in obs)

    return f"""## 9 · What we just observed

The cells above ran Voyager on 3 sequential tasks — two factorials and a Fibonacci — to exercise both the **reuse** path and the **write-new** path.

### 9.1 · Per-task summary

| Tag | Decision | Skill used | Invocation | Library size | Exec OK | Executed stdout | LLM predicted | Match |
|---|---|---|---|---|---|---|---|---|
{rows}

**Final library size**: {info['final_lib_size']}

### 9.2 · The skill library that built up

{skill_block}

### 9.3 · Patterns surfaced in this run

{obs_block}

### 9.4 · The takeaway

Voyager's value lives in two signals: the **`Decision`** column (how often does the agent reuse vs write?) and the **`Library size`** column (does it grow monotonically or plateau?). A healthy run:

1. **First instance of a task type** → write_new, library grows.
2. **Second instance of the same task type** → reuse, library stays.
3. **New task type** → write_new, library grows again.

When reuse fires correctly (§ 9.3), the architecture is amortising LLM cost across similar future tasks. The deterministic-picker is the LLM's `decision: Literal['reuse', 'write_new']` field plus a Python `if`-route — no numeric scoring."""


def main():
    nb = nbformat.read(NB_PATH, as_version=4)
    info = extract_run(nb)
    new_md = make_commentary(info)
    replaced = False
    for cell in nb.cells:
        if cell.cell_type == "markdown" and cell.source.lstrip().startswith("## 9 · What we just observed"):
            cell.source = new_md
            replaced = True; break
    if not replaced: raise RuntimeError("§9 not found")
    nbformat.write(nb, NB_PATH)
    print(f"tailored: {len(info['tasks'])} tasks, lib_size={info['final_lib_size']}, skills={[s['name'] for s in info['skills']]}")


if __name__ == "__main__": main()
