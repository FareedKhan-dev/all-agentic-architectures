"""Post-process notebook 34: rewrite § 9 against real-browser BrowserAgent run."""
from __future__ import annotations
import ast
import re
from pathlib import Path
import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "34_computer_use.ipynb"
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
    r"\s+ITERATIONS:\s+(\d+)\s*\n"
    r"\s+ACTION_SEQUENCE:\s+(\[.*?\])\s*\n"
    r"\s+N_BLOCKED:\s+(\d+)\s*\n"
    r"\s+CURRENT_URL:\s+([^\n]*)\s*\n"
    r"\s+PAGE_TEXT_CHARS:\s+(\d+)\s*\n"
    r"\s+ANSWER:\s+([^\n]*(?:\n(?![ \t]*TASK_TAG:|=== )[^\n]*)*)"
)
PERACTION = re.compile(
    r"---\s+(\w+)\s+---\s*\n"
    r"((?:\s+\[\d+\]\s+[^\n]+(?:\n\s+→ BLOCKED:\s+[^\n]+)?\n)+)"
)


def extract(nb):
    info = {"tasks": [], "action_log": {}}
    for cell in nb.cells:
        if cell.cell_type != "code": continue
        text = cell_output_text(cell)
        if "TASK_TAG:" not in text: continue
        for m in TASK.finditer(text):
            tag, q, iters, seq_s, blocked, url, chars, ans = m.groups()
            try: seq = ast.literal_eval(seq_s)
            except Exception: seq = []
            info["tasks"].append({
                "tag": tag, "task": q.strip(), "iter": int(iters),
                "seq": seq, "n_blocked": int(blocked), "url": url.strip(),
                "chars": int(chars), "answer": ans.strip(),
            })
        for am in PERACTION.finditer(text):
            tag = am.group(1)
            entries = []
            for line in am.group(2).splitlines():
                lm = re.match(r"\s+\[(\d+)\]\s+([✅🛑])\s+action=(\w+)\s+target=(.+)", line)
                if lm:
                    entries.append({
                        "i": int(lm.group(1)),
                        "allowed": lm.group(2) == "✅",
                        "action": lm.group(3),
                        "target": lm.group(4).strip("'\""),
                    })
                bm = re.match(r"\s+→ BLOCKED:\s+(.+)", line)
                if bm and entries:
                    entries[-1]["reason"] = bm.group(1).strip()
            info["action_log"][tag] = entries
    return info


def _esc(s): return s.replace("|", "\\|").strip()


def make_commentary(info):
    tasks = info["tasks"]
    if tasks:
        rows = "\n".join(
            f"| `{t['tag']}` | {t['iter']} | {t['seq']} | {t['n_blocked']} | {t['url'][:60]} | {_esc(t['answer'])[:80]} |"
            for t in tasks
        )
    else:
        rows = "| — | — | — | — | — | — |"

    action_blocks = []
    for tag, entries in info["action_log"].items():
        if not entries:
            action_blocks.append(f"**`{tag}`** — _(no log)_")
            continue
        lines = "\n".join(
            f"| [{e['i']}] | {'✅' if e['allowed'] else '🛑 BLOCKED'} | `{e['action']}` | {_esc(e.get('target', ''))[:60]} | {_esc(e.get('reason', '—'))[:60]} |"
            for e in entries
        )
        action_blocks.append(f"**`{tag}`** action log:\n\n| # | verdict | action | target | block reason |\n|---|---|---|---|---|\n{lines}")
    log_section = "\n\n".join(action_blocks) or "_(no action logs)_"

    obs = []
    real_tasks = [t for t in tasks if t["tag"] == "real_nav"]
    blocked_tasks = [t for t in tasks if t["tag"] == "blocked_nav"]
    if real_tasks and "example" in real_tasks[0]["answer"].lower():
        obs.append("**✅ Real navigation produced the expected answer** (`Example Domain`). Playwright opened headless Chromium, the agent navigated to example.com and returned the correct heading.")
        if real_tasks[0]["chars"] == 0:
            obs.append("**🤔 Agent answered without calling `extract_text` first.** It produced the right heading anyway (Llama remembers example.com from its training). Stronger prompt could force an extract before answer for unfamiliar sites.")
    elif real_tasks:
        obs.append(f"**🤔 Real navigation finished but answer unclear** (`{real_tasks[0]['answer'][:80]}`). Check the action log for issues.")
    if blocked_tasks and blocked_tasks[0]["n_blocked"] > 0:
        obs.append(f"**✅ Safety gate fired** on `blocked_nav` ({blocked_tasks[0]['n_blocked']} action(s) blocked). The Python check stopped the navigation BEFORE Playwright touched the URL.")
    elif blocked_tasks:
        obs.append("**🤔 No blocks on `blocked_nav`** — either the agent didn't attempt the blocked URL, or the safety gate missed it. Audit the action log.")
    obs_block = "\n\n".join(f"- {o}" for o in obs)

    return f"""## 9 · What we just observed

We ran two tasks against a real headless Chromium browser:
1. **`real_nav`** — visit example.com, extract the heading (should succeed end-to-end).
2. **`blocked_nav`** — try to navigate to a domain in `blocked_domains` (safety gate must block).

### 9.1 · Per-task summary

| Tag | Iters | Action sequence | Blocked | Final URL | Answer |
|---|---|---|---|---|---|
{rows}

### 9.2 · Per-action verdict log

{log_section}

### 9.3 · Patterns surfaced

{obs_block}

### 9.4 · The takeaway

This nb's value is showing **the safety gate working on a real browser, not a mock**. The two columns to watch in § 9.1 are **`Action sequence`** (was there an `answer` at the end?) and **`Blocked`** (did the gate fire when expected?). A healthy `real_nav` ends with `answer` and 0 blocks; a healthy `blocked_nav` shows `navigate` attempted and the gate blocking it before Playwright sees the URL.

The deterministic-picker pattern lives in `_check_safety()` — it's a Python function comparing literal strings against pattern lists. The LLM is never asked "is this URL safe?" because that question is prompt-injectable. The Python check looks at the raw `target` field that came out of the structured-output schema and decides allowed/blocked deterministically."""


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
    print(f"tailored: {len(info['tasks'])} tasks, action_logs={list(info['action_log'].keys())}")

if __name__ == "__main__": main()
