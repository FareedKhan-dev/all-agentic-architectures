"""Post-process notebook 31: rewrite § 9 against MemGPT run."""
from __future__ import annotations
import ast
import re
from pathlib import Path
import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "31_memgpt.ipynb"
ANSI = re.compile(r"\x1b\[[0-9;]*[mGKH]")


def cell_output_text(cell):
    chunks = []
    for o in cell.outputs:
        t = o.get("text", "") or o.get("data", {}).get("text/plain", "")
        if isinstance(t, list): t = "".join(t)
        chunks.append(ANSI.sub("", str(t)))
    return "\n".join(chunks)


TURN = re.compile(
    r"TURN_(\d+):\s+(.+?)\s*\n"
    r"\s+ACTIONS:\s+(\[.*?\])\s*\n"
    r"\s+CONTEXT:\s+(\d+)\s+->\s+(\d+)\s*\n"
    r"\s+ARCHIVAL:\s+(\d+)\s+->\s+(\d+)\s*\n"
    r"\s+ANSWER:\s+([^\n]+(?:\n(?![ \t]*TURN_)[^\n]*)*)"
)


def extract(nb):
    info = {"turns": [], "final_context": [], "final_archival_count": 0}
    for cell in nb.cells:
        if cell.cell_type != "code": continue
        text = cell_output_text(cell)
        if "TURN_1:" not in text: continue
        for m in TURN.finditer(text):
            n, turn, actions_s, cb, ca, ab, aa, ans = m.groups()
            try: actions = ast.literal_eval(actions_s)
            except Exception: actions = []
            info["turns"].append({
                "n": int(n), "input": turn.strip(), "actions": actions,
                "ctx_before": int(cb), "ctx_after": int(ca),
                "arch_before": int(ab), "arch_after": int(aa),
                "answer": ans.strip(),
            })
        fm = re.search(r"FINAL_CONTEXT:\s+(\[.*?\])", text, re.DOTALL)
        if fm:
            try: info["final_context"] = ast.literal_eval(fm.group(1))
            except Exception: pass
        am = re.search(r"FINAL_ARCHIVAL_COUNT:\s+(\d+)", text)
        if am: info["final_archival_count"] = int(am.group(1))
    return info


def _esc(s): return s.replace("|", "\\|").strip()


def make_commentary(info):
    turns = info["turns"]
    if turns:
        rows = "\n".join(
            f"| {t['n']} | {_esc(t['input'])[:60]} | {t['actions']} | "
            f"{t['ctx_before']}→{t['ctx_after']} | {t['arch_before']}→{t['arch_after']} | "
            f"{_esc(t['answer'])[:60]} |"
            for t in turns
        )
    else:
        rows = "| — | — | — | — | — | — |"

    obs = []
    # Eviction check: ctx_after should hit the cap
    if turns and any(t["ctx_after"] >= 3 for t in turns):
        obs.append("**✅ Context tier reached capacity** — FIFO eviction triggered, pushed older facts to archival.")
    # Search check: any turn use search_archival?
    searched = any("search_archival" in t["actions"] for t in turns)
    if searched:
        obs.append("**✅ Agent searched archival** at least once — paged a fact back from disk to answer.")
    else:
        obs.append("**🤔 Agent never searched archival** — either all facts stayed in context, or it failed to recognise the recall need (and may have hallucinated).")
    # Final state
    obs.append(f"**Final state**: context tier has {len(info['final_context'])} items, archival has {info['final_archival_count']}.")

    return f"""## 9 · What we just observed

The cells above ran 5 turns through MemGPT with `context_limit=3`. Turns 1-4 ingest facts; turn 5 queries the first fact (which should have been evicted by then).

### 9.1 · Per-turn memory state

| Turn | Input | Actions | Context size | Archival size | Answer |
|---|---|---|---|---|---|
{rows}

### 9.2 · Patterns surfaced

- {chr(10).join(f'- {o}' for o in obs)}

### 9.3 · The takeaway

MemGPT's value is the **two-tier discipline**: the agent never loses information (everything evicted survives in archival), but the active context stays bounded. Watch the **`ACTIONS`** column in § 9.1: a healthy run shows `write_to_archival` for new facts and `search_archival` when a query needs an evicted fact. If turn 5 didn't search but still answered correctly, the agent might have hallucinated; if it searched, the architecture's paging worked end-to-end."""


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
    print(f"tailored: {len(info['turns'])} turns, final_ctx={len(info['final_context'])}, archival={info['final_archival_count']}")

if __name__ == "__main__": main()
