"""Post-process notebook 33: rewrite § 9 against SWE-Agent run."""
from __future__ import annotations
import ast
import re
from pathlib import Path
import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "33_swe_agent.ipynb"
ANSI = re.compile(r"\x1b\[[0-9;]*[mGKH]")


def cell_output_text(cell):
    chunks = []
    for o in cell.outputs:
        t = o.get("text", "") or o.get("data", {}).get("text/plain", "")
        if isinstance(t, list): t = "".join(t)
        chunks.append(ANSI.sub("", str(t)))
    return "\n".join(chunks)


def extract(nb):
    info = {"iterations": 0, "action_seq": [], "counts": {}, "observations": [], "final_answer": ""}
    for cell in nb.cells:
        if cell.cell_type != "code": continue
        text = cell_output_text(cell)
        if "ITERATIONS:" not in text or "ACTION_SEQUENCE" not in text: continue
        m = re.search(r"ITERATIONS:\s+(\d+)", text); info["iterations"] = int(m.group(1)) if m else 0
        m = re.search(r"ACTION_SEQUENCE:\s+(\[.*?\])", text)
        if m:
            try: info["action_seq"] = ast.literal_eval(m.group(1))
            except Exception: pass
        m = re.search(r"N_LIST:\s+(\d+),\s+N_READ:\s+(\d+),\s+N_WRITE:\s+(\d+),\s+N_RUN:\s+(\d+)", text)
        if m:
            info["counts"] = {"list": int(m.group(1)), "read": int(m.group(2)),
                              "write": int(m.group(3)), "run": int(m.group(4))}
        for om in re.finditer(r"\[obs (\d+)\]\s+(.+?)(?=\[obs |\Z)", text, re.DOTALL):
            info["observations"].append({"i": int(om.group(1)), "text": om.group(2).strip()[:300]})
        am = re.search(r"=== FINAL ANSWER ===\n(.+?)\n\n=== FINAL", text, re.DOTALL)
        if am: info["final_answer"] = am.group(1).strip()
    return info


def _esc(s): return s.replace("|", "\\|").strip()


def make_commentary(info):
    counts = info["counts"]
    seq = info["action_seq"]
    obs = info["observations"]
    obs_table = "\n".join(f"| {o['i']} | {_esc(o['text'])[:200]} |" for o in obs[:8]) or "| — | — |"

    flags = []
    if counts:
        if counts.get("read", 0) >= 1 and counts.get("write", 0) >= 1 and counts.get("run", 0) >= 1:
            flags.append("**✅ Full read → write → run cycle** — agent gathered context, fixed, verified.")
        else:
            missing = [k for k in ("read", "write", "run") if counts.get(k, 0) == 0]
            flags.append(f"**⚠️ Missing actions**: {missing}. Incomplete diagnose-fix-verify cycle.")
    if any("PASS" in o["text"] for o in obs):
        flags.append("**✅ A `run_check` returned `PASS`** — the bug fix was verified.")
    elif any("AssertionError" in o["text"] or "rc=1" in o["text"] for o in obs):
        flags.append("**⚠️ run_check still failing at end** — the fix didn't take.")
    flags_block = "\n\n".join(f"- {f}" for f in flags) if flags else "- No notable patterns."

    return f"""## 9 · What we just observed

The cells above ran SWE-Agent to fix a buggy `factorial.py` in a sandboxed directory.

### 9.1 · Action sequence

- **Iterations**: {info['iterations']}
- **Sequence**: {seq}
- **Counts**: list={counts.get('list', 0)}, read={counts.get('read', 0)}, write={counts.get('write', 0)}, run_check={counts.get('run', 0)}

### 9.2 · Per-observation log (first 8)

| # | observation excerpt |
|---|---|
{obs_table}

### 9.3 · Patterns surfaced

{flags_block}

### 9.4 · The takeaway

SWE-Agent's headline behaviour is the **diagnose → fix → verify** cycle visible in the action sequence: a healthy run shows at least one `read_file` (diagnose), one `write_file` (fix), and one `run_check` (verify). The categorical action enum keeps the agent on a finite, auditable tool surface — and `_safe_path()` ensures it can't escape the sandbox even when it tries."""


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
    print(f"tailored: iter={info['iterations']} seq={info['action_seq']} counts={info['counts']}")

if __name__ == "__main__": main()
