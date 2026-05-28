"""Post-process notebook 28: rewrite § 9 against the Debate captured run."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "28_debate.ipynb"
ANSI = re.compile(r"\x1b\[[0-9;]*[mGKH]")


def cell_output_text(cell: nbformat.NotebookNode) -> str:
    chunks: list[str] = []
    for o in cell.outputs:
        t = o.get("text", "") or o.get("data", {}).get("text/plain", "")
        if isinstance(t, list):
            t = "".join(t)
        chunks.append(ANSI.sub("", str(t)))
    return "\n".join(chunks)


def extract_run(nb: nbformat.NotebookNode) -> dict[str, object]:
    info: dict[str, object] = {"answer": None, "expected": None, "match": None, "convergence": None,
                                "round_unique": None, "tally": None, "rounds": []}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        text = cell_output_text(cell)
        if "FINAL_ANSWER:" not in text or "FINAL_TALLY" not in text:
            continue
        m = re.search(r"FINAL_ANSWER:\s+(.+?)\n", text)
        if m: info["answer"] = m.group(1).strip()
        m = re.search(r"EXPECTED:\s+(.+?)\n", text)
        if m: info["expected"] = m.group(1).strip()
        m = re.search(r"MATCH:\s+(True|False)", text)
        if m: info["match"] = m.group(1) == "True"
        m = re.search(r"CONVERGENCE:\s+(\w+)", text)
        if m: info["convergence"] = m.group(1)
        m = re.search(r"ROUND_UNIQUE_COUNTS:\s+(\[.*?\])", text)
        if m:
            try: info["round_unique"] = ast.literal_eval(m.group(1))
            except Exception: pass
        m = re.search(r"FINAL_TALLY:\s+(\{.*?\})", text)
        if m:
            try: info["tally"] = ast.literal_eval(m.group(1))
            except Exception: pass
        # Per-round responses
        rounds_data = []
        for rm in re.finditer(r"--- ROUND (\d+) ---\n((?:\s*Agent.+?\n\s+critique:.+?\n)+)", text, re.DOTALL):
            round_num = int(rm.group(1))
            agents = []
            for am in re.finditer(r"\s*Agent\s+(\w+):\s+answer=([^\s]+)\n\s+critique:\s+(.+?)(?=\n\s*Agent|\n---|\Z)", rm.group(2), re.DOTALL):
                agents.append({"agent": am.group(1), "answer": am.group(2).strip("'\""), "critique": am.group(3).strip()})
            rounds_data.append({"round": round_num, "agents": agents})
        info["rounds"] = rounds_data
    return info


def _esc(s):
    return s.replace("|", "\\|").replace("\n", " ").strip()


def make_commentary(info):
    answer = info.get("answer") or "(no answer)"
    expected = info.get("expected") or "?"
    match = info.get("match")
    convergence = info.get("convergence") or "?"
    round_unique = info.get("round_unique") or []
    tally = info.get("tally") or {}
    rounds = info.get("rounds") or []

    summary = (
        f"- **Winner**: `{answer}` — {'✅ matches' if match else '❌ differs from'} expected `{expected}`\n"
        f"- **Convergence**: {convergence}\n"
        f"- **Unique answers per round**: {round_unique}\n"
        f"- **Final tally**: {tally}"
    )

    if rounds:
        round_blocks = []
        for rd in rounds:
            agent_rows = "\n".join(
                f"| Agent {a['agent']} | `{a['answer']}` | {_esc(a['critique'])[:120]}{'…' if len(a['critique']) > 120 else ''} |"
                for a in rd["agents"]
            )
            round_blocks.append(
                f"#### Round {rd['round']}\n\n"
                f"| Agent | Answer | Critique |\n"
                f"|---|---|---|\n"
                f"{agent_rows}"
            )
        rounds_section = "\n\n".join(round_blocks)
    else:
        rounds_section = "_(no rounds captured)_"

    obs = []
    if round_unique:
        if len(round_unique) >= 2 and round_unique[0] > 1 and round_unique[-1] == 1:
            obs.append(f"**✅ Debate converged** — agents disagreed in round 1 ({round_unique[0]} unique answers) but agreed by round {len(round_unique)} ({round_unique[-1]} unique). Cross-talk worked.")
        elif round_unique and round_unique[-1] == 1:
            obs.append(f"**🟰 Started in agreement** — all {len(rounds[0]['agents']) if rounds else '?'} agents agreed from round 1. Debate added no lift on this task.")
        elif round_unique and round_unique[-1] > 1:
            obs.append(f"**⚠️ No full convergence** — {round_unique[-1]} unique answers in the final round; majority vote decided. Consider more rounds or a judge LLM.")
    if match:
        obs.append(f"**✅ Majority vote landed on the correct answer** (`{answer}`).")
    elif match is False:
        obs.append(f"**❌ Majority vote was wrong** (got `{answer}`, expected `{expected}`). Group-think can fail; pair with verification.")
    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns."

    return f"""## 9 · What we just observed

The cells above ran a 3-agent × 2-round debate on the Sally-siblings trick problem.

### 9.1 · Summary

{summary}

### 9.2 · Per-round agent responses

{rounds_section}

### 9.3 · Patterns surfaced in this run

{obs_block}

### 9.4 · The takeaway

Debate's value is in `ROUND_UNIQUE_COUNTS`: a `[2, 1]` sequence means agents disagreed in round 1, then converged by round 2 — that's the cross-talk paying off. A `[1, 1]` sequence means everyone agreed from the start (debate wasted N×K calls); `[2, 2]` means no convergence and majority vote decided based on first-round noise.

The deterministic-picker is `Counter.most_common(1)` on the last round — same pattern as Self-Consistency (nb 21), but the votes were *informed by* peers' arguments, not independent."""


def main():
    nb = nbformat.read(NB_PATH, as_version=4)
    info = extract_run(nb)
    new_md = make_commentary(info)
    replaced = False
    for cell in nb.cells:
        if cell.cell_type == "markdown" and cell.source.lstrip().startswith("## 9 · What we just observed"):
            cell.source = new_md
            replaced = True
            break
    if not replaced:
        raise RuntimeError("section 9 not found")
    nbformat.write(nb, NB_PATH)
    print(f"tailored: answer={info.get('answer')}, match={info.get('match')}, round_unique={info.get('round_unique')}, rounds={len(info.get('rounds') or [])}")


if __name__ == "__main__":
    main()
