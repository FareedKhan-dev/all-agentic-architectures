"""Post-process notebook 14: rewrite § 9 against the DryRun captured run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "14_dry_run.ipynb"
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
    info: dict[str, object] = {"tasks": [], "details": []}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "TASKS = [" in cell.source:
            text = cell_output_text(cell)
            blocks = re.findall(
                r"\[(\w+)\]\s+approved=(\w+)\s+·\s+decided_by=(\w+)\s+·\s+irreversibility=(\d+)/5\s*\n(.+?)(?=\n\s*[›>]\s+\[\w+\]|\nGate-decision|\Z)",
                text,
                re.DOTALL,
            )
            tasks = []
            for tag, approved, decided, irrev, body in blocks:
                cmd_m = re.search(r"command:\s+(.+?)(?=\n\s*outcome:|\Z)", body, re.DOTALL)
                outcome_m = re.search(r"outcome:\s+(.+?)(?=\n\s*[›>]|\Z)", body, re.DOTALL)
                tasks.append({
                    "tag": tag,
                    "approved": approved == "True",
                    "decided_by": decided,
                    "irreversibility": int(irrev),
                    "command": _normalize_ws(cmd_m.group(1)) if cmd_m else "",
                    "outcome": _normalize_ws(outcome_m.group(1)) if outcome_m else "",
                })
            info["tasks"] = tasks
        if "for tag, task, r in results" in cell.source:
            text = cell_output_text(cell)
            sections = re.findall(
                r"===\s+(\w+)\s+===\s*\n(.+?)(?=\n\s*[─-╿]+\s*\n===|\Z)",
                text,
                re.DOTALL,
            )
            details = []
            for tag, body in sections:
                details.append({"tag": tag.lower(), "body": _normalize_ws(body)[:1500]})
            info["details"] = details
    return info


def make_commentary(info: dict[str, object]) -> str:
    tasks: list[dict] = info.get("tasks", [])  # type: ignore[assignment]

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ").strip()

    summary_table = (
        "\n".join(
            f"| {t['tag']} | {t['irreversibility']}/5 | `{t['decided_by']}` | "
            f"{'✓' if t['approved'] else '✗'} | {esc(t['command'])[:60]}{'…' if len(t['command']) > 60 else ''} |"
            for t in tasks
        )
        if tasks
        else "| — | — | — | — | _(no tasks captured)_ |"
    )

    obs: list[str] = []

    # Gate-decision distribution
    python_caps = sum(1 for t in tasks if t["decided_by"] == "python_hard_cap")
    llm_revs = sum(1 for t in tasks if t["decided_by"] == "llm_reviewer")
    if tasks:
        obs.append(
            f"**Gate-decision distribution**: `python_hard_cap` fired on {python_caps}/{len(tasks)} "
            f"task(s); `llm_reviewer` decided {llm_revs}/{len(tasks)}. The Python hard-cap is the "
            "deterministic backstop — it should fire on the destructive task."
        )

    # Destructive task should be blocked by python_hard_cap
    destructive = next((t for t in tasks if t["tag"] == "destructive"), None)
    if destructive:
        if destructive["irreversibility"] >= 4 and destructive["decided_by"] == "python_hard_cap":
            obs.append(
                "**Hard-cap fired correctly on destructive task** — irreversibility "
                f"{destructive['irreversibility']}/5 ≥ threshold 4 → Python blocked "
                "unconditionally without asking the LLM reviewer. This is the "
                "deterministic-picker pattern doing its job."
            )
        elif destructive["decided_by"] == "llm_reviewer":
            obs.append(
                f"**Hard-cap DIDN'T fire on destructive task** — irreversibility "
                f"only {destructive['irreversibility']}/5 (below threshold 4); LLM "
                "reviewer had to decide. Either the dry-runner under-rated the "
                "destructive action (a real failure mode), or the threshold should be lower."
            )

    # Routine should be approved (or be informative if over-blocked)
    routine = next((t for t in tasks if t["tag"] == "routine"), None)
    if routine:
        if routine["approved"]:
            obs.append(
                "**Routine task approved** — the LLM reviewer correctly distinguished "
                "low-risk from high-risk actions."
            )
        else:
            obs.append(
                "**Routine task BLOCKED by LLM reviewer** — same-model reviewers "
                "are often over-conservative (see § 11.1). The Python hard-cap "
                "isn't the issue here; the LLM reviewer is. Mitigation: stricter "
                "approval-prompt examples of 'safe to approve', or use a different "
                "model in the reviewer seat."
            )

    # Irreversibility ordering check
    if len(tasks) >= 2:
        irrev_seq = [t["irreversibility"] for t in tasks]
        if irrev_seq == sorted(irrev_seq):
            obs.append(
                f"**Irreversibility escalated correctly** across tasks: {irrev_seq}. "
                "The dry-runner has reasonable risk calibration."
            )
        else:
            obs.append(
                f"**Irreversibility not monotone** across tasks: {irrev_seq}. "
                "The dry-runner is rating risk inconsistently — possible calibration "
                "issue. Compare to the schema's rubric examples."
            )

    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns."

    return f"""## 9 · What we just observed

The cells above ran **3 tasks of escalating risk** through the same DryRun architecture (threshold=4) to exercise all three decision branches.

### 9.1 · Quantitative summary

| Task | irreversibility | Decided by | Approved | Proposed command (truncated) |
|---|---|---|---|---|
{summary_table}

### 9.2 · Patterns surfaced in this run

{obs_block}

### 9.3 · The takeaway

A *healthy* DryRun run produces this distribution across escalating tasks:

1. **Routine** → LLM approves → mock-execute (irreversibility 1-2)
2. **Moderate** → LLM judges (could go either way; conservative default)
3. **Destructive** → **Python hard-cap blocks** without LLM input (irreversibility ≥ 4)

The Python hard-cap exists because LLM safety reviewers can be sycophantic, prompt-injected, or just calibrated wrong. **Never** ship a Dry-Run pattern without a deterministic backstop on the most dangerous category."""


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
    print(f"tailored section 9: {len(info['tasks'])} task results captured")


if __name__ == "__main__":
    main()
