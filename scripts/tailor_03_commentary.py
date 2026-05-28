"""Post-process notebook 03: rewrite § 9 with the actual ReAct trace."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "03_react.ipynb"
ANSI = re.compile(r"\x1b\[[0-9;]*[mGKH]")


def _normalize_ws(s: str) -> str:
    """Collapse rich console wrapping (multi-line w/ box chars) into a single line."""
    s = re.sub(r"[─-╿]", "", s)  # box-drawing chars
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
    info: dict[str, object] = {
        "answer": "",
        "tool_calls": 0,
        "thought_count": 0,
        "rounds": 0,
        "thoughts": [],
        "queries": [],
    }
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "result = arch.run(TASK)" in cell.source:
            text = cell_output_text(cell)
            # Capture everything between "Final answer" header and the metrics line.
            m = re.search(
                r"Final answer[\s\S─-╿]*?\n(.+?)\n[\s─-]*\n[\s─-╿]*?\d+\s+tool call",
                text,
                re.DOTALL,
            )
            if m:
                # Strip trailing horizontal-rule characters and whitespace.
                info["answer"] = re.sub(r"[\s─-╿]+$", "", m.group(1)).strip()
            m = re.search(
                r"(\d+)\s+tool call\(s\)\s+\W\s+(\d+)\s+thought\(s\)\s+\W\s+(\d+)\s+agent round",
                text,
            )
            if m:
                info["tool_calls"] = int(m.group(1))
                info["thought_count"] = int(m.group(2))
                info["rounds"] = int(m.group(3))
        if "for i, t in enumerate(result.trace" in cell.source:
            text = cell_output_text(cell)
            # THOUGHT lines from the trace pretty-printer (rich console).
            info["thoughts"] = [
                _normalize_ws(m).strip()
                for m in re.findall(
                    r"\bTHOUGHT\b[^\n]*\n(.+?)(?=\n\s*\[\d+\]\s|\Z)",
                    text,
                    re.DOTALL,
                )
            ]
            info["queries"] = [
                _normalize_ws(q)
                for q in re.findall(r"\bACTION\b[^`]*`([^`]+)`", text)
            ]
            # Try to recover the final answer from the trace too, as a fallback.
            if not info["answer"]:
                m = re.search(
                    r"FINAL ANSWER[^\n]*\n(.+?)(?=\n\s*\[\d+\]\s|\Z)", text, re.DOTALL
                )
                if m:
                    info["answer"] = _normalize_ws(m.group(1)).strip()
    return info


def make_commentary(info: dict[str, object]) -> str:
    n_calls = info.get("tool_calls", 0)
    n_thoughts = info.get("thought_count", 0)
    n_rounds = info.get("rounds", 0)
    thoughts: list[str] = info.get("thoughts", [])  # type: ignore[assignment]
    queries: list[str] = info.get("queries", [])  # type: ignore[assignment]
    answer: str = info.get("answer", "")  # type: ignore[assignment]

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ").strip()

    pairs_table = ""
    pairs = list(zip(thoughts, queries))
    if pairs:
        pairs_table = "\n".join(
            f"| {i+1} | {esc(t)[:160]}{'…' if len(t) > 160 else ''} | `{esc(q)}` |"
            for i, (t, q) in enumerate(pairs)
        )
    else:
        pairs_table = "| — | _(no thought–action pairs captured)_ | _(no actions)_ |"

    obs: list[str] = []

    if n_thoughts == 0 and n_calls > 0:
        obs.append(
            "**ReAct degraded to plain Tool Use.** `thought_count = 0` despite "
            f"`tool_calls = {n_calls}` — the model ignored the system prompt's "
            "Thought-discipline. Mitigation: tighten the prompt with an explicit "
            "\"You MUST start with 'Thought:'\" — or switch to a model that follows "
            "instructions more strictly."
        )
    elif n_thoughts < n_calls:
        obs.append(
            f"**Partial Thought discipline.** {n_thoughts} thought(s) for {n_calls} "
            "tool call(s) — some actions had no preceding thought. The thought "
            "discipline is on the edge of breaking."
        )
    elif n_thoughts == n_calls:
        obs.append(
            "**Clean ReAct discipline.** Every tool call was preceded by exactly one "
            "thought — the model is following the system-prompt format correctly."
        )

    if queries and len(set(queries)) < len(queries):
        obs.append(
            f"**Query repetition.** {len(queries) - len(set(queries))} of "
            f"{len(queries)} actions were duplicate queries. ReAct's thought step is "
            "supposed to *prevent* this by forcing the model to justify the next call. "
            "When it happens anyway, the thoughts were probably hollow."
        )

    if thoughts:
        avg_len = sum(len(t) for t in thoughts) / len(thoughts)
        if avg_len < 50:
            obs.append(
                f"**Hollow thoughts.** Average thought length is {avg_len:.0f} chars — "
                "too short to be substantive reasoning. The model is producing "
                "thought-shaped strings without doing real planning."
            )

    if answer and "http" not in answer.lower():
        obs.append(
            "**No URL in the final answer** despite the task asking for citation. "
            "The agent had tool results in context but didn't ground the final answer "
            "in them — see Corrective RAG (nb 24) for the proper fix."
        )

    if not obs:
        obs.append(
            "No common pathologies surfaced — the agent thought, searched, and "
            "answered cleanly. Re-run to see whether this is repeatable; ReAct "
            "discipline is provider- and model-sensitive."
        )

    obs_block = "\n\n".join(f"- {o}" for o in obs)

    answer_block = (
        "> " + (answer[:500].replace("\n", "\n> ") if answer else "_(no answer captured)_")
        + ("…" if len(answer) > 500 else "")
    )

    return f"""## 9 · What we just observed

The cells above are live. Below: a quantitative + qualitative breakdown of the **actual** ReAct loop the Nebius-hosted Llama-3.3-70B agent produced on this run.

### 9.1 · Quantitative summary

| Metric | Value |
|---|---|
| Tool calls (Actions) | **{n_calls}** |
| Thoughts | **{n_thoughts}** |
| Agent rounds | {n_rounds} |
| Thought:Action ratio | {f"{n_thoughts / n_calls:.2f}" if n_calls else '—'} (target: 1.0 for clean ReAct) |
| Final answer length | {len(answer)} chars |

### 9.2 · Thought ↔ Action alignment

| # | Thought (truncated) | Action |
|---|---|---|
{pairs_table}

### 9.3 · Pathologies surfaced in this run

{obs_block}

### 9.4 · The final answer (verbatim)

{answer_block}

### 9.5 · The takeaway

When the Thought:Action ratio is **1.0** and queries are *different* round-to-round, ReAct is buying you genuine value over Tool Use — the thoughts are doing real work. When the ratio drops below 1.0 or thoughts are short and queries repeat, you're paying ReAct's extra-token cost for no benefit — switch back to Tool Use (notebook 02) or escalate to **Planning (nb 04)** which decomposes the task upfront."""


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
    print(
        f"tailored section 9: tool_calls={info['tool_calls']}, "
        f"thoughts={info['thought_count']}, queries={len(info['queries'])}"
    )


if __name__ == "__main__":
    main()
