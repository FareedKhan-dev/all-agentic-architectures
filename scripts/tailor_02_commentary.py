"""Post-process notebook 02: rewrite § 9 against the actual captured Tool-Use run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "02_tool_use.ipynb"
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
    """Pull the answer + tool-call summary out of the captured cells."""
    info: dict[str, object] = {
        "answer": "",
        "tool_calls": 0,
        "tools_used": [],
        "queries": [],
        "snippets": [],
    }
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "result = arch.run(TASK)" in cell.source:
            text = cell_output_text(cell)
            info["answer"] = _extract_after(text, "Final answer")
            m = re.search(r"(\d+)\s+tool call\(s\)\s+\W\s+(\d+)\s+final agent round", text)
            if m:
                info["tool_calls"] = int(m.group(1))
                info["agent_rounds"] = int(m.group(2))
            m = re.search(r"tools used:\s+([^\n─]+)", text)
            if m:
                raw = re.sub(r"[─\s]+$", "", m.group(1)).strip()
                info["tools_used"] = [
                    s.strip() for s in raw.split(",") if s.strip() and s.strip() != "none"
                ]
        if "for i, t in enumerate(result.trace" in cell.source:
            text = cell_output_text(cell)
            info["queries"] = re.findall(r"TOOL CALL\s+\W\s+\w+\s*`([^`]+)`", text)
            info["snippets"] = re.findall(r"TOOL RESULT \([^)]+\)\s*([^\n]+)", text)
    return info


def _extract_after(text: str, marker: str) -> str:
    """Return the substantive answer text following a header marker."""
    idx = text.find(marker)
    if idx < 0:
        return ""
    after = text[idx + len(marker):]
    # Drop the header rule line + leading whitespace.
    after = re.sub(r"^[\s─-▟]+", "", after)
    # Stop at the next big header
    stop = re.search(r"\n\s*─{5,}|\n\s*tool call\(", after)
    if stop:
        after = after[: stop.start()]
    return after.strip()


def make_commentary(info: dict[str, object]) -> str:
    n_calls = info.get("tool_calls", 0)
    queries: list[str] = info.get("queries", [])  # type: ignore[assignment]
    answer: str = info.get("answer", "")  # type: ignore[assignment]
    tools_used: list[str] = info.get("tools_used", [])  # type: ignore[assignment]

    def _esc_pipe(s: str) -> str:
        return s.replace("|", "\\|")

    queries_table = (
        "\n".join(f"| {i+1} | `{_esc_pipe(q)}` |" for i, q in enumerate(queries))
        if queries
        else "| — | _(no tool calls were made)_ |"
    )

    # Heuristics — detect interesting patterns in this specific run.
    obs: list[str] = []
    if n_calls == 0:
        obs.append(
            "**The agent never called a tool.** It answered straight from parametric "
            "knowledge. That's fine when the model already knows the answer, but it's a "
            "missed grounding opportunity for time-sensitive facts. If the question really "
            "required current data, the system prompt was too permissive — tighten with "
            "*'You MUST cite at least one source URL'*."
        )
    else:
        if n_calls >= 4:
            obs.append(
                f"**Over-search.** The agent made **{n_calls}** tool calls — more than "
                "necessary for this question. Even with the cap-search instruction in the "
                "system prompt, chatty models like Llama-3.3-70B reach for the search button "
                "repeatedly. To tighten further, lower `max_rounds` or add *'You may use at "
                "most 2 tool calls.'* to the system prompt."
            )
        if queries and len(set(queries)) < len(queries):
            dupes = len(queries) - len(set(queries))
            obs.append(
                f"**Repeated queries.** {dupes} of the {len(queries)} queries were "
                "duplicates. The agent has no memory that it already asked. This is a real "
                "limitation of Tool Use — ReAct's *thought* step partly fixes it because the "
                "model has to justify each search."
            )
        if queries and any(len(q.split()) <= 3 for q in queries):
            short = [q for q in queries if len(q.split()) <= 3]
            obs.append(
                "**Vague queries.** "
                f"`{short[0]}` is only {len(short[0].split())} word(s) long — too vague to "
                "narrow Tavily's index effectively. Forcing a thought step (ReAct, notebook "
                "03) usually improves query selection because the model has to commit to "
                "a hypothesis first."
            )
    if answer and "http" not in answer.lower():
        obs.append(
            "**No URLs in the final answer.** Even though the task explicitly asked for "
            "source citations, the model omitted them. This is *result drift* — the tool "
            "results were in context, but the model fell back to parametric knowledge for "
            "the final answer. Mitigation: ground the final-answer step with a separate "
            "Pydantic schema that has a required `sources: list[HttpUrl]` field."
        )
    # Hallucination check: model claimed a stale year as "most recent"
    if answer and re.search(r"\b(claude\s*2\.?0|gpt-?3\.5)\b.*(most recent|latest|current)", answer, re.IGNORECASE):
        obs.append(
            "**Result drift / hallucination.** The model named an *old* model as the "
            "most recent — even though the Tavily results in earlier rounds presumably "
            "contained newer information. The agent searched, then *ignored its own "
            "evidence* and reverted to parametric knowledge. This is the most dangerous "
            "Tool-Use failure mode because the answer *looks confident*. Self-RAG (nb 25) "
            "and Corrective RAG (nb 24) exist specifically to plug this hole."
        )

    obs_block = (
        "\n\n### 9.2 · Pathologies surfaced in this run\n\n"
        + "\n\n".join(f"- {o}" for o in obs)
        if obs
        else "\n\n### 9.2 · No common pathologies surfaced — the agent stayed focused.\n"
    )

    answer_block = (
        f"\n\n### 9.3 · The final answer (verbatim)\n\n> "
        + (answer[:500].replace("\n", "\n> ") if answer else "_(no answer captured)_")
        + ("…" if len(answer) > 500 else "")
    )

    return f"""## 9 · What we just observed

The cells above are live. Below: a quantitative breakdown of the **actual** tool-call sequence the Nebius-hosted Llama-3.3-70B agent produced on this run.

### 9.1 · Quantitative summary

| Metric | Value |
|---|---|
| Tool calls made | **{n_calls}** |
| Tools used | {', '.join(tools_used) if tools_used else '_none_'} |
| Final agent rounds | {info.get('agent_rounds', '?')} |
| Final answer length (chars) | {len(answer)} |

**Queries the agent issued to Tavily:**

| # | Query |
|---|---|
{queries_table}
{obs_block}
{answer_block}

### 9.4 · The takeaway

Tool Use is the **right** pattern when the model needs one or two facts from outside its training data. It's the **wrong** pattern when you need:

- *Multi-step reasoning* between calls → use **ReAct (nb 03)**.
- *Guaranteed grounding* of the final answer → use **Self-RAG (nb 25)** or **Corrective RAG (nb 24)**.
- *Recovery* from failed tool calls → use **PEV (nb 06)**.

The pathologies you saw above are not bugs in the implementation — they're inherent to the act-only loop. They motivate the next several notebooks."""


def main() -> None:
    nb = nbformat.read(NB_PATH, as_version=4)
    info = extract_run(nb)
    if not info["queries"] and info["tool_calls"] == 0:
        print("warning: no tool-use trace found; leaving section 9 untouched")
        return

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
        raise RuntimeError("section 9 markdown cell not found")

    nbformat.write(nb, NB_PATH)
    print(
        f"tailored section 9 with {info['tool_calls']} tool call(s), "
        f"{len(info['queries'])} unique queries"
    )


if __name__ == "__main__":
    main()
