"""Post-process notebook 05: rewrite § 9 against the Multi-Agent captured run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "05_multi_agent.ipynb"
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
    info: dict[str, object] = {
        "report": "",
        "invoked": 0,
        "available": 0,
        "specialists": [],
        "snippets": [],
    }
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "result = arch.run(TASK)" in cell.source:
            text = cell_output_text(cell)
            m = re.search(
                r"Final report[\s\S─-╿]*?\n(.+?)\n[\s─-]*\n[\s─-╿]*?\d+/\d+",
                text,
                re.DOTALL,
            )
            if m:
                info["report"] = re.sub(r"[\s─-╿]+$", "", m.group(1)).strip()
            m = re.search(r"(\d+)/(\d+)\s+specialists\s+contributed", text)
            if m:
                info["invoked"] = int(m.group(1))
                info["available"] = int(m.group(2))
        if "for i, t in enumerate(result.trace" in cell.source:
            text = cell_output_text(cell)
            specialists = re.findall(
                r"SPECIALIST:\s+(\w+)\s*\n(.+?)(?=\n\s*[›>]\s+|\Z)",
                text,
                re.DOTALL,
            )
            info["specialists"] = [name.lower() for name, _ in specialists]
            info["snippets"] = [_normalize_ws(content) for _, content in specialists]
    return info


def make_commentary(info: dict[str, object]) -> str:
    invoked = info.get("invoked", 0)
    available = info.get("available", 0)
    specialists: list[str] = info.get("specialists", [])  # type: ignore[assignment]
    snippets: list[str] = info.get("snippets", [])  # type: ignore[assignment]
    report: str = info.get("report", "")  # type: ignore[assignment]

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ").strip()

    spec_table = (
        "\n".join(
            f"| {i+1} | {name} | {esc(snip)[:200]}{'…' if len(snip) > 200 else ''} |"
            for i, (name, snip) in enumerate(zip(specialists, snippets))
        )
        if specialists
        else "| — | _(no specialists invoked)_ | — |"
    )

    # Heuristic pathology detection
    obs: list[str] = []
    if invoked == 0:
        obs.append(
            "**Supervisor finalised before any specialist ran.** Complete coordination "
            "failure — the supervisor judged no specialist was relevant. Inspect the "
            "supervisor decision logs in LangSmith to see why."
        )
    elif invoked < available:
        obs.append(
            f"**Incomplete team participation.** Only {invoked}/{available} specialists "
            "contributed. The supervisor routed to the Writer before everyone had a turn. "
            "Mitigation: tighten the supervisor prompt to require every specialist contributes."
        )
    else:
        obs.append(
            f"**Full team coverage.** All {available} specialists contributed once each — "
            "the supervisor protocol worked as designed."
        )

    # Role-drift detection: cross-check each specialist's content for "wrong-domain" keywords
    role_keywords = {
        "news": [r"\bannounce", r"\brelease", r"\blaunch", r"\bunveil"],
        "technical": [r"\bmodel\b", r"\bproduct\b", r"\bspec", r"\btech"],
        "financial": [r"\brevenue", r"\bmargin", r"\bearnings", r"\bstock", r"\bmarket cap"],
    }
    for name, snip in zip(specialists, snippets):
        my_kw = role_keywords.get(name, [])
        if not my_kw or not snip:
            continue
        own = sum(1 for kw in my_kw if re.search(kw, snip, re.IGNORECASE))
        other_kw = [
            kw for n, kws in role_keywords.items() if n != name for kw in kws
        ]
        other = sum(1 for kw in other_kw if re.search(kw, snip, re.IGNORECASE))
        if other > 2 * max(own, 1):
            obs.append(
                f"**Role drift on `{name}`.** Its output contains more "
                f"sibling-domain keywords ({other}) than its own ({own}). "
                f"The {name} specialist is generalising outside its lane — "
                "tighten its role prompt or use an LLMJudge to penalise off-role outputs."
            )

    # URL coverage in final report
    if report and "http" not in report.lower():
        obs.append(
            "**Writer dropped all URLs.** The specialists provided sources but the "
            "Writer's synthesis omitted them. Add an explicit `MUST preserve all URLs` "
            "rule to the Writer's prompt."
        )
    elif report:
        url_count = len(re.findall(r"https?://", report))
        if url_count < invoked:
            obs.append(
                f"**Writer dropped some URLs** (kept {url_count}, expected ~{invoked}+). "
                "Some specialists' citations didn't make it into the synthesis."
            )

    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No pathologies surfaced — clean coordination + grounded report."

    report_block = (
        "> " + (report[:600].replace("\n", "\n> ") if report else "_(no report captured)_")
        + ("…" if len(report) > 600 else "")
    )

    return f"""## 9 · What we just observed

The cells above are live. Below: a breakdown of the **actual** Multi-Agent coordination Nebius-hosted Llama-3.3-70B produced on this run.

### 9.1 · Quantitative summary

| Metric | Value |
|---|---|
| Specialists invoked | **{invoked}** / {available} |
| Routing order | {' → '.join(specialists) if specialists else '—'} → writer |
| Final report length | {len(report)} chars |
| URLs preserved in report | {len(re.findall(r'https?://', report))} |

### 9.2 · Specialist contributions

| # | Role | Output snippet |
|---|---|---|
{spec_table}

### 9.3 · Pathologies / patterns surfaced in this run

{obs_block}

### 9.4 · The final report (verbatim)

{report_block}

### 9.5 · The takeaway

Multi-Agent wins when *specialisation* genuinely narrows each agent's job. The signs of a healthy run: **full team coverage** (all specialists contributed), **no role drift** (each output stays in its lane), **all URLs preserved** by the Writer. When any of those fail, the cost of coordination isn't being earned back by quality."""


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
        f"tailored section 9: {info['invoked']}/{info['available']} specialists, "
        f"{len(info['specialists'])} captured"
    )


if __name__ == "__main__":
    main()
