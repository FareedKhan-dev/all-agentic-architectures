"""Post-process notebook 07: rewrite § 9 against the Blackboard captured run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "07_blackboard.ipynb"
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
        "synthesis": "",
        "rounds": 0,
        "max_rounds": 0,
        "contributed": 0,
        "available": 0,
        "counts": {},
        "contributions": [],
    }
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "result = arch.run(TASK)" in cell.source:
            text = cell_output_text(cell)
            m = re.search(
                r"Final synthesis[\s\S─-╿]*?\n(.+?)\n[\s─-]*\n[\s─-╿]*?rounds:",
                text,
                re.DOTALL,
            )
            if m:
                info["synthesis"] = re.sub(r"[\s─-╿]+$", "", m.group(1)).strip()
            m = re.search(r"rounds:\s+(\d+)/(\d+).*?distinct agents contributed:\s+(\d+)/(\d+)", text, re.DOTALL)
            if m:
                info["rounds"] = int(m.group(1))
                info["max_rounds"] = int(m.group(2))
                info["contributed"] = int(m.group(3))
                info["available"] = int(m.group(4))
        if "for agent, n in sorted(counts" in cell.source:
            text = cell_output_text(cell)
            counts = {}
            for line in text.splitlines():
                m = re.match(r"\s*(\w+):\s*█+\s*(\d+)\s*$", line)
                if m:
                    counts[m.group(1)] = int(m.group(2))
            info["counts"] = counts
            # Round-by-round
            contribs = re.findall(
                r"\[\d+\]\s+round\s+(\d+)\s+→\s+(\w+)\s*\n(.+?)(?=\n\s*[›>]\s+|\Z)",
                text,
                re.DOTALL,
            )
            info["contributions"] = [
                {"round": int(r), "agent": a.lower(), "content": _normalize_ws(c)}
                for r, a, c in contribs
            ]
    return info


def make_commentary(info: dict[str, object]) -> str:
    rounds = info.get("rounds", 0)
    max_r = info.get("max_rounds", 0)
    contributed = info.get("contributed", 0)
    available = info.get("available", 0)
    counts: dict[str, int] = info.get("counts", {})  # type: ignore[assignment]
    contribs: list[dict] = info.get("contributions", [])  # type: ignore[assignment]
    synthesis: str = info.get("synthesis", "")  # type: ignore[assignment]

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ").strip()

    contrib_table = (
        "\n".join(
            f"| {i+1} | {c['round']} | {c['agent']} | {esc(c['content'])[:180]}{'…' if len(c['content']) > 180 else ''} |"
            for i, c in enumerate(contribs)
        )
        if contribs
        else "| — | — | — | _(no contributions captured)_ |"
    )

    obs: list[str] = []
    fairness = contributed / max(available, 1)
    if fairness == 1.0:
        obs.append(
            f"**Full team participation** — all {available} agents contributed at "
            "least once. The fairness signal in the bid prompt (§ 3.4) successfully "
            "prevented domination on this run."
        )
    elif fairness >= 0.5:
        obs.append(
            f"**Partial team participation** — {contributed}/{available} agents "
            "contributed. Some roles didn't bid above `min_confidence`. Inspect the "
            "agent-counts chart above to see which role(s) were silent. Could mean "
            "they genuinely had nothing to add, OR that their bid LLM was too cautious."
        )
    else:
        obs.append(
            f"**Domination pathology** — only {contributed}/{available} agents "
            "contributed. The bidding mechanism failed to distribute turns. Consider "
            "lowering `min_confidence`, or use a HARD quota (`max_per_agent`)."
        )

    # Detect domination by single agent
    if counts:
        max_agent = max(counts.items(), key=lambda x: x[1])
        if max_agent[1] / rounds >= 0.6 and rounds >= 3:
            obs.append(
                f"**`{max_agent[0]}` dominated** — won {max_agent[1]}/{rounds} rounds "
                f"({max_agent[1]/rounds*100:.0f}%). The fairness nudge helped but "
                "didn't fully prevent over-bidding. A hard quota would."
            )

    if rounds < max_r:
        obs.append(
            f"**Early convergence** — stopped at round {rounds} (budget was {max_r}). "
            "Every agent bid `will_contribute=False` or below `min_confidence`. "
            "This is the *desired* termination signal: the agents collectively "
            "decided there was nothing more to add."
        )
    elif rounds == max_r:
        obs.append(
            f"**Budget exhausted** at round {max_r}. Agents would have kept "
            "bidding — synthesis is forced. Inspect the last few contributions: "
            "if they're substantive, raise `max_rounds`; if they're filler, "
            "raise `min_confidence`."
        )

    obs_block = "\n\n".join(f"- {o}" for o in obs)

    synthesis_block = (
        "> " + (synthesis[:600].replace("\n", "\n> ") if synthesis else "_(no synthesis captured)_")
        + ("…" if len(synthesis) > 600 else "")
    )

    return f"""## 9 · What we just observed

The cells above are live. Below: a breakdown of the **actual** bidding pattern + contributions Nebius-hosted Llama-3.3-70B produced on this run.

### 9.1 · Quantitative summary

| Metric | Value |
|---|---|
| Total rounds | **{rounds}** / {max_r} |
| Distinct agents who contributed | **{contributed}** / {available} ({fairness*100:.0f}%) |
| Synthesis length | {len(synthesis)} chars |
| Invocation counts | {counts or '—'} |

### 9.2 · Contribution sequence

| # | Round | Agent | Contribution snippet |
|---|---|---|---|
{contrib_table}

### 9.3 · Patterns surfaced in this run

{obs_block}

### 9.4 · The final synthesis (verbatim)

{synthesis_block}

### 9.5 · The takeaway

A *healthy* Blackboard run looks like:

1. **Full or near-full team participation** (every role contributes at least once).
2. **No single agent winning >50% of rounds** (no domination).
3. **Early convergence** rather than budget exhaustion (agents agree they're done).
4. **Synthesis preserves the multi-perspective nature** — minority views still visible.

Compare those four signals to what you see above to judge the run."""


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
        f"tailored section 9: {info['rounds']} rounds, "
        f"{info['contributed']}/{info['available']} agents, "
        f"{len(info['contributions'])} captured contributions"
    )


if __name__ == "__main__":
    main()
