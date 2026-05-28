"""Post-process notebook 19: rewrite § 9 against the Self-Discover captured run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "19_self_discover.ipynb"
ANSI = re.compile(r"\x1b\[[0-9;]*[mGKH]")


def cell_output_text(cell: nbformat.NotebookNode) -> str:
    chunks: list[str] = []
    for o in cell.outputs:
        t = o.get("text", "") or o.get("data", {}).get("text/plain", "")
        if isinstance(t, list):
            t = "".join(t)
        chunks.append(ANSI.sub("", str(t)))
    return "\n".join(chunks)


PRIMARY = re.compile(
    r"SELECTED_IDS:\s+\[([\d, ]*)\]\s*\n"
    r"SELECTED_COUNT:\s+(\d+)\s*\n"
    r"((?:\s+module:.+\n)+)"
    r"\n"
    r"PLAN_STEP_COUNT:\s+(\d+)\s*\n"
    r"((?:\s+step\s+\d+:.+\n\s+→ expected_output:.+\n)+)"
    r"\n"
    r"FINAL_ANSWER_FORMAT:\s+(.+?)\n"
    r"FINAL_ANSWER:\s+(.+?)\n"
    r"EXPECTED:\s+(.+?)\n"
    r"MATCH:\s+(True|False)"
)
LLAMA = re.compile(
    r"LLAMA_SELECTED_COUNT:\s+(\d+)\s*\n"
    r"LLAMA_PLAN_STEP_COUNT:\s+(\d+)\s*\n"
    r"LLAMA_FINAL_ANSWER:\s+(.+?)\n"
    r"LLAMA_MATCH:\s+(True|False)"
)


def extract_run(nb: nbformat.NotebookNode) -> dict[str, object]:
    info: dict[str, object] = {"primary": None, "llama": None}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        text = cell_output_text(cell)
        if "SELECTED_IDS:" in cell.source and "MATCH" in cell.source:
            m = PRIMARY.search(text)
            if m:
                ids_str, count, modules_block, plan_count, plan_block, fmt, answer, expected, match = m.groups()
                ids = [int(x) for x in ids_str.split(",") if x.strip()]
                modules = [
                    line.strip().removeprefix("module:").strip()
                    for line in modules_block.strip().splitlines()
                ]
                step_lines = plan_block.strip().splitlines()
                steps = []
                # Pair each "step N: desc" with its "→ expected_output: ..." follow-up
                for i in range(0, len(step_lines), 2):
                    desc_line = step_lines[i].strip()
                    out_line = step_lines[i + 1].strip() if i + 1 < len(step_lines) else ""
                    n_match = re.match(r"step\s+(\d+):\s*(.+)", desc_line)
                    if n_match:
                        steps.append({
                            "n": int(n_match.group(1)),
                            "description": n_match.group(2),
                            "expected_output": out_line.removeprefix("→ expected_output:").strip(),
                        })
                info["primary"] = {
                    "selected_ids": ids,
                    "selected_count": int(count),
                    "selected_modules": modules,
                    "plan_step_count": int(plan_count),
                    "plan_steps": steps,
                    "final_answer_format": fmt.strip(),
                    "final_answer": answer.strip(),
                    "expected": expected.strip(),
                    "match": match == "True",
                }
        if "LLAMA_SELECTED_COUNT" in cell.source:
            m = LLAMA.search(text)
            if m:
                info["llama"] = {
                    "selected_count": int(m.group(1)),
                    "plan_step_count": int(m.group(2)),
                    "final_answer": m.group(3).strip(),
                    "match": m.group(4) == "True",
                }
    return info


def _esc(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def make_commentary(info: dict[str, object]) -> str:
    primary = info.get("primary") or {}
    llama = info.get("llama") or {}

    # ---- 9.1 stage-by-stage summary ----
    if primary:
        p = primary  # type: ignore[assignment]
        modules_rows = "\n".join(
            f"| `[{i}]` | {_esc(m)} |"
            for i, m in zip(p["selected_ids"], p["selected_modules"])  # type: ignore[index]
        )
        plan_rows = "\n".join(
            f"| {s['n']} | {_esc(s['description'])} | {_esc(s['expected_output'])} |"
            for s in p["plan_steps"]  # type: ignore[index]
        )
        summary_table = (
            f"| SELECTED modules | {p['selected_count']} of 16 |\n"
            f"| Module ids | `{p['selected_ids']}` |\n"
            f"| PLAN steps | {p['plan_step_count']} |\n"
            f"| Final-answer format | {_esc(p['final_answer_format'])} |\n"
            f"| Final answer | `{_esc(p['final_answer'])}` |\n"
            f"| Expected | `{_esc(p['expected'])}` |\n"
            f"| Match | {'✅' if p['match'] else '❌'} |"
        )
    else:
        modules_rows = "| — | _(no run captured)_ |"
        plan_rows = "| — | — | — |"
        summary_table = "| — | _(no run captured)_ |"

    # ---- 9.2 reasoning-vs-plain comparison ----
    if primary and llama:
        comp_table = (
            f"| Reasoning (Qwen3-Thinking) | {primary['selected_count']} | {primary['plan_step_count']} | `{_esc(primary['final_answer'])}` | {'✅' if primary['match'] else '❌'} |\n"
            f"| Plain (Llama-3.3-70B)      | {llama['selected_count']} | {llama['plan_step_count']} | `{_esc(llama['final_answer'])}` | {'✅' if llama['match'] else '❌'} |"
        )
    else:
        comp_table = "| — | — | — | — | — |"

    # ---- 9.3 auto-flags ----
    obs: list[str] = []
    if primary:
        sc = primary["selected_count"]  # type: ignore[index]
        if sc < 3:
            obs.append(f"**⚠️  SELECT picked only {sc} module(s)** — below the suggested 3-6 range. Plan may be too narrow.")
        elif sc > 6:
            obs.append(f"**⚠️  SELECT picked {sc} modules** — above the suggested 3-6 range. Plan may become kitchen-sink.")
        else:
            obs.append(f"**✅ SELECT picked {sc} modules** — inside the recommended 3-6 range.")
        psc = primary["plan_step_count"]  # type: ignore[index]
        if psc < 3:
            obs.append(f"**⚠️  IMPLEMENT produced {psc}-step plan** — under-decomposed; modules collapsed.")
        elif psc > 7:
            obs.append(f"**⚠️  IMPLEMENT produced {psc}-step plan** — over-decomposed; likely redundant steps.")
        if primary["match"]:  # type: ignore[index]
            obs.append("**✅ SOLVE produced the expected answer** — the discovered structure executed correctly.")
        else:
            obs.append(
                f"**❌ SOLVE missed the expected answer.** Got `{_esc(primary['final_answer'])}` vs expected "  # type: ignore[index]
                f"`{_esc(primary['expected'])}`. Check the step outputs (§ 8.1) to see where the deduction broke."
            )

    if primary and llama:
        if primary["match"] and llama["match"]:  # type: ignore[index]
            obs.append(
                "**🟰 Both reasoning and plain LLM converged on the correct answer.** On this puzzle, "
                "the structure-discovery pipeline produced sensible recipes for both — the underlying "
                "model's reasoning depth wasn't the bottleneck. Bigger differentiation would show on "
                "harder logic-deduction tasks (e.g., 7-element ordering with multi-hop transitivity)."
            )
        elif primary["match"] and not llama["match"]:  # type: ignore[index]
            obs.append(
                f"**✅ Reasoning model matched; plain Llama got `{_esc(llama['final_answer'])}` (wrong).** "  # type: ignore[index]
                "Either Llama's SELECT picked a weaker module set or its SOLVE drifted off the plan. "
                "This is the case for using Qwen-Thinking on the SELECT/IMPLEMENT stages even if you "
                "delegate SOLVE to a cheaper LLM."
            )
        elif not primary["match"] and llama["match"]:  # type: ignore[index]
            obs.append(
                f"**🤔 Plain Llama matched but Qwen-Thinking got `{_esc(primary['final_answer'])}` (wrong).** "  # type: ignore[index]
                "Unusual. Re-run to check it isn't sampling variance; if reproducible, examine Qwen's "
                "plan vs Llama's for what diverged."
            )

        # Structure-divergence note
        if primary["selected_count"] != llama["selected_count"] or primary["plan_step_count"] != llama["plan_step_count"]:  # type: ignore[index]
            obs.append(
                "**Structure divergence**: the two models chose different module counts / plan lengths "
                f"({primary['selected_count']}/{primary['plan_step_count']} vs "  # type: ignore[index]
                f"{llama['selected_count']}/{llama['plan_step_count']}). Self-Discover's recipe is "  # type: ignore[index]
                "LLM-specific, not task-specific — same task, different model = different reasoning shape."
            )

    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns surfaced."

    return f"""## 9 · What we just observed

The cells above ran the SELECT → ADAPT → IMPLEMENT → SOLVE pipeline on a 5-musician height-ordering puzzle, then re-ran the same puzzle with a plain (non-reasoning) LLM for contrast.

### 9.1 · The recipe the reasoning model designed

**Stage 1 — SELECT** picked these modules from the 16-module library:

| id | module |
|---|---|
{modules_rows}

**Stage 3 — IMPLEMENT** composed them into this plan:

| # | description | expected output |
|---|---|---|
{plan_rows}

**Run summary:**

| Field | Value |
|---|---|
{summary_table}

### 9.2 · Reasoning vs non-reasoning LLM on the same task

| Model | SELECT count | PLAN steps | Final answer | Match |
|---|---|---|---|---|
{comp_table}

### 9.3 · Patterns surfaced in this run

{obs_block}

### 9.4 · The takeaway

Self-Discover converts the implicit "how should I think about this?" choice into an *explicit, inspectable* artefact (the plan in § 9.1). Two consequences:

1. **Plans are auditable.** Unlike chain-of-thought scratchpad, the recipe is structured Pydantic data — you can save it, diff it across model versions, or hand-edit it before SOLVE runs.
2. **Plans are reusable.** Tasks of the same type share a recipe; SELECT/ADAPT/IMPLEMENT only need to run once per task family. § 11.3 extension #1 sketches a `discover() + solve()` split that makes this concrete and cuts cost 4×.

The architecture's headline behaviour is that the *reasoning structure* itself becomes a first-class object the agent decides on, rather than an emergent property of a fixed CoT prompt. Whether that produces a better *answer* depends on the underlying model (§ 9.2 above) — but it always produces a more *transparent* one."""


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
    p = info["primary"] or {}
    l = info["llama"] or {}
    print(
        f"tailored section 9: primary modules={p.get('selected_count')} steps={p.get('plan_step_count')} "
        f"match={p.get('match')}; llama modules={l.get('selected_count')} steps={l.get('plan_step_count')} "
        f"match={l.get('match')}"
    )


if __name__ == "__main__":
    main()
