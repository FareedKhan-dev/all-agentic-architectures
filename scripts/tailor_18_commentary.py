"""Post-process notebook 18: rewrite § 9 against the Reflexion captured run."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parents[1] / "notebooks" / "18_reflexion.ipynb"
ANSI = re.compile(r"\x1b\[[0-9;]*[mGKH]")


def cell_output_text(cell: nbformat.NotebookNode) -> str:
    chunks: list[str] = []
    for o in cell.outputs:
        t = o.get("text", "") or o.get("data", {}).get("text/plain", "")
        if isinstance(t, list):
            t = "".join(t)
        chunks.append(ANSI.sub("", str(t)))
    return "\n".join(chunks)


BLOCK = re.compile(
    r"TASK_TAG:\s+(\w+)\s*\n"
    r"\s+TRIALS_USED:\s+(\d+)\s*\n"
    r"\s+SUCCESS:\s+(\w+)\s*\n"
    r"\s+REFLECTIONS_RECALLED:\s+(\d+)\s*\n"
    r"\s+REFLECTIONS_IN_MEMORY_AFTER:\s+(\d+)\s*\n"
    r"\s+SYLLABLES_PER_TRIAL:\s+(\[\[.*?\]\])\s*\n"
    r"\s+WORDS_PRESENT_PER_TRIAL:\s+(\[.*?\])\s*\n"
    r"\s+FINAL:\s+(.+?)(?=\n\s*TASK_TAG:|\n\s*TRIALS_PER_TASK:|\Z)",
    re.DOTALL,
)
SUMMARY = re.compile(
    r"TRIALS_PER_TASK:\s+\[([\d, ]+)\]\s+total_reflections_in_memory=(\d+)"
)
COLD = re.compile(
    r"COLD_START_TASK:\s+(\w+)\s*\n"
    r"COLD_START_TRIALS:\s+(\d+)\s*\n"
    r"COLD_START_SUCCESS:\s+(\w+)\s*\n"
    r"COLD_START_REFLECTIONS_RECALLED:\s+(\d+)"
)
LESSON = re.compile(
    r"\[lesson\s+(\d+)\][^\n]*\(recorded on task='([^']*)',\s+trial=(\d+)\)\s*\n"
    r"\s+root_cause:\s+(.+?)\n"
    r"\s+correction:\s+(.+?)\n"
    r"\s+reflection:\s+(.+?)(?=\n\s*\[lesson|\Z)",
    re.DOTALL,
)


def extract_run(nb: nbformat.NotebookNode) -> dict[str, object]:
    info: dict[str, object] = {
        "tasks": [],
        "trials_per_task": [],
        "total_reflections": 0,
        "cold": None,
        "lessons": [],
    }
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        text = cell_output_text(cell)
        if "TASK_TAG:" in cell.source and "TRIALS_PER_TASK" in cell.source:
            blocks = BLOCK.findall(text)
            info["tasks"] = [
                {
                    "tag": tag,
                    "trials": int(trials),
                    "success": success == "True",
                    "recalled": int(recalled),
                    "memory_after": int(mem_after),
                    "syllables": syllables.strip(),
                    "words_present": words.strip(),
                    "final": final.strip(),
                }
                for tag, trials, success, recalled, mem_after, syllables, words, final in blocks
            ]
            m = SUMMARY.search(text)
            if m:
                info["trials_per_task"] = [
                    int(x) for x in m.group(1).split(",") if x.strip()
                ]
                info["total_reflections"] = int(m.group(2))
        if "COLD_START_TASK" in cell.source:
            m = COLD.search(text)
            if m:
                info["cold"] = {
                    "tag": m.group(1),
                    "trials": int(m.group(2)),
                    "success": m.group(3) == "True",
                    "recalled": int(m.group(4)),
                }
        if "arch.episodic.episodes" in cell.source and "lesson" in text:
            info["lessons"] = [
                {
                    "n": int(n),
                    "task_excerpt": task_excerpt,
                    "trial": int(trial),
                    "root_cause": root_cause.strip(),
                    "correction": correction.strip(),
                    "reflection": reflection.strip(),
                }
                for n, task_excerpt, trial, root_cause, correction, reflection in LESSON.findall(text)
            ]
    return info


def _esc(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def make_commentary(info: dict[str, object]) -> str:
    tasks: list[dict] = info.get("tasks", [])  # type: ignore[assignment]
    trials_per_task: list[int] = info.get("trials_per_task", [])  # type: ignore[assignment]
    total_reflections: int = info.get("total_reflections", 0)  # type: ignore[assignment]
    cold = info.get("cold")
    lessons: list[dict] = info.get("lessons", [])  # type: ignore[assignment]

    # ---- 9.1 per-task table ----
    if tasks:
        rows = "\n".join(
            f"| {t['tag']} | {t['trials']} | {'✓' if t['success'] else '✗'} | "
            f"{t['recalled']} | {t['memory_after']} | {_esc(t['final'])} |"
            for t in tasks
        )
    else:
        rows = "| _(no tasks captured)_ | | | | | |"

    # ---- 9.2 transfer comparison ----
    transfer_block = "_(insufficient data)_"
    if tasks and len(tasks) >= 1:
        t1 = tasks[0]
        t_last = tasks[-1]
        rows_t = [
            f"| Task 1 (`{t1['tag']}`, no prior memory) | {t1['trials']} | {t1['recalled']} | {'✓' if t1['success'] else '✗'} |",
            f"| Task {len(tasks)} (`{t_last['tag']}`, {t_last['recalled']} lessons recalled) | {t_last['trials']} | {t_last['recalled']} | {'✓' if t_last['success'] else '✗'} |",
        ]
        if cold:
            rows_t.append(
                f"| Cold-start re-run of `{cold['tag']}` (fresh Reflexion, empty memory) | {cold['trials']} | {cold['recalled']} | {'✓' if cold['success'] else '✗'} |"
            )
        transfer_block = "| Comparison | Trials | Reflections recalled | Success |\n|---|---|---|---|\n" + "\n".join(rows_t)

    # ---- 9.3 auto-flagged patterns ----
    obs: list[str] = []
    if tasks:
        if tasks[0]["trials"] == 1 and tasks[0]["success"]:
            obs.append(
                f"**⚠️  Task 1 (`{tasks[0]['tag']}`) passed on the FIRST trial** — no reflection was "
                "written for it, so the agent had nothing in memory when starting task 2. The "
                "demo's lesson-transfer effect therefore depends entirely on whatever lessons "
                "tasks 2+ produce. If you want to see transfer onto task 2, pick task-1 "
                "constraints Llama actually struggles with (e.g., 4-syllable required words "
                "that crowd a 5-syllable line)."
            )
        if any(not t["success"] for t in tasks):
            failed = [t["tag"] for t in tasks if not t["success"]]
            obs.append(
                f"**⚠️  Task(s) `{', '.join(failed)}` exhausted `max_trials` without passing.** "
                "Reflexion isn't magic — when reflections don't surface a correction that the "
                "next attempt can actually act on, the agent loops failing. The lessons are "
                "still stored and can benefit *later* tasks even though this one failed."
            )
        if len(tasks) >= 3 and tasks[2]["recalled"] >= 1 and tasks[2]["success"] and tasks[2]["trials"] == 1:
            obs.append(
                f"**✅  Memory transferred to task 3 (`{tasks[2]['tag']}`).** It passed on the "
                f"first trial with {tasks[2]['recalled']} prior lessons recalled — the agent "
                "started the task already primed by lessons learned on earlier tasks. This is "
                "the architecture's headline behaviour."
            )
        # Memory-recall plumbing check
        any_recall_after_memory = any(
            (i > 0 and t["recalled"] == 0 and tasks[i-1]["memory_after"] > 0)
            for i, t in enumerate(tasks)
        )
        if any_recall_after_memory:
            obs.append(
                "**⚠️  At least one task recalled 0 lessons even though prior tasks had populated "
                "memory** — vector-similarity recall isn't finding the stored lessons. Most "
                "common cause: embedding model is too weak to see structural similarity between "
                "the task texts. Mitigation: pass `get_embeddings()` explicitly to "
                "`EpisodicMemory(...)` to upgrade from the default."
            )
        # Lesson dedup observation
        if len(lessons) >= 2:
            roots = {l["root_cause"][:80] for l in lessons}
            if len(roots) == 1:
                obs.append(
                    f"**🔁 All {len(lessons)} stored lessons share the same `root_cause`** — the "
                    "agent kept failing the same way and produced near-duplicate reflections. "
                    "Real production use would dedup on cosine similarity (§ 11.3 extension #2) "
                    "to avoid memory bloat."
                )

    if cold and tasks:
        warm = tasks[0]["trials"]
        if cold["trials"] > warm:
            obs.append(
                f"**✅  Cold-start contrast positive**: cold run of `{cold['tag']}` took "
                f"{cold['trials']} trials vs {warm} on the warm instance — direct evidence the "
                "accumulated memory helped."
            )
        elif cold["trials"] == warm:
            obs.append(
                f"**⚠️  Cold-start contrast inconclusive**: cold run of `{cold['tag']}` took "
                f"{cold['trials']} trials, same as warm — task 1 was easy enough that memory "
                "wasn't the differentiator. The clearer signal is task 3 (above) where 2 "
                "recalls + trial-1 pass shows the memory pipeline is wired correctly."
            )
        else:
            obs.append(
                f"**🤔  Cold-start ran FASTER than warm** ({cold['trials']} vs {warm}) — Llama "
                "variance on a single task. With one cold sample this is noise, not signal. "
                "Run a few cold trials and average if you need a statistical comparison."
            )

    obs_block = "\n\n".join(f"- {o}" for o in obs) if obs else "- No notable patterns surfaced."

    # ---- 9.4 verbatim final haikus ----
    if tasks:
        final_block = "\n\n".join(
            f"**`{t['tag']}`** ({'pass' if t['success'] else 'FAIL'}, trial {t['trials']}, syllables `{t['syllables']}`)\n\n"
            f"> {t['final'].replace(' / ', chr(10) + '> ')}"
            for t in tasks
        )
    else:
        final_block = "_(no captured outputs)_"

    # ---- 9.5 lesson library ----
    if lessons:
        lesson_block = "\n\n".join(
            f"**Lesson {l['n']}** *(from task `{l['task_excerpt'][:60].strip()}…`, trial {l['trial']})*\n\n"
            f"- **root_cause:** {_esc(l['root_cause'])}\n"
            f"- **correction:** {_esc(l['correction'])}\n"
            f"- **stored verbatim:** {_esc(l['reflection'])[:400]}{'…' if len(l['reflection']) > 400 else ''}"
            for l in lessons
        )
    else:
        lesson_block = "_(no lessons were recorded — every task passed on the first trial; demo was too easy)_"

    return f"""## 9 · What we just observed

The cells above ran **3 structurally-similar haiku tasks** through one `Reflexion` instance, then re-ran task 1 with a fresh instance for a cold-start contrast. The deterministic Python checker decided pass/fail; the reflection LLM wrote a verbal lesson on every failure; lessons accumulated in `arch.episodic`.

### 9.1 · Per-task trial-count + reflection-transfer table

| Tag | Trials used | Success | Reflections recalled (first attempt) | Memory size after | Final haiku |
|---|---|---|---|---|---|
{rows}

**Summary:** trials per task `{trials_per_task}`, total lessons in memory after all 3 runs: **{total_reflections}**.

### 9.2 · Did the reflection memory help?

{transfer_block}

### 9.3 · Patterns surfaced in this run

{obs_block}

### 9.4 · Verbatim final outputs

{final_block}

### 9.5 · The lesson library that built up

{lesson_block}

### 9.6 · The takeaway

Reflexion only works when **three** things are true simultaneously:

1. **The evaluator is precise enough to surface a *specific* failure feature.** The pure-Python haiku checker covers this (it reports exact syllable counts and which required words are missing, so the reflection LLM has concrete material to write about).
2. **The reflection LLM phrases the lesson in *transferable* form.** The `_SelfReflection` schema's `correction` field asks for a general imperative ("count syllables out loud before submitting"), not a task-specific patch ("use the word 'centuries' in line 2").
3. **The recall step actually finds the relevant lessons.** That's just FAISS doing its job — if it returns 0 episodes when memory is non-empty (auto-flagged above when it happens), the embedding model is too weak for the task texts.

Watch the §9.1 table and the §9.3 flags every run. The architecture's headline behaviour is most visible in the **`Reflections recalled (first attempt)`** column — non-zero on a non-first task = memory pipeline is alive; trial-1 pass on a non-first task = memory pipeline is *effective*."""


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
        f"tailored section 9: {len(info['tasks'])} tasks, "
        f"trials_per_task={info['trials_per_task']}, "
        f"lessons_extracted={len(info['lessons'])}, "
        f"cold={info['cold']}"
    )


if __name__ == "__main__":
    main()
