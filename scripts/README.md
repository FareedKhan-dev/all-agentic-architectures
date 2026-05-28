# scripts/

Maintenance and reproduction tooling. **End users don't need to read or run anything in here** — `pip install agentic-architectures`, import a class, and you're done. This folder is for maintainers and contributors.

## What lives here

| File pattern | Count | Purpose |
|---|---|---|
| `notebook_builder.py` | 1 | Tiny helper (`md`, `code`, `build_notebook`) used by every `build_NN_*.py` to emit a `.ipynb` from a list of cells. |
| `build_NN_<name>.py` | 35 | The **source of truth for each notebook**. Each script generates `notebooks/NN_<name>.ipynb` cells from a uniform 11-section template. Re-run when prompts/structure change. |
| `tailor_NN_commentary.py` | 35 | Rewrites the §9 "What we just observed" cell of each notebook using values captured from the executed run. Run AFTER papermill. |
| `execute_notebooks.py` | 1 | Driver that calls `build → papermill → tailor` for one or many notebooks. Used by the `notebook-execute.yml` GitHub Actions workflow. |

## The reproduction pipeline

```
scripts/build_NN_<name>.py     →  notebooks/NN_<name>.ipynb         (raw cells)
papermill                      →  notebooks/NN_<name>.ipynb         (executed; outputs captured)
scripts/tailor_NN_<name>.py    →  notebooks/NN_<name>.ipynb         (§9 rewritten from captured output)
```

`scripts/execute_notebooks.py` runs all three stages for you.

## When you'd touch these

| Scenario | Touch |
|---|---|
| Add a new architecture | Create `build_NN_<name>.py` + `tailor_NN_<name>.py` (template-copy from `build_15_rlhf.py`) |
| Llama-3.3 deprecated; want to switch to a new model | Update `.env`, re-run `python scripts/execute_notebooks.py` |
| Found a prompt-tuning improvement | Edit the relevant `build_NN_<name>.py`, re-run the pipeline for that NN |
| Tailor's regex broke after an output-format change | Edit `tailor_NN_commentary.py` |

See [docs/tutorials/adding-your-own](https://fareedkhan-dev.github.io/all-agentic-architectures/tutorials/adding-your-own/) for the full 5-step add-an-architecture recipe.
