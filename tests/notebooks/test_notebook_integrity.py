"""Static integrity checks for every notebook in notebooks/.

We do NOT execute the notebooks here — that takes ~30 minutes for all 35.
Instead we verify each .ipynb meets the standards from the handoff §5:
  - Has been executed (all code cells have non-None execution_count).
  - No cell contains an error output.
  - The mandatory §9 "What we just observed" markdown cell is present AND tailored
    (not still showing the placeholder text).
  - Roughly matches the 11-section template.

For a full top-to-bottom run, use `papermill` (see scripts/) or `pytest --nbval-lax`.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest


NOTEBOOKS_DIR = Path(__file__).parents[2] / "notebooks"
ALL_NB_PATHS = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))


def _read(p: Path) -> nbformat.NotebookNode:
    return nbformat.read(p, as_version=4)


def test_notebooks_directory_has_at_least_thirty_five() -> None:
    """Phase 2 (17) + Phase 3 (18) = 35 notebooks expected."""
    assert len(ALL_NB_PATHS) >= 35, f"Expected ≥35 notebooks, got {len(ALL_NB_PATHS)}: {[p.name for p in ALL_NB_PATHS]}"


@pytest.mark.parametrize("nb_path", ALL_NB_PATHS, ids=[p.stem for p in ALL_NB_PATHS])
def test_notebook_was_executed(nb_path: Path) -> None:
    """Every code cell must have a non-None execution_count (i.e., was run)."""
    nb = _read(nb_path)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    if not code_cells:
        pytest.skip("notebook has no code cells")
    unexecuted = [i for i, c in enumerate(code_cells) if c.execution_count is None]
    assert not unexecuted, f"{nb_path.name}: code cells not executed: {unexecuted}"


@pytest.mark.parametrize("nb_path", ALL_NB_PATHS, ids=[p.stem for p in ALL_NB_PATHS])
def test_notebook_has_no_error_outputs(nb_path: Path) -> None:
    """No cell may contain an error output (raised exception, traceback)."""
    nb = _read(nb_path)
    errors: list[tuple[int, str]] = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.outputs:
            if out.get("output_type") == "error":
                errors.append((i, out.get("ename", "Error")))
    assert not errors, f"{nb_path.name}: error outputs in cells {errors}"


@pytest.mark.parametrize("nb_path", ALL_NB_PATHS, ids=[p.stem for p in ALL_NB_PATHS])
def test_notebook_has_tailored_section_9(nb_path: Path) -> None:
    """Every notebook must contain the mandatory §9 'What we just observed' cell,
    and it must be tailored (not the placeholder)."""
    nb = _read(nb_path)
    s9_cell = None
    for cell in nb.cells:
        if cell.cell_type == "markdown" and cell.source.lstrip().startswith("## 9 · What we just observed"):
            s9_cell = cell
            break
    assert s9_cell is not None, f"{nb_path.name}: missing §9 'What we just observed' cell"
    # Tailored cells reference concrete data; placeholder always contains "tailor_"
    assert "Automatically tailored" not in s9_cell.source, (
        f"{nb_path.name}: §9 is still the placeholder — run scripts/tailor_NN_commentary.py"
    )
    # Tailored §9 should be substantive (>200 chars)
    assert len(s9_cell.source) > 200, f"{nb_path.name}: §9 cell is too short ({len(s9_cell.source)} chars)"


@pytest.mark.parametrize("nb_path", ALL_NB_PATHS, ids=[p.stem for p in ALL_NB_PATHS])
def test_notebook_has_multiple_section_headings(nb_path: Path) -> None:
    """The 11-section template — sanity-check that the notebook has multiple section headings."""
    nb = _read(nb_path)
    headings = sum(
        1 for c in nb.cells
        if c.cell_type == "markdown" and c.source.lstrip().startswith("## ")
    )
    # We allow a wide range — some compact notebooks have ~6, full template has 11.
    assert headings >= 4, f"{nb_path.name}: only {headings} section headings, expected ≥4"
