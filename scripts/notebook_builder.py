"""Tiny helper for building notebooks programmatically.

Lets us define each notebook as a clean sequence of (markdown, code) cells,
then materialize it as a .ipynb that papermill can execute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import nbformat as nbf

CellKind = Literal["md", "code"]


def md(text: str) -> tuple[CellKind, str]:
    return ("md", text)


def code(text: str) -> tuple[CellKind, str]:
    return ("code", text.strip("\n"))


def build_notebook(
    cells: list[tuple[CellKind, str]],
    out_path: str | Path,
    kernel_name: str = "python3",
    display_name: str = "Python 3",
) -> Path:
    """Materialize a list of (kind, text) cells as a .ipynb file."""
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": display_name,
            "language": "python",
            "name": kernel_name,
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    }
    nb_cells = []
    for kind, text in cells:
        if kind == "md":
            nb_cells.append(nbf.v4.new_markdown_cell(text))
        elif kind == "code":
            nb_cells.append(nbf.v4.new_code_cell(text))
        else:  # pragma: no cover -- exhaustive Literal
            raise ValueError(f"unknown cell kind: {kind!r}")
    nb["cells"] = nb_cells

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    return out
