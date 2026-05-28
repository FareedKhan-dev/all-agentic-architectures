"""Safe filesystem tools — used by SWE-Agent / coding architectures.

Reads / writes are scoped to a configurable root directory to prevent the agent
from escaping the sandbox.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

_ROOT: Path = Path.cwd().resolve()


def set_sandbox_root(root: Path | str) -> None:
    """Set the directory all read/write tools are confined to."""
    global _ROOT
    _ROOT = Path(root).resolve()


def _resolve(p: str) -> Path:
    resolved = (_ROOT / p).resolve()
    if _ROOT not in resolved.parents and resolved != _ROOT:
        raise ValueError(f"Path escapes sandbox: {p}")
    return resolved


@tool
def read_file(path: str) -> str:
    """Return the contents of a file (sandboxed)."""
    return _resolve(path).read_text(encoding="utf-8")


@tool
def write_file(path: str, content: str) -> str:
    """Write `content` to `path` (sandboxed). Overwrites existing files."""
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


@tool
def list_dir(path: str = ".") -> list[str]:
    """List the contents of a directory (sandboxed)."""
    return [p.name for p in _resolve(path).iterdir()]
