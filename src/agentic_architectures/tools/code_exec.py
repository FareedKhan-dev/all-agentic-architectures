"""Sandboxed code execution tool for coding-style agents (Reflection, SWE-Agent).

NOTE: `PythonREPL` does NOT provide a strong security boundary. For real
production use, swap in a container-based sandbox like e2b or modal. This tool
is suitable for trusted notebook demos.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    pass


@tool
def python_repl_tool(code: str) -> str:
    """Execute a snippet of Python code and return stdout (or the error).

    Use this when you need to compute something deterministic — math,
    string manipulation, parsing, simple data transformation.
    """
    try:
        from langchain_experimental.utilities import PythonREPL
    except ImportError:
        # Fallback to a built-in minimal exec for environments without langchain-experimental.
        import contextlib
        import io

        buf = io.StringIO()
        local_vars: dict[str, object] = {}
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, {"__builtins__": __builtins__}, local_vars)  # noqa: S102
            out = buf.getvalue()
            return out if out else repr(local_vars)
        except Exception as e:  # pragma: no cover  (deterministic enough)
            return f"ERROR: {type(e).__name__}: {e}"

    repl = PythonREPL()
    return repl.run(code)
