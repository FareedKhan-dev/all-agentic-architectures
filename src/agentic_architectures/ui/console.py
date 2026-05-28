"""Pretty-print helpers shared across notebooks (replaces ~85 lines of dup boilerplate)."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty
from rich.rule import Rule

console = Console()


def print_md(text: str) -> None:
    """Render markdown to the console."""
    console.print(Markdown(text))


def print_header(title: str, subtitle: str | None = None) -> None:
    """Bold section header used at the start of each notebook stage."""
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", align="left"))
    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]\n")


def print_step(step: str, body: str | None = None) -> None:
    """Per-iteration step marker used inside agent loops."""
    console.print(f"[bold magenta]›[/bold magenta] [bold]{step}[/bold]")
    if body:
        console.print(body)


def print_state(state: dict[str, Any], title: str = "State") -> None:
    """Render an agent state dict in a panel."""
    console.print(Panel(Pretty(state, expand_all=True), title=title, border_style="dim"))
