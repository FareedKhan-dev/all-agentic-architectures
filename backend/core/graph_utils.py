"""Utilitaires LangGraph (conditions, contrôleurs, etc.)."""

from typing import Any, Dict


def passthrough(state: Dict[str, Any]) -> Dict[str, Any]:
    return state
