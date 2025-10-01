from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# A01 — Reflection
class ReflectionIn(BaseModel):
    prompt: str
    criteria: Optional[List[str]] = None
    max_loops: int = 1

class ReflectionOut(BaseModel):
    answer: str
    critic: Optional[str] = None

# A02 — Tool Use
class ToolUseIn(BaseModel):
    question: str

class ToolUseOut(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)

# RAG QA (optionnel)
class RAGQAIn(BaseModel):
    question: str
    k: int = 5

class RAGQAOut(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)

# ---------------- Catalogue Patterns ----------------
class PatternMeta(BaseModel):
    id_pattern: str
    nom_fr: str
    categorie: str  # Raisonnement | Action | Amélioration | Orchestration
    fonctionnement_court: str
    utilite_concrets: List[str] = Field(default_factory=list)
    detail_technique: str = ""           # texte long (markdown accepté)
    implication_backend: str = ""        # texte long
    compromis_cout_latence: str = ""     # texte long
    has_demo: bool = False               # true si /api/execute/{id} est implémenté
    tags: List[str] = Field(default_factory=list)

class PatternListOut(BaseModel):
    items: List[PatternMeta]

class PatternExecIn(BaseModel):
    input: Dict[str, Any] = Field(default_factory=dict)

class PatternExecOut(BaseModel):
    ok: bool
    id_pattern: str
    output: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)  # latence, nb appels, $
