"""STORM — Synthesis of Topic Outlines via Retrieval and Multi-perspective questioning.

Pipeline:
  1. **PERSPECTIVES** — brainstorm N distinct viewpoints on the topic.
  2. **QUESTIONS** — each perspective generates K questions.
  3. **ANSWER** — answer each question (via web search OR LLM).
  4. **OUTLINE** — synthesise into a structured article outline.
  5. **WRITE** — draft each section.

Origin: Shao et al., *STORM: Assisting in Writing Wikipedia-like Articles
From Scratch with Large Language Models* (Stanford 2024).
https://arxiv.org/abs/2402.14207

Composes [Multi-Agent (nb 05)](./05_multi_agent.ipynb) (perspectives as
specialists) + [Planning (nb 04)](./04_planning.ipynb) (outline-as-plan).
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_architectures.architectures.base import Architecture, ArchitectureResult


class _Perspectives(BaseModel):
    perspectives: list[str] = Field(
        description="N distinct viewpoints / framings of the topic. "
                    "Each is 1-2 sentences. Be substantively different — not paraphrases.",
        min_length=2, max_length=6,
    )


class _Questions(BaseModel):
    questions: list[str] = Field(
        description="K specific research questions about the topic, "
                    "framed from this perspective.",
        min_length=1, max_length=5,
    )


class _OutlineSection(BaseModel):
    title: str = Field(description="Section heading.")
    key_points: list[str] = Field(description="3-6 bullet points the section will cover.")


class _Outline(BaseModel):
    sections: list[_OutlineSection] = Field(min_length=2, max_length=8)


class _ArticleSection(BaseModel):
    title: str
    body: str = Field(description="2-4 paragraphs of polished prose.")


class STORMState(TypedDict, total=False):
    topic: str
    perspectives: list[str]
    questions: list[dict[str, Any]]
    answers: list[dict[str, str]]
    outline: list[dict[str, Any]]
    article_sections: list[dict[str, str]]
    final_answer: str
    history: Annotated[list[dict[str, Any]], operator.add]


class STORM(Architecture):
    """Multi-perspective research → outline → article."""

    name = "storm"
    description = (
        "Multi-perspective research pipeline: brainstorm perspectives, generate "
        "questions per perspective, answer them, build outline, write article."
    )
    reference = "https://arxiv.org/abs/2402.14207"

    def __init__(
        self,
        n_perspectives: int = 3,
        questions_per_perspective: int = 2,
        web_search_fn: Callable[[str], list[str]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_perspectives = n_perspectives
        self.questions_per_perspective = questions_per_perspective
        self.web_search_fn = web_search_fn
        self._perspectives = self.llm.with_structured_output(_Perspectives)
        self._questions = self.llm.with_structured_output(_Questions)
        self._outline = self.llm.with_structured_output(_Outline)
        self._writer = self.llm.with_structured_output(_ArticleSection)

    def _gen_perspectives(self, state: STORMState) -> dict[str, Any]:
        p = self._perspectives.invoke(
            f"Brainstorm {self.n_perspectives} distinct viewpoints / framings for an "
            f"article about: '{state['topic']}'. Each viewpoint should be a different "
            "angle (technical, social, historical, etc.)."
        )
        return {
            "perspectives": list(p.perspectives)[: self.n_perspectives],
            "history": [{"stage": "perspectives", "n": len(p.perspectives)}],
        }

    def _gen_questions(self, state: STORMState) -> dict[str, Any]:
        all_q: list[dict[str, Any]] = []
        for persp in state["perspectives"]:
            try:
                q = self._questions.invoke(
                    f"# Topic\n{state['topic']}\n\n"
                    f"# Perspective\n{persp}\n\n"
                    f"Generate {self.questions_per_perspective} specific research questions "
                    "from this perspective."
                )
                for qt in list(q.questions)[: self.questions_per_perspective]:
                    all_q.append({"perspective": persp, "question": qt})
            except Exception:
                continue
        return {
            "questions": all_q,
            "history": [{"stage": "questions", "n_total": len(all_q)}],
        }

    def _answer_questions(self, state: STORMState) -> dict[str, Any]:
        answers: list[dict[str, str]] = []
        for item in state["questions"]:
            q = item["question"]
            if self.web_search_fn:
                try:
                    web = self.web_search_fn(q)
                    ctx = "\n".join(f"- {w[:300]}" for w in web[:3]) if web else "(no web results)"
                    ans = str(self.llm.invoke(
                        f"Answer concisely (1-2 sentences) using the web snippets.\n\n# Web\n{ctx}\n\n# Q: {q}\nA:"
                    ).content).strip()
                except Exception as e:
                    ans = f"(web answer failed: {e})"
            else:
                ans = str(self.llm.invoke(
                    f"Answer concisely (1-2 sentences) from your knowledge.\n\n# Q: {q}\nA:"
                ).content).strip()
            answers.append({"question": q, "answer": ans})
        return {
            "answers": answers,
            "history": [{"stage": "answer_questions", "n": len(answers)}],
        }

    def _build_outline(self, state: STORMState) -> dict[str, Any]:
        qa_block = "\n\n".join(f"Q: {a['question']}\nA: {a['answer']}" for a in state["answers"])
        o = self._outline.invoke(
            f"# Topic\n{state['topic']}\n\n"
            f"# Research Q&A\n{qa_block}\n\n"
            "Build a 3-5 section outline for an article on this topic. Each section "
            "should have a title and 3-6 key points to cover."
        )
        sections = [{"title": s.title, "key_points": list(s.key_points)} for s in o.sections]
        return {
            "outline": sections,
            "history": [{"stage": "outline", "n_sections": len(sections)}],
        }

    def _write_article(self, state: STORMState) -> dict[str, Any]:
        sections: list[dict[str, str]] = []
        qa_block = "\n".join(f"  - {a['question']} → {a['answer'][:120]}" for a in state.get("answers", []))
        for sec in state["outline"]:
            try:
                w = self._writer.invoke(
                    f"# Topic\n{state['topic']}\n\n"
                    f"# Section to write\nTitle: {sec['title']}\nKey points: {sec['key_points']}\n\n"
                    f"# Research Q&A available\n{qa_block}\n\n"
                    "Write 2-4 paragraphs of polished prose for this section."
                )
                sections.append({"title": w.title, "body": w.body})
            except Exception as e:
                sections.append({"title": sec["title"], "body": f"(section write failed: {e})"})
        # Compose final
        final = "\n\n".join(f"## {s['title']}\n\n{s['body']}" for s in sections)
        return {
            "article_sections": sections,
            "final_answer": final,
            "history": [{"stage": "write_article", "n_sections": len(sections)}],
        }

    def build(self) -> Any:
        g: StateGraph = StateGraph(STORMState)
        g.add_node("perspectives", self._gen_perspectives)
        g.add_node("questions", self._gen_questions)
        g.add_node("answer", self._answer_questions)
        g.add_node("outline", self._build_outline)
        g.add_node("write", self._write_article)
        g.add_edge(START, "perspectives")
        g.add_edge("perspectives", "questions")
        g.add_edge("questions", "answer")
        g.add_edge("answer", "outline")
        g.add_edge("outline", "write")
        g.add_edge("write", END)
        return g.compile()

    def run(self, task: str, **kwargs: Any) -> ArchitectureResult:
        graph = self.build()
        final_state = graph.invoke({"topic": task}, config={"recursion_limit": 30})
        return ArchitectureResult(
            output=final_state.get("final_answer", ""),
            state={
                "n_perspectives": len(final_state.get("perspectives", [])),
                "n_questions": len(final_state.get("questions", [])),
                "n_sections": len(final_state.get("outline", [])),
            },
            trace=final_state.get("history", []),
            metadata={
                "perspectives": final_state.get("perspectives", []),
                "questions": final_state.get("questions", []),
                "answers": final_state.get("answers", []),
                "outline": final_state.get("outline", []),
                "n_perspectives": len(final_state.get("perspectives", [])),
                "n_questions": len(final_state.get("questions", [])),
                "n_sections": len(final_state.get("outline", [])),
                "article_chars": len(final_state.get("final_answer", "")),
            },
        )
