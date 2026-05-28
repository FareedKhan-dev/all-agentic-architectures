"""Tests for the pure-Python helper functions that ARE the deterministic-picker.

These functions are the architecture's escape from the LLM-as-Scorer flat-band
pathology — they MUST work correctly for the safety/quality guarantees to hold.
No mocking needed: they take Python data, return Python data.
"""

from __future__ import annotations

import pytest

from agentic_architectures.architectures.constitutional_ai import DEFAULT_CONSTITUTION
from agentic_architectures.architectures.lats import LATS
from agentic_architectures.architectures.reflexion import (
    _count_syllables_word,
    _line_syllables,
    default_haiku_checker,
)
from agentic_architectures.architectures.rlhf import RLHFSelfImprovement
from agentic_architectures.architectures.self_discover import MODULE_LIBRARY
from agentic_architectures.architectures.voyager import _exec_skill


# ---------------------------------------------------------------------------
# nb 18 Reflexion — haiku checker (the deciding signal for the loop)
# ---------------------------------------------------------------------------
class TestHaikuChecker:
    def test_count_syllables_basic(self) -> None:
        assert _count_syllables_word("cat") == 1
        assert _count_syllables_word("butter") == 2
        assert _count_syllables_word("centuries") == 3  # cen-tu-ries, silent 'e' rule applies

    def test_count_syllables_silent_e(self) -> None:
        assert _count_syllables_word("make") == 1
        assert _count_syllables_word("rage") == 1

    def test_count_syllables_empty_or_punct(self) -> None:
        assert _count_syllables_word("") == 0
        assert _count_syllables_word("!?") == 0

    def test_line_syllables(self) -> None:
        # "Frozen silence deep" — 2+2+1 = 5
        assert _line_syllables("Frozen silence deep") == 5
        # "Centuries of slow descent" — 3+1+1+2 = 7
        assert _line_syllables("Centuries of slow descent") == 7

    def test_haiku_passes_when_valid(self) -> None:
        cand = "Frozen silence deep\nCenturies of slow descent\nGlacial morning calm"
        spec = "X. spec=topic=glacier; required_words=silence,centuries"
        feats = default_haiku_checker(cand, spec)
        assert feats["passed"] is True
        assert feats["meets_5_7_5"] is True
        assert feats["required_words_present"] is True
        assert feats["words_present_detail"] == {"silence": True, "centuries": True}

    def test_haiku_fails_on_wrong_syllables(self) -> None:
        cand = "Frozen silence\nCenturies of slow descent\nGlacial morning calm"  # line 1 = 4
        spec = "spec=topic=glacier; required_words=silence,centuries"
        feats = default_haiku_checker(cand, spec)
        assert feats["meets_5_7_5"] is False
        assert feats["passed"] is False

    def test_haiku_fails_on_missing_word(self) -> None:
        cand = "Frozen silence deep\nWinter mornings stretching long\nGlacial morning calm"
        spec = "spec=topic=glacier; required_words=silence,centuries"
        feats = default_haiku_checker(cand, spec)
        assert feats["required_words_present"] is False
        assert feats["words_present_detail"]["centuries"] is False
        assert feats["passed"] is False


# ---------------------------------------------------------------------------
# nb 15 RLHF — composite scoring (the deterministic-picker generalization)
# ---------------------------------------------------------------------------
class TestRLHFCompositeScore:
    @staticmethod
    def _score(**features: object) -> int:
        return RLHFSelfImprovement._composite_score(features, (30, 100))

    def test_all_true_max_score(self) -> None:
        assert self._score(
            is_on_brief=True, word_count=60, has_concrete_imagery=True,
            avoids_cliches=True, is_engaging=True,
        ) == 10

    def test_all_false_zero(self) -> None:
        assert self._score(
            is_on_brief=False, word_count=0, has_concrete_imagery=False,
            avoids_cliches=False, is_engaging=False,
        ) == 0

    def test_only_on_brief_contributes_4(self) -> None:
        assert self._score(
            is_on_brief=True, word_count=0,
        ) == 4

    def test_word_count_out_of_range_zero_contribution(self) -> None:
        # 200 is outside (30, 100) range
        s_in = self._score(is_on_brief=True, word_count=50)
        s_out = self._score(is_on_brief=True, word_count=200)
        assert s_in - s_out == 2  # word_count loses 2 points


# ---------------------------------------------------------------------------
# nb 22 LATS — composite value (deterministic-picker on tree leaves)
# ---------------------------------------------------------------------------
class TestLATSComposite:
    def test_high_confidence_complete_tops_out(self) -> None:
        v = LATS._composite_value({
            "is_complete": True, "makes_progress": True,
            "avoids_loops": True, "confidence": "high",
        })
        assert v == 10  # 5 + 2 + 1 + 2

    def test_low_confidence_no_progress(self) -> None:
        assert LATS._composite_value({
            "is_complete": False, "makes_progress": False,
            "avoids_loops": False, "confidence": "low",
        }) == 0

    def test_unknown_confidence_defaults_to_zero(self) -> None:
        v = LATS._composite_value({
            "is_complete": False, "makes_progress": True,
            "avoids_loops": True, "confidence": "weirdvalue",
        })
        assert v == 3  # 2 (progress) + 1 (no_loops) + 0 (unknown conf)


# ---------------------------------------------------------------------------
# nb 19 Self-Discover — the static module library
# ---------------------------------------------------------------------------
def test_self_discover_module_library_has_at_least_10() -> None:
    assert len(MODULE_LIBRARY) >= 10
    assert all(isinstance(m, str) and m for m in MODULE_LIBRARY)


# ---------------------------------------------------------------------------
# nb 32 Constitutional AI — default constitution
# ---------------------------------------------------------------------------
def test_constitutional_default_constitution_non_empty() -> None:
    assert len(DEFAULT_CONSTITUTION) >= 3
    assert all(isinstance(r, str) and r for r in DEFAULT_CONSTITUTION)


# ---------------------------------------------------------------------------
# nb 29 Voyager — real subprocess skill execution
# ---------------------------------------------------------------------------
class TestVoyagerExec:
    def test_simple_arithmetic_runs(self) -> None:
        code = "def add(a, b):\n    return a + b\n"
        stdout, ok, err = _exec_skill(code, "add(2, 3)")
        assert ok is True
        assert stdout == "5"
        assert err == ""

    def test_factorial_via_subprocess(self) -> None:
        code = "def factorial(n):\n    return 1 if n == 0 else n * factorial(n - 1)\n"
        stdout, ok, _ = _exec_skill(code, "factorial(5)")
        assert ok
        assert stdout == "120"

    def test_bad_code_returns_error(self) -> None:
        code = "def broken(\nthis is not python"
        stdout, ok, err = _exec_skill(code, "broken()")
        assert ok is False
        assert err  # some stderr from the subprocess

    def test_timeout_caps_runaway_loop(self) -> None:
        code = "def loop():\n    while True: pass\n"
        stdout, ok, err = _exec_skill(code, "loop()", timeout=1)
        assert ok is False
        assert "timeout" in err.lower()


# ---------------------------------------------------------------------------
# nb 34 BrowserAgent — safety gate
# ---------------------------------------------------------------------------
class TestBrowserSafetyGate:
    def _arch(self, mock_llm):
        from agentic_architectures.architectures import BrowserAgent
        return BrowserAgent(llm=mock_llm, blocked_domains=["evil.com"])

    def test_navigate_to_blocked_domain_rejected(self, mock_llm) -> None:
        arch = self._arch(mock_llm)
        allowed, reason = arch._check_safety({"action": "navigate", "target": "https://evil.com/login", "value": "", "answer": ""})
        assert allowed is False
        assert "evil.com" in reason

    def test_navigate_to_allowed_url_passes(self, mock_llm) -> None:
        arch = self._arch(mock_llm)
        allowed, _ = arch._check_safety({"action": "navigate", "target": "https://example.com", "value": "", "answer": ""})
        assert allowed is True

    def test_non_http_url_rejected(self, mock_llm) -> None:
        arch = self._arch(mock_llm)
        allowed, reason = arch._check_safety({"action": "navigate", "target": "file:///etc/passwd", "value": "", "answer": ""})
        assert allowed is False
        assert "http" in reason.lower()

    def test_answer_with_password_pattern_rejected(self, mock_llm) -> None:
        arch = self._arch(mock_llm)
        allowed, reason = arch._check_safety({"action": "answer", "target": "", "value": "the password is hunter2", "answer": ""})
        assert allowed is False
        assert "password" in reason.lower()

    def test_clean_answer_passes(self, mock_llm) -> None:
        arch = self._arch(mock_llm)
        allowed, _ = arch._check_safety({"action": "answer", "target": "", "value": "The capital is Paris.", "answer": ""})
        assert allowed is True


# ---------------------------------------------------------------------------
# nb 33 SWE-Agent — sandbox path resolution
# ---------------------------------------------------------------------------
class TestSWESandbox:
    def test_sandbox_rejects_path_escape(self, mock_llm, tmp_path) -> None:
        from agentic_architectures.architectures import SWEAgent
        arch = SWEAgent(llm=mock_llm, working_dir=tmp_path)
        with pytest.raises(PermissionError):
            arch._safe_path("../../../etc/passwd")

    def test_sandbox_allows_relative_path(self, mock_llm, tmp_path) -> None:
        from agentic_architectures.architectures import SWEAgent
        arch = SWEAgent(llm=mock_llm, working_dir=tmp_path)
        p = arch._safe_path("foo.txt")
        assert str(p).startswith(str(tmp_path.resolve()))
