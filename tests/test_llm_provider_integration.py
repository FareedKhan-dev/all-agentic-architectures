"""Integration tests for utils/llm_provider.py.

These tests make real API calls to verify end-to-end provider integration.
They are skipped when the corresponding API key env var is not set.

Run with: python -m pytest tests/test_llm_provider_integration.py -v
"""

import os
import unittest

from dotenv import load_dotenv

load_dotenv()


@unittest.skipUnless(
    os.getenv("MINIMAX_API_KEY"), "MINIMAX_API_KEY not set — skipping MiniMax integration tests"
)
class TestMiniMaxIntegration(unittest.TestCase):
    """Live MiniMax API integration tests."""

    def test_minimax_simple_invoke(self):
        from utils.llm_provider import get_llm

        llm = get_llm(provider="minimax", model="MiniMax-M2.7-highspeed")
        response = llm.invoke("Say 'hello' and nothing else.")
        self.assertTrue(len(response.content) > 0)
        self.assertIn("hello", response.content.lower())

    def test_minimax_with_system_message(self):
        from langchain_core.messages import HumanMessage, SystemMessage
        from utils.llm_provider import get_llm

        llm = get_llm(provider="minimax", model="MiniMax-M2.7-highspeed")
        response = llm.invoke([
            SystemMessage(content="You are a helpful assistant. Reply concisely."),
            HumanMessage(content="What is 2+2? Reply with just the number."),
        ])
        self.assertIn("4", response.content)

    def test_minimax_temperature_clamping_live(self):
        from utils.llm_provider import get_llm

        llm = get_llm(provider="minimax", temperature=0.0)
        response = llm.invoke("What is 2+2? Reply with just the number.")
        self.assertIn("4", response.content)


@unittest.skipUnless(
    os.getenv("NEBIUS_API_KEY"), "NEBIUS_API_KEY not set — skipping Nebius integration tests"
)
class TestNebiusIntegration(unittest.TestCase):
    """Live Nebius API integration tests."""

    def test_nebius_simple_invoke(self):
        from utils.llm_provider import get_llm

        llm = get_llm(provider="nebius")
        response = llm.invoke("Say 'hello' and nothing else.")
        self.assertTrue(len(response.content) > 0)


@unittest.skipUnless(
    os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set — skipping OpenAI integration tests"
)
class TestOpenAIIntegration(unittest.TestCase):
    """Live OpenAI API integration tests."""

    def test_openai_simple_invoke(self):
        from utils.llm_provider import get_llm

        llm = get_llm(provider="openai")
        response = llm.invoke("Say 'hello' and nothing else.")
        self.assertTrue(len(response.content) > 0)


if __name__ == "__main__":
    unittest.main()
