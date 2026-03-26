"""Unit tests for utils/llm_provider.py."""

import os
import unittest
from unittest.mock import MagicMock, patch


class TestProviderDefaults(unittest.TestCase):
    """Verify PROVIDER_DEFAULTS and SUPPORTED_PROVIDERS constants."""

    def test_supported_providers_contains_expected(self):
        from utils.llm_provider import SUPPORTED_PROVIDERS

        self.assertIn("nebius", SUPPORTED_PROVIDERS)
        self.assertIn("minimax", SUPPORTED_PROVIDERS)
        self.assertIn("openai", SUPPORTED_PROVIDERS)

    def test_provider_defaults_have_model_and_temperature(self):
        from utils.llm_provider import PROVIDER_DEFAULTS

        for provider, defaults in PROVIDER_DEFAULTS.items():
            self.assertIn("model", defaults, f"{provider} missing model")
            self.assertIn("temperature", defaults, f"{provider} missing temperature")

    def test_minimax_default_model(self):
        from utils.llm_provider import PROVIDER_DEFAULTS

        self.assertEqual(PROVIDER_DEFAULTS["minimax"]["model"], "MiniMax-M2.7")


class TestClampTemperature(unittest.TestCase):
    """Temperature clamping for MiniMax (0, 1]."""

    def test_zero_is_clamped(self):
        from utils.llm_provider import _clamp_temperature

        self.assertGreater(_clamp_temperature(0.0), 0.0)

    def test_negative_is_clamped(self):
        from utils.llm_provider import _clamp_temperature

        self.assertGreater(_clamp_temperature(-1.0), 0.0)

    def test_above_one_is_clamped(self):
        from utils.llm_provider import _clamp_temperature

        self.assertEqual(_clamp_temperature(1.5), 1.0)

    def test_valid_value_unchanged(self):
        from utils.llm_provider import _clamp_temperature

        self.assertEqual(_clamp_temperature(0.5), 0.5)

    def test_one_is_kept(self):
        from utils.llm_provider import _clamp_temperature

        self.assertEqual(_clamp_temperature(1.0), 1.0)

    def test_small_positive_kept(self):
        from utils.llm_provider import _clamp_temperature

        self.assertEqual(_clamp_temperature(0.01), 0.01)


class TestGetLlmUnsupported(unittest.TestCase):
    """get_llm raises on unknown provider."""

    def test_invalid_provider_raises(self):
        from utils.llm_provider import get_llm

        with self.assertRaises(ValueError) as ctx:
            get_llm(provider="unknown_provider")
        self.assertIn("unknown_provider", str(ctx.exception))

    def test_error_lists_supported(self):
        from utils.llm_provider import get_llm

        with self.assertRaises(ValueError) as ctx:
            get_llm(provider="bad")
        self.assertIn("nebius", str(ctx.exception))
        self.assertIn("minimax", str(ctx.exception))


class TestGetLlmEnvFallback(unittest.TestCase):
    """get_llm reads LLM_PROVIDER from env."""

    @patch.dict(os.environ, {"LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": "test_key"})
    @patch("langchain_openai.ChatOpenAI")
    def test_env_provider_minimax(self, mock_cls):
        mock_cls.return_value = MagicMock()
        from utils.llm_provider import get_llm

        llm = get_llm()
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args
        self.assertEqual(call_kwargs.kwargs["model"], "MiniMax-M2.7")
        self.assertEqual(
            call_kwargs.kwargs["openai_api_base"], "https://api.minimax.io/v1"
        )

    @patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test_key"})
    @patch("langchain_openai.ChatOpenAI")
    def test_env_provider_openai(self, mock_cls):
        mock_cls.return_value = MagicMock()
        from utils.llm_provider import get_llm

        llm = get_llm()
        mock_cls.assert_called_once()
        self.assertEqual(mock_cls.call_args.kwargs["model"], "gpt-4o-mini")


class TestGetLlmMinimax(unittest.TestCase):
    """MiniMax-specific logic in get_llm."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"})
    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_uses_openai_compat(self, mock_cls):
        mock_cls.return_value = MagicMock()
        from utils.llm_provider import get_llm

        get_llm(provider="minimax")
        kw = mock_cls.call_args.kwargs
        self.assertEqual(kw["openai_api_base"], "https://api.minimax.io/v1")
        self.assertEqual(kw["openai_api_key"], "test_key")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"})
    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_temp_clamped(self, mock_cls):
        mock_cls.return_value = MagicMock()
        from utils.llm_provider import get_llm

        get_llm(provider="minimax", temperature=0.0)
        kw = mock_cls.call_args.kwargs
        self.assertGreater(kw["temperature"], 0.0)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"})
    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_model_override(self, mock_cls):
        mock_cls.return_value = MagicMock()
        from utils.llm_provider import get_llm

        get_llm(provider="minimax", model="MiniMax-M2.7-highspeed")
        kw = mock_cls.call_args.kwargs
        self.assertEqual(kw["model"], "MiniMax-M2.7-highspeed")

    @patch.dict(os.environ, {}, clear=False)
    def test_minimax_missing_key_raises(self):
        env = os.environ.copy()
        env.pop("MINIMAX_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            from utils.llm_provider import get_llm

            with self.assertRaises(EnvironmentError):
                get_llm(provider="minimax")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "key123"})
    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_kwargs_forwarded(self, mock_cls):
        mock_cls.return_value = MagicMock()
        from utils.llm_provider import get_llm

        get_llm(provider="minimax", max_tokens=1024)
        kw = mock_cls.call_args.kwargs
        self.assertEqual(kw["max_tokens"], 1024)


class TestGetLlmModelEnvOverride(unittest.TestCase):
    """LLM_MODEL env var overrides default model."""

    @patch.dict(
        os.environ,
        {"LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": "k", "LLM_MODEL": "custom-m"},
    )
    @patch("langchain_openai.ChatOpenAI")
    def test_model_env_override(self, mock_cls):
        mock_cls.return_value = MagicMock()
        from utils.llm_provider import get_llm

        get_llm()
        self.assertEqual(mock_cls.call_args.kwargs["model"], "custom-m")


class TestGetLlmNebius(unittest.TestCase):
    """Nebius provider path."""

    @patch("langchain_nebius.ChatNebius")
    def test_nebius_default(self, mock_cls):
        mock_cls.return_value = MagicMock()
        from utils.llm_provider import get_llm

        get_llm(provider="nebius")
        mock_cls.assert_called_once()
        kw = mock_cls.call_args.kwargs
        self.assertEqual(kw["model"], "meta-llama/Meta-Llama-3.1-8B-Instruct")

    @patch("langchain_nebius.ChatNebius")
    def test_nebius_custom_temp(self, mock_cls):
        mock_cls.return_value = MagicMock()
        from utils.llm_provider import get_llm

        get_llm(provider="nebius", temperature=0.7)
        self.assertEqual(mock_cls.call_args.kwargs["temperature"], 0.7)


if __name__ == "__main__":
    unittest.main()
