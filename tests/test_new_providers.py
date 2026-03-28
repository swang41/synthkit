"""Tests for AnthropicProvider and OpenAIProvider."""
from unittest.mock import MagicMock, patch

import pytest

from synthkit.providers.anthropic import AnthropicProvider
from synthkit.providers.openai import OpenAIProvider


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


def test_anthropic_provider_requires_anthropic_package():
    provider = AnthropicProvider(model="claude-haiku-4-5-20251001")

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(RuntimeError, match="anthropic is not installed"):
            provider.generate("hello")


def test_anthropic_provider_calls_messages_create(monkeypatch):
    fake_content = MagicMock()
    fake_content.text = "Claude says hi"

    fake_response = MagicMock()
    fake_response.content = [fake_content]

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response

    fake_anthropic_module = MagicMock()
    fake_anthropic_module.Anthropic.return_value = fake_client

    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_anthropic_module)

    provider = AnthropicProvider(model="claude-test", api_key="sk-test")
    # Reset cached client so the mock is used
    provider._client = None

    result = provider.generate("Say hi", max_new_tokens=32, temperature=0.5, seed=7)

    assert result == "Claude says hi"
    fake_client.messages.create.assert_called_once_with(
        model="claude-test",
        max_tokens=32,
        temperature=0.5,
        messages=[{"role": "user", "content": "Say hi"}],
    )


def test_anthropic_provider_empty_content_returns_empty_string(monkeypatch):
    fake_response = MagicMock()
    fake_response.content = []

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response

    fake_anthropic_module = MagicMock()
    fake_anthropic_module.Anthropic.return_value = fake_client

    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_anthropic_module)

    provider = AnthropicProvider(model="claude-test")
    provider._client = None

    assert provider.generate("hello") == ""


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


def test_openai_provider_requires_openai_package():
    provider = OpenAIProvider(model="gpt-4o-mini")

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(RuntimeError, match="openai is not installed"):
            provider.generate("hello")


def test_openai_provider_calls_chat_completions_create(monkeypatch):
    fake_message = MagicMock()
    fake_message.content = " GPT says hi "

    fake_choice = MagicMock()
    fake_choice.message = fake_message

    fake_response = MagicMock()
    fake_response.choices = [fake_choice]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    fake_openai_module = MagicMock()
    fake_openai_module.OpenAI.return_value = fake_client

    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai_module)

    provider = OpenAIProvider(model="gpt-test", api_key="sk-test")
    provider._client = None

    result = provider.generate("Say hi", max_new_tokens=16, temperature=0.3, seed=99)

    assert result == "GPT says hi"
    fake_client.chat.completions.create.assert_called_once_with(
        model="gpt-test",
        messages=[{"role": "user", "content": "Say hi"}],
        max_tokens=16,
        temperature=0.3,
        seed=99,
    )


def test_openai_provider_empty_choices_returns_empty_string(monkeypatch):
    fake_response = MagicMock()
    fake_response.choices = []

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    fake_openai_module = MagicMock()
    fake_openai_module.OpenAI.return_value = fake_client

    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai_module)

    provider = OpenAIProvider(model="gpt-test")
    provider._client = None

    assert provider.generate("hello") == ""


def test_openai_provider_base_url_passed_to_client(monkeypatch):
    fake_response = MagicMock()
    fake_response.choices = []

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    fake_openai_module = MagicMock()
    fake_openai_module.OpenAI.return_value = fake_client

    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai_module)

    provider = OpenAIProvider(model="local", base_url="http://localhost:11434/v1")
    provider._client = None
    provider.generate("hi")

    fake_openai_module.OpenAI.assert_called_once_with(base_url="http://localhost:11434/v1")
