import json

import pytest

from synthkit.providers.huggingface import HuggingFaceProvider
from synthkit.providers.remote import RemoteLLMProvider


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_huggingface_provider_requires_transformers(monkeypatch):
    provider = HuggingFaceProvider(model="dummy/model")

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RuntimeError, match="transformers is not installed"):
        provider.generate("hello")


def test_remote_provider_openai_compatible_response(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse({"choices": [{"message": {"content": "ok from remote"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    provider = RemoteLLMProvider(
        endpoint_url="http://localhost:8000/v1/chat/completions",
        model="test-model",
        api_key="secret",
    )

    result = provider.generate("Say hello", max_new_tokens=12, temperature=0.2, seed=9)

    assert result == "ok from remote"
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["headers"].get("Authorization") == "Bearer secret"
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["messages"][0]["content"] == "Say hello"
    assert captured["body"]["max_tokens"] == 12
    assert captured["body"]["temperature"] == 0.2
    assert captured["body"]["seed"] == 9
    assert captured["timeout"] == 60.0
