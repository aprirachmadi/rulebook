"""VRAM guard: unload other models first; refuse a model that would not fit."""

import json

import httpx
import pytest

from rulebook.harness.memory import MemoryGuard, MemoryGuardError, ollama_root


class FakeOllama:
    def __init__(self, loaded: list[str]) -> None:
        self.loaded = list(loaded)
        self.unloaded: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.loaded]})
        if request.url.path == "/api/generate":
            body = json.loads(request.content)
            assert body["keep_alive"] == 0
            self.unloaded.append(body["model"])
            self.loaded.remove(body["model"])
            return httpx.Response(200, json={})
        return httpx.Response(404)

    def guard(self, free: float) -> MemoryGuard:
        http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(self))
        return MemoryGuard(http=http, free_gb=lambda: free)


def test_unloads_other_models_before_loading() -> None:
    ollama = FakeOllama(["gemma4:12b-16k"])
    ollama.guard(free=14.0).prepare("llama3.2:3b-8k")
    assert ollama.unloaded == ["gemma4:12b-16k"]


def test_refuses_when_it_would_not_fit() -> None:
    ollama = FakeOllama([])
    with pytest.raises(MemoryGuardError, match="not loading"):
        ollama.guard(free=10.0).prepare("gemma4:12b-16k")  # needs 9.8 + 1.5


def test_refuses_model_without_estimate() -> None:
    with pytest.raises(MemoryGuardError, match="calculate it"):
        FakeOllama([]).guard(free=15.0).prepare("some-new-model:70b")


def test_already_loaded_alone_is_fine() -> None:
    ollama = FakeOllama(["qwen3.5:4b-8k"])
    ollama.guard(free=0.5).prepare("qwen3.5:4b-8k")  # no load needed, no check
    assert ollama.unloaded == []


def test_every_configured_model_has_an_estimate_that_fits_alone() -> None:
    from rulebook.config import load_config

    cfg = load_config()
    models = [*cfg["models"]["labelers"].values(), cfg["models"]["orchestrator"],
              cfg["models"]["vision"], cfg["models"]["toy_agent"]]
    est, head = cfg["memory"]["estimate_gb"], cfg["memory"]["headroom_gb"]
    for m in models:
        assert est[m] + head < 16.0, m


def test_ollama_root() -> None:
    assert ollama_root("http://localhost:11434/v1") == "http://localhost:11434"


def test_implicit_latest_tag_is_the_same_model() -> None:
    ollama = FakeOllama(["nomic-embed-text:latest"])
    ollama.guard(free=0.5).prepare("nomic-embed-text")
    assert ollama.unloaded == []
