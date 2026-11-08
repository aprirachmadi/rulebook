"""VRAM guard: one model loaded at a time, and never load a model that doesn't fit.

Out-of-memory on this machine can take the whole desktop down, so before a model's first
call we (1) unload every other model Ollama has loaded, then (2) check free VRAM against the
model's estimate (`config.yaml` `memory.estimate_gb`) plus headroom. If it doesn't fit, refuse.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

import httpx

from rulebook.config import load_config


class MemoryGuardError(RuntimeError):
    pass


def ollama_root(base_url: str) -> str:
    """`http://host:11434/v1` -> `http://host:11434` (the native API)."""
    return base_url.rstrip("/").removesuffix("/v1")


def free_vram_gb() -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout
    return min(float(x) for x in out.split()) / 1024  # MiB -> GiB; smallest GPU if several


class MemoryGuard:
    def __init__(
        self,
        http: httpx.Client | None = None,
        free_gb: Callable[[], float] = free_vram_gb,
    ) -> None:
        cfg = load_config()
        self.mem = cfg["memory"]
        self.http = http or httpx.Client(base_url=ollama_root(cfg["llm"]["base_url"]), timeout=120)
        self.free_gb = free_gb
        self.ready: set[str] = set()

    def loaded(self) -> list[str]:
        r = self.http.get("/api/ps")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    @staticmethod
    def same(a: str, b: str) -> bool:
        """'nomic-embed-text' and 'nomic-embed-text:latest' are the same model."""
        tag = lambda m: m if ":" in m else f"{m}:latest"  # noqa: E731
        return tag(a) == tag(b)

    def unload(self, model: str) -> None:
        r = self.http.post("/api/generate", json={"model": model, "keep_alive": 0})
        r.raise_for_status()

    def estimate_gb(self, model: str) -> float:
        est = self.mem["estimate_gb"]
        if model not in est:
            raise MemoryGuardError(
                f"no VRAM estimate for {model!r} in config.yaml memory.estimate_gb; "
                "calculate it before running this model"
            )
        return float(est[model])

    def prepare(self, model: str) -> None:
        """Call before using `model`. Unloads others; refuses if it would not fit."""
        loaded = self.loaded()
        if len(loaded) == 1 and self.same(loaded[0], model):
            self.ready.add(model)
            return
        for other in loaded:
            if not self.same(other, model):
                self.unload(other)
        if any(self.same(m, model) for m in self.loaded()):
            self.ready.add(model)
            return
        need = self.estimate_gb(model) + float(self.mem["headroom_gb"])
        free = self.free_gb()
        if free < need:
            raise MemoryGuardError(
                f"{model} needs ~{need:.1f} GB VRAM (estimate + headroom) but only "
                f"{free:.1f} GB is free; not loading it"
            )
        self.ready.add(model)


_guard: MemoryGuard | None = None


def default_guard() -> MemoryGuard:
    global _guard
    if _guard is None:
        _guard = MemoryGuard()
    return _guard
