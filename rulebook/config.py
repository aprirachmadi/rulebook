"""Load `config.yaml`."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


@cache
def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    if base_url := os.environ.get("OLLAMA_BASE_URL"):
        cfg["llm"]["base_url"] = base_url
    return cfg


def agent_temperature(agent_name: str) -> float | None:
    """Temperature for an agent from `config.yaml`; None means the model's default."""
    return (load_config().get("temperature") or {}).get(agent_name)
