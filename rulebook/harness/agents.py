"""Agent definition and loading from `agents.yaml`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from rulebook.config import ROOT, agent_temperature, context_window, load_config

AGENTS_PATH = ROOT / "agents.yaml"


@dataclass
class Agent:
    name: str
    system_prompt: str
    model: str
    tools: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    max_steps: int = 10
    budget: int | None = None  # token budget per call; see context.py
    output_schema: type[BaseModel] | None = None
    temperature: float | None = None  # from config.yaml `temperature`; None = model default
    preload_skills: bool = False  # put full skill text in the system prompt (no load_skill)


class AgentConfigError(ValueError):
    pass


def load_specs(path: Path = AGENTS_PATH) -> dict[str, dict[str, Any]]:
    """Raw agent entries with defaults applied."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    defaults = raw.get("defaults") or {}
    return {name: {**defaults, **(spec or {})} for name, spec in raw["agents"].items()}


def resolve_model(key: str) -> str:
    node: Any = load_config()["models"]
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise AgentConfigError(f"model key {key!r} not found in config.yaml `models`")
        node = node[part]
    if not isinstance(node, str):
        raise AgentConfigError(f"model key {key!r} does not name a model")
    return node


def build_agent(
    name: str,
    system_prompt: str | None = None,
    output_schema: type[BaseModel] | None = None,
    path: Path = AGENTS_PATH,
) -> Agent:
    specs = load_specs(path)
    if name not in specs:
        raise AgentConfigError(f"unknown agent {name!r}; have {sorted(specs)}")
    spec = specs[name]
    prompt = system_prompt or spec.get("system_prompt")
    if not prompt:
        raise AgentConfigError(f"agent {name!r} has no system prompt yet")
    if spec.get("output_schema") and output_schema is None:
        raise AgentConfigError(f"agent {name!r} needs its {spec['output_schema']} schema")
    model = resolve_model(spec["model"])
    budget = spec.get("budget")
    window = context_window(model)
    if budget and budget >= window:
        raise AgentConfigError(
            f"agent {name!r}: budget {budget} must be below {model}'s context window {window}"
        )
    return Agent(
        name=name,
        system_prompt=prompt,
        model=model,
        tools=list(spec.get("tools") or []),
        skills=list(spec.get("skills") or []),
        max_steps=spec["max_steps"],
        budget=budget,
        output_schema=output_schema,
        temperature=agent_temperature(name),
        preload_skills=bool(spec.get("preload_skills", False)),
    )
