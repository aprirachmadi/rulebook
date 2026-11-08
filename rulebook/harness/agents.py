"""Agent definition. Loading from `agents.yaml` comes with Milestone 3."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel


@dataclass
class Agent:
    name: str
    system_prompt: str
    model: str
    tools: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    max_steps: int = 10
    budget: int | None = None  # token budget; enforced by context.py (Milestone 3)
    output_schema: type[BaseModel] | None = None
    temperature: float | None = None  # from config.yaml `temperature`; None = model default
