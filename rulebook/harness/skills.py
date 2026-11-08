"""Skills: `skills/<name>/SKILL.md` with `name` and `description` in YAML frontmatter.

Agents always see the name + description of their allowed skills; `load_skill(name)` returns
the full text, and only for skills on the agent's allowlist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from rulebook.config import ROOT


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path


def parse_skill(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path}: missing frontmatter")
    _, front, body = text.split("---", 2)
    meta = yaml.safe_load(front) or {}
    for key in ("name", "description"):
        if not meta.get(key):
            raise ValueError(f"{path}: frontmatter needs {key!r}")
    if meta["name"] != path.parent.name:
        raise ValueError(f"{path}: name {meta['name']!r} must match its folder")
    return Skill(meta["name"], str(meta["description"]).strip(), body.strip(), path)


class SkillLibrary:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or ROOT / "skills"
        self.skills: dict[str, Skill] = {}
        if self.root.exists():
            for p in sorted(self.root.glob("*/SKILL.md")):
                s = parse_skill(p)
                self.skills[s.name] = s

    def allowed(self, allowlist: list[str]) -> list[Skill]:
        missing = [n for n in allowlist if n not in self.skills]
        if missing:
            raise KeyError(f"unknown skills {missing}; have {sorted(self.skills)}")
        return [self.skills[n] for n in allowlist]

    def catalog(self, allowlist: list[str]) -> str:
        """The block appended to an agent's system prompt."""
        items = "\n".join(f"- {s.name}: {s.description}" for s in self.allowed(allowlist))
        return (
            "Skills you can load with the load_skill tool (call it before you need one):\n"
            + items
        )

    def load(self, name: str, allowlist: list[str]) -> str:
        if name not in allowlist:
            return f"error: skill {name!r} is not available to you. Available: {allowlist}"
        return self.skills[name].body
