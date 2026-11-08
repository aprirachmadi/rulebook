"""Guideline versions (`guidelines/v<N>.md`) and the `cord-guideline` skill built from them."""

from __future__ import annotations

import re
from pathlib import Path

from rulebook.config import ROOT

GUIDELINES = ROOT / "guidelines"
SKILL_PATH = ROOT / "skills" / "cord-guideline" / "SKILL.md"


def versions() -> list[str]:
    nums = sorted(int(m.group(1)) for p in GUIDELINES.glob("v*.md")
                  if (m := re.fullmatch(r"v(\d+)\.md", p.name)))
    return [f"v{n}" for n in nums]


def current_version() -> str:
    return versions()[-1]


def guideline_text(version: str) -> str:
    """The guideline as labelers see it: the file without HTML comments (provenance notes)."""
    text = (GUIDELINES / f"{version}.md").read_text(encoding="utf-8")
    return re.sub(r"<!--.*?-->\s*", "", text, flags=re.DOTALL).strip()


def sync_skill(version: str | None = None, path: Path = SKILL_PATH) -> str:
    """Write the `cord-guideline` skill for `version` (default: current). Returns the version."""
    version = version or current_version()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\nname: cord-guideline\n"
        f"description: The receipt labeling guideline, version {version}. "
        "Label definitions and examples.\n---\n"
        f"{guideline_text(version)}\n",
        encoding="utf-8",
    )
    return version
