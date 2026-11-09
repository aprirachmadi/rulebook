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


def sync_skill(version: str | None = None, path: Path | None = None) -> str:
    """Write the `cord-guideline` skill for `version` (default: current). Returns the version."""
    version = version or current_version()
    path = path or SKILL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\nname: cord-guideline\n"
        f"description: The receipt labeling guideline, version {version}. "
        "Label definitions and examples.\n---\n"
        f"{guideline_text(version)}\n",
        encoding="utf-8",
    )
    return version


def sections(text: str) -> list[str]:
    """Section numbers from '## N. Title' headings."""
    return re.findall(r"^## (\d+)\. ", text, flags=re.MULTILINE)


def next_rule_number(text: str, section: str, taken: set[str] | None = None) -> str:
    """Next free 'section.k' given the rules in `text` and numbers already handed out."""
    used = {int(k) for k in re.findall(rf"^### {re.escape(section)}\.(\d+) ", text,
                                       flags=re.MULTILINE)}
    used |= {int(n.split(".")[1]) for n in (taken or set()) if n.split(".")[0] == section}
    return f"{section}.{max(used, default=0) + 1}"


def render_patch(number: str, rule_text: str, examples: list[str]) -> str:
    ex = "; ".join(f"`{e}`" for e in examples)
    return f"### {number} Rule\n{rule_text.strip()}\nExamples: {ex}\n"


def with_patches(text: str, patches: list[tuple[str, str]]) -> str:
    """Insert rendered rules (section, block) at the end of their sections."""
    for section, block in patches:
        heading = re.search(rf"^## {re.escape(section)}\. .*$", text, flags=re.MULTILINE)
        if heading is None:
            raise ValueError(f"guideline has no section {section}")
        nxt = re.search(r"^## ", text[heading.end():], flags=re.MULTILINE)
        at = heading.end() + nxt.start() if nxt else len(text)
        text = text[:at].rstrip() + "\n\n" + block + "\n" + text[at:]
    return text


def write_skill_text(text: str, version_label: str, root: Path) -> Path:
    """A cord-guideline skill with custom text (e.g. guideline + candidate patch) under root."""
    path = root / "cord-guideline" / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\nname: cord-guideline\n"
        f"description: The receipt labeling guideline, version {version_label}. "
        "Label definitions and examples.\n---\n" + text.strip() + "\n",
        encoding="utf-8",
    )
    return root


def publish_version(section: str, block: str, note: str) -> str:
    """Write guidelines/v<N+1>.md = current version + this rule, sync the skill, return the
    new version. The changelog line goes in an HTML comment (labelers never see it)."""
    prev = current_version()
    new = f"v{int(prev[1:]) + 1}"
    old_log = re.findall(r"<!-- changelog: (.*?) -->",
                         (GUIDELINES / f"{prev}.md").read_text(encoding="utf-8"))
    log = [*old_log, f"{new}: {note}"]
    header = "\n".join(f"<!-- changelog: {line} -->" for line in log)
    text = with_patches(guideline_text(prev), [(section, block)])
    (GUIDELINES / f"{new}.md").write_text(f"{header}\n{text}\n", encoding="utf-8")
    sync_skill(new)
    return new


def changelog(version: str | None = None) -> list[str]:
    version = version or current_version()
    return re.findall(r"<!-- changelog: (.*?) -->",
                      (GUIDELINES / f"{version}.md").read_text(encoding="utf-8"))
