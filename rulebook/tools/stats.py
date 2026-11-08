"""Agreement statistics. Nothing here reads v2 labels."""

from __future__ import annotations

import krippendorff
import numpy as np


def _matrix(units: list[dict[str, str]], agents: list[str], value: dict[str, float]) -> np.ndarray:
    return np.array(
        [[value.get(u[a], np.nan) if a in u else np.nan for u in units] for a in agents],
        dtype=float,
    )


def alpha(units: list[dict[str, str]], agents: list[str]) -> float | None:
    """Krippendorff's alpha (nominal). `units` = one {agent: label} per word; missing = no vote."""
    labels = sorted({lab for u in units for lab in u.values()})
    if len(labels) < 2:
        return None
    m = _matrix(units, agents, {lab: i for i, lab in enumerate(labels)})
    if not ((~np.isnan(m)).sum(axis=0) >= 2).any():
        return None  # no word was labeled by two or more labelers
    return float(krippendorff.alpha(reliability_data=m, level_of_measurement="nominal"))


def alpha_per_label(units: list[dict[str, str]], agents: list[str]) -> dict[str, float | None]:
    """One-vs-rest alpha per label: agreement on 'is it this label or not'."""
    out: dict[str, float | None] = {}
    for lab in sorted({lab for u in units for lab in u.values()}):
        binary = [{a: ("yes" if v == lab else "no") for a, v in u.items()} for u in units]
        out[lab] = alpha(binary, agents)
    return out


def agreement_pattern(units: list[dict[str, str]], agents: list[str]) -> dict[str, int]:
    """How many words got 3/3, 2/3 or no agreement (only words every agent voted on)."""
    counts = {"all_agree": 0, "two_agree": 0, "all_differ": 0, "incomplete": 0}
    for u in units:
        if any(a not in u for a in agents):
            counts["incomplete"] += 1
            continue
        distinct = len(set(u[a] for a in agents))
        key = {1: "all_agree", 2: "two_agree"}.get(distinct, "all_differ")
        counts[key] += 1
    return counts
