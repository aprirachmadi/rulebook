"""Disagreements -> spans -> clusters (SPEC section 2, decisions 2026-11-08).

A word is accepted when every labeler voted the same label at >= `accept_min_confidence`.
Other words are disagreements; consecutive disagreement words in one line with the same vote
pattern form a span (one cluster item). Items are grouped by (label set, normalized text);
big groups, or too many tiny groups, are re-clustered with embeddings (average-linkage
agglomerative clustering on cosine distance). Reads labelers' votes only, never v2.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from rulebook.data.store import Vote
from rulebook.tools.labels import LabelDoc

Embed = Callable[[list[str]], np.ndarray]


@dataclass
class Item:
    receipt_id: str
    line_idx: int
    words: list[int]
    text: str
    line_text: str
    votes: dict[str, Vote]  # agent -> (label, confidence, reason) for the span's first word
    labels: tuple[str, ...] = ()  # sorted distinct labels voted
    pattern: str = ""

    @property
    def item_id(self) -> str:
        return f"{self.receipt_id}:{self.line_idx}:{self.words[0]}-{self.words[-1]}"


@dataclass
class Cluster:
    cluster_id: int
    items: list[Item]
    labels: tuple[str, ...]
    pattern: str
    visible: list[Item] = field(default_factory=list)  # what the rule writer sees
    held: list[Item] = field(default_factory=list)  # what the verifier scores on
    held_out: bool = False


def normalize(text: str) -> str:
    """Lowercase, digits -> '#', collapse spaces: 'DISC 10%' -> 'disc ##%'."""
    return re.sub(r"\s+", " ", re.sub(r"\d", "#", text.lower())).strip()


def disagreement_spans(
    docs: list[LabelDoc],
    votes: dict[tuple[str, int, int], dict[str, Vote]],
    agents: list[str],
    min_conf: float,
) -> tuple[int, list[Item]]:
    """Returns (number of accepted words, disagreement spans)."""
    accepted = 0
    items: list[Item] = []
    for d in docs:
        for ln in d.lines:
            current: Item | None = None
            current_pattern: tuple[str, ...] | None = None
            for wi, w in enumerate(ln.words):
                v = votes.get((d.receipt_id, ln.line_idx, wi), {})
                labs = tuple(v[a][0] if a in v else "-" for a in agents)
                if (len(v) == len(agents) and len(set(labs)) == 1
                        and all(v[a][1] >= min_conf for a in agents)):
                    accepted += 1
                    current, current_pattern = None, None
                    continue
                if current is not None and labs == current_pattern and current.words[-1] == wi - 1:
                    current.words.append(wi)
                    current.text += " " + w.text
                    continue
                current = Item(d.receipt_id, ln.line_idx, [wi], w.text, ln.text, dict(v))
                current_pattern = labs
                items.append(current)
    for it in items:
        it.labels = tuple(sorted({lab for lab, _, _ in it.votes.values()}))
        it.pattern = normalize(it.text)
    return accepted, items


def agglomerative(vectors: np.ndarray, threshold: float) -> list[int]:
    """Average-linkage clustering on cosine distance; stop when the closest pair is farther
    than `threshold`. Returns a cluster index per row."""
    n = len(vectors)
    if n <= 1:
        return [0] * n
    x = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    dist = 1.0 - x @ x.T
    np.fill_diagonal(dist, np.inf)
    members = {i: [i] for i in range(n)}
    active = list(range(n))
    while len(active) > 1:
        sub = dist[np.ix_(active, active)]
        k = int(np.argmin(sub))
        a, b = active[k // len(active)], active[k % len(active)]
        if dist[a, b] > threshold:
            break
        na, nb = len(members[a]), len(members[b])
        for c in active:  # Lance-Williams update for average linkage
            if c not in (a, b):
                dist[a, c] = dist[c, a] = (na * dist[a, c] + nb * dist[b, c]) / (na + nb)
        members[a] += members.pop(b)
        active.remove(b)
        dist[b, :] = dist[:, b] = np.inf
    out = [0] * n
    for idx, (_, rows) in enumerate(sorted(members.items())):
        for r in rows:
            out[r] = idx
    return out


def _split(items: list[Item], embed: Embed, threshold: float) -> list[list[Item]]:
    labels = agglomerative(embed([it.text for it in items]), threshold)
    groups: dict[int, list[Item]] = {}
    for it, g in zip(items, labels, strict=True):
        groups.setdefault(g, []).append(it)
    return list(groups.values())


def cluster_items(items: list[Item], embed: Embed, cfg: dict) -> tuple[list[list[Item]], dict]:
    """Group by (label set, pattern); re-cluster big groups and, if too many groups are tiny,
    the tiny ones (within the same label set). Largest first. Returns (groups, stats)."""
    keyed: dict[tuple[tuple[str, ...], str], list[Item]] = {}
    for it in items:
        keyed.setdefault((it.labels, it.pattern), []).append(it)
    groups = list(keyed.values())
    tiny = [g for g in groups if len(g) < cfg["tiny_group_size"]]
    big = [g for g in groups if len(g) > cfg["max_group_size"]]
    stats = {"key_groups": len(groups), "tiny_groups": len(tiny), "big_groups": len(big),
             "resplit_big": bool(big), "resplit_tiny": False}

    out = [g for g in groups if g not in big and g not in tiny]
    for g in big:
        out += _split(g, embed, cfg["distance_threshold"])
    if groups and len(tiny) / len(groups) > cfg["tiny_share_resplit"]:
        stats["resplit_tiny"] = True
        by_labels: dict[tuple[str, ...], list[Item]] = {}
        for g in tiny:
            for it in g:
                by_labels.setdefault(it.labels, []).append(it)
        for same in by_labels.values():
            out += _split(same, embed, cfg["distance_threshold"])
    else:
        out += tiny
    out.sort(key=lambda g: (-len(g), g[0].item_id))
    return out, stats


def make_clusters(groups: list[list[Item]], cfg: dict) -> tuple[list[Cluster], list[list[Item]]]:
    """Top N groups with >= min size become clusters (with a hold-out split); the rest go to
    the review queue."""
    big_enough = [g for g in groups if len(g) >= cfg["min_cluster_for_rule"]]
    chosen = big_enough[: cfg["max_clusters_per_run"]]
    rest = [g for g in groups if not any(g is c for c in chosen)]
    clusters = []
    rng = random.Random(cfg["seed"])
    for i, g in enumerate(chosen):
        c = Cluster(i, g, g[0].labels, g[0].pattern)
        if len(g) >= cfg["min_size_for_holdout"]:
            shuffled = sorted(g, key=lambda it: it.item_id)
            rng.shuffle(shuffled)
            n_held = max(1, round(len(g) * cfg["holdout_fraction"]))
            c.held, c.visible, c.held_out = shuffled[:n_held], shuffled[n_held:], True
        else:
            c.visible, c.held, c.held_out = list(g), list(g), False
        clusters.append(c)
    return clusters, rest
