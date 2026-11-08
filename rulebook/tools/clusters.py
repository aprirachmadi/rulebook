"""Disagreements -> spans -> clusters (SPEC section 2, decisions 2026-11-08).

A word is accepted when every labeler voted the same label at >= `accept_min_confidence`.
Other words are disagreements; consecutive disagreement words in one line with the same vote
pattern form a span (one cluster item). Spans where every labeler voted the same label but
someone was below the confidence bar go to the review queue as "low confidence". The rest are
grouped by label set; a label set with more than `max_group_size` items is split with
embeddings (average-linkage agglomerative clustering on cosine distance). Reads labelers'
votes only, never v2.
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


def split_low_confidence(items: list[Item], agents: list[str]) -> tuple[list[Item], list[Item]]:
    """(unanimous-but-low-confidence spans, real disagreements)."""
    low = [it for it in items if len(it.labels) == 1 and len(it.votes) == len(agents)]
    return low, [it for it in items if it not in low]


def top_patterns(items: list[Item], n: int = 3) -> str:
    """The most common normalized texts in a group: 'total / grand total / jumlah'."""
    counts: dict[str, int] = {}
    for it in items:
        counts[it.pattern] = counts.get(it.pattern, 0) + 1
    return " / ".join(sorted(counts, key=lambda p: (-counts[p], p))[:n])


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
    """Group by label set; split a label set bigger than `max_group_size` with embeddings.
    Largest first. Returns (groups, stats)."""
    by_labels: dict[tuple[str, ...], list[Item]] = {}
    for it in items:
        by_labels.setdefault(it.labels, []).append(it)
    groups = list(by_labels.values())
    big = [g for g in groups if len(g) > cfg["max_group_size"]]
    stats = {"label_sets": len(groups), "big_groups": len(big)}
    out = [g for g in groups if g not in big]
    for g in big:
        out += _split(g, embed, cfg["distance_threshold"])
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
        c = Cluster(i, g, g[0].labels, top_patterns(g))
        if len(g) >= cfg["min_size_for_holdout"]:
            shuffled = sorted(g, key=lambda it: it.item_id)
            rng.shuffle(shuffled)
            n_held = max(1, round(len(g) * cfg["holdout_fraction"]))
            c.held, c.visible, c.held_out = shuffled[:n_held], shuffled[n_held:], True
        else:
            c.visible, c.held, c.held_out = list(g), list(g), False
        clusters.append(c)
    return clusters, rest
