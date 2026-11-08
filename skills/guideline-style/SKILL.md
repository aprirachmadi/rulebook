---
name: guideline-style
description: How a new guideline rule (patch) must be written. Load before writing a rule.
---
# Guideline rule style

The guideline has one section per label group (1 Menu items, 2 Subtotal, 3 Total and
payment, 4 Other labels). A new rule is added to one section and gets the next number there
(e.g. 1.7); the system assigns the number.

A rule has:
- **rule_text**: 1-3 short sentences. Say which label to use and when, using exact label
  names (e.g. `menu.cnt`). Describe the text pattern in general terms, not a specific receipt.
- **examples**: 3-5 items from the cluster you were shown (by item id). Only from that list.
- **rationale**: one sentence on why the current guideline leads to disagreement.

Rules must not contradict existing rules. Prefer rules that say how to tell two similar
labels apart (e.g. a count vs an item number), because that is where labelers disagree.
Do not invent labels; use only labels from the guideline's label list.
