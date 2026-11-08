# Number reading policies vs the sum check (2026-11-08)

One-off analysis (script kept out of the repo). Line roles (item, subtotal, tax, total) come from v2 labels;
this is an offline design check, not something the pipeline does.

The only ambiguous shape is a single `.` or `,` followed by exactly 3 digits (`2.000`, `75,000`).
Every other CORD shape (`10.500,00`, `44,000.00`, `1,346,000`, `0,00`, `116000.00`, `10.00 %`) reads the same either way.

88 of 800 train receipts mix both thousands styles inside one receipt (e.g. `TOTAL 49,636` next to `TUNAI 54.600`).

| policy | dev: sum check passes (of 99 checkable) | train (of 786) |
|---|---|---|
| whole receipt `.` = thousands | 79 | 643 |
| whole receipt `,` = thousands | 83 | 676 |
| per token: lone separator + 3 digits = thousands, either symbol | **89** | **721** |

Per-receipt rule "pick the reading that makes the sum check pass, else ambiguous" on dev:
- 77 receipts: both readings pass with different values (scale x1000 cancels out) -> ambiguous
- 16 receipts: neither reading passes -> ambiguous
- 4 decided, 2 both pass with the same values
=> 93 of 99 dev receipts would be "ambiguous".
