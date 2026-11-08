# Milestone 1 decision gate: CORD v1 vs v2 (2026-11-08)

Exploratory numbers from one-off scripts (not yet the final `reports/v1_v2_diff.csv`).
Datasets: `naver-clova-ix/cord-v1`, `naver-clova-ix/cord-v2` (v2 ID verified on the Hub).

## Receipt matching
- Both have 800 train / 100 validation / 100 test.
- Same image bytes (sha1) and same `meta.image_id` at the same index in every split, so receipts match 1:1 trivially.

## Line / word matching
- v1 lines: 11,107 / 1,247 / 1,336 (train/val/test). v2: 10,987 / 1,219 / 1,309.
- Words: 23,907 match exactly on (quad, text); 2 match by IoU >= 0.5 with different text; 4 have no match.
- Every v1 line maps to exactly one v2 line (13,686), except 4 lines with no match.
- v2 merges v1 lines: 115 / 28 / 26 v2 lines are made of more than one v1 line (for example v1 `"1"` + `"x"` becomes v2 `"1 x"`).

## Label changes
- Label space: v1 has 30 categories, v2 has 29. v1 `menu.sub_{nm,cnt,price,unitprice}` were renamed to `menu.sub.{nm,cnt,price,unitprice}` in v2. v1 `menu.sub_etc` does not exist in v2.
- If the rename counts as a change: 915 / 179 / 119 words differ, almost all of them the rename.
- With the rename mapped (`menu.sub_X` -> `menu.sub.X`), **18 v1 lines change label in total**: about 16 train, 1 val, 1 test. They are spread over 6 / 1 / 1 receipts.

| count | v1 label | v2 label |
|---|---|---|
| 6 | menu.cnt | menu.sub.cnt |
| 4 | menu.sub_etc | menu.sub.nm |
| 3 | menu.price | menu.sub.price |
| 2 | total.total_price | menu.price |
| 2 | total.total_etc | total.emoneyprice |
| 1 | sub_total.subtotal_price | menu.price |

## Conclusion
18 changed lines is far below the ~100 gate, and the eval (test) split has about 1.
v2 is mostly a renaming and line-merging release, not a label-correction release.
