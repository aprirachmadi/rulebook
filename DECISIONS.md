# Decisions

| Date | Question | Answer |
|---|---|---|
| 2026-11-08 | Python version and interpreter | Python 3.12, `/usr/bin/python3.12` (same as sibling projects), `requires-python >=3.12,<3.13` |
| 2026-11-08 | torch | Not installed for now; nothing in the spec needs it. If it is ever needed: `2.10.0+cu128` from the cu128 index, as in crumb / geo-lens |
| 2026-11-08 | PaddleOCR versions | Pin `paddleocr==3.7.0`, `paddlepaddle==3.3.1` (as in crumb), added in Milestone 4 |
| 2026-11-08 | Git branch | Commit on `master` (no objection raised) |
