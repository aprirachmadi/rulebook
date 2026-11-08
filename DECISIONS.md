# Decisions

| Date | Question | Answer |
|---|---|---|
| 2026-11-08 | Python version and interpreter | Python 3.12, `/usr/bin/python3.12` (same as sibling projects), `requires-python >=3.12,<3.13` |
| 2026-11-08 | torch | Not installed for now; nothing in the spec needs it. If it is ever needed: `2.10.0+cu128` from the cu128 index, as in crumb / geo-lens |
| 2026-11-08 | PaddleOCR versions | Pin `paddleocr==3.7.0`, `paddlepaddle==3.3.1` (as in crumb), added in Milestone 4 |
| 2026-11-08 | Git branch | Commit on `master` (no objection raised) |
| 2026-11-08 | M1 decision gate: v1 vs v2 differ on only 18 lines | Drop CORD v1. v2 is the only label source and the answer key. Gate findings kept in `reports/m1_gate_findings.md` |
| 2026-11-08 | Noise injection for label-error detection | Not now. No label-error-detection metric for now |
| 2026-11-08 | Scope | Labeling from scratch: agents label blind from image / OCR text + guideline |
| 2026-11-08 | Evaluation | Accuracy vs v2, agreement per guideline version, % sent to human, OCR metrics, ablation |
| 2026-11-08 | Checking v2 itself | After a run, list the top 50 lines where all 3 labelers confidently disagree with v2, for a hand check |
| 2026-11-08 | Pillow, PyYAML | Approved; added with Poetry and to the SPEC allowed list |
