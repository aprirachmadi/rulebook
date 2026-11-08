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
| 2026-11-08 | Label space from train category names | Fine (names only) |
| 2026-11-08 | New M1 finish line (`reports/cord_v2_summary.md` + tests) | Fine |
| 2026-11-08 | Gold-set access | One sanctioned function for the Verifier, gold (dev) receipts only; leakage test must fail if it can return train or test labels. 30 receipts, fixed seed, stratified for rare classes |
| 2026-11-08 | Guideline v1 source | Not from train labels. Class names + 1–2 examples from dev only (never test). Leave out discounts vs bill discounts, service charge, sub-items, void lines, etc fields. Show before use |
| 2026-11-08 | Reading order | User checks samples; same ordering for OCR-vs-CORD text comparison |
| 2026-11-08 | Run sizes | Dev runs 20–50 receipts; full 800 only for the final report |
| 2026-11-08 | Rare classes | Report overall accuracy + macro-F1; flag classes with < 5 eval lines as "too few to score" |
| 2026-11-08 | Suspect-v2 definition | All 3 labelers at confidence >= 0.7, none matches v2, ranked by average confidence |
| 2026-11-08 | Models | Report GPU/VRAM and propose; wait for OK |
| 2026-11-08 | Repo on NTFS (fuseblk) drive | Not an issue (single machine, small data). Keep as is; revisit only if SQLite locking problems show up in M7 |
| 2026-11-08 | Labeler models | `qwen3.5:4b`, `nemotron-3-nano:4b`, `llama3.2:3b` (three families, ~8–9 GB, all fit in VRAM at once). User pulling them. Orchestrator / vision / embedding models still open |
| 2026-11-08 | Guideline v1 | Approved as a starting draft; revisit after the first real run |
| 2026-11-08 | Verifier | Simple config defaults: reject if gold accuracy drops > 1%; patches on classes with no gold lines marked "not gold-verified"; nothing more for now |
| 2026-11-08 | Tilted two-row lines | No fix |
| 2026-11-08 | Orchestrator / diagnoser / rule writer / vision / toy-agent models | User left it to Claude ("<your decision>"): `gemma4:12b` for orchestrator, diagnoser, rule writer and vision; `qwen3.5:4b` for the M2 toy agent; `nomic-embed-text` kept |
| 2026-11-08 | Tuning questions | Log under "revisit at M8" in PROGRESS.md instead of asking, unless blocking |
| 2026-11-08 | Temperature | Per agent in config.yaml: 0 for labelers and verifier, 0.2 for rule writer, diagnoser, orchestrator (toy agent 0) |
| 2026-11-08 | Number rule | All Indonesian number parsing in code; tools return parsed numbers; models never parse prices or do arithmetic. Handle both CORD formats; return "ambiguous" instead of guessing. Added to SPEC and CLAUDE.md |
| 2026-11-08 | Model choices | Approved; locked before the final runs |
