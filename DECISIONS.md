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
| 2026-11-08 | Number resolution | Per token: a lone separator + exactly 3 digits = thousands. Lines that fail the sum check are flagged, never guessed |
| 2026-11-08 | Context windows | 16k (`gemma4:12b-16k`) for orchestrator, diagnoser, rule writer, verifier. Labelers one model at a time with 8k (VRAM allows: ~3.9–4.7 GB each vs 14.8 GB free) |
| 2026-11-08 | Prompt size check | Warn before a labeler call when receipt + guideline + skills exceed 80% of the budget |
| 2026-11-08 | ocr_agent temperature | 0 |
| 2026-11-08 | Memory safety | Calculate VRAM before running any model; one model loaded at a time; refuse a load that doesn't fit (user: OOM can kill the computer) |
| 2026-11-08 | OCR install | paddleocr 3.7.0 + paddlepaddle 3.3.1 CPU (no CUDA, no torch), PyYAML loosened to >=6.0.2 for paddlex; PP-OCRv6 small det/rec copied from crumb; offline (model source check disabled) |
| 2026-11-08 | M4 A: OCR setup | Approved as above |
| 2026-11-08 | M4 B: labeling unit | Labelers label word spans inside OCR lines: input = each OCR line with numbered words; output = [{line_idx, words: [indices], label, confidence, reason}]; every word exactly one label (validate, one repair retry). Scoring and alpha at word level: OCR word -> CORD word by center-in-box, then text similarity; that word's v2 label. Unmatched words = OCR misses. Same format in gold-text mode with CORD's words |
| 2026-11-08 | M4 C: OCR agent | Plain code in M4; vision re-read agent in M5 |
| 2026-11-08 | M4 D: receipt selection | First N by id for dev runs; final report uses a fixed-seed random sample |
| 2026-11-08 | M4 E: labeler input | Line index, numbered words, rough position (0–100), answer format as in B |
| 2026-11-08 | Embedding model | Pull nomic-embed-text |
| 2026-11-08 | After first 20-receipt run | Show word-level alpha and each labeler's agreement with v2 before building anything else |
| 2026-11-08 | First 20-receipt run: labeler failures | Fix before considering other models: add an "other" label (text CORD doesn't label: store name, date, address, cashier, footer); words labeled "other" excluded from accuracy vs v2, report each labeler's use of it |
| 2026-11-08 | Chunking (spec change) | ~20 lines per labeler call; "reason" only when confidence < 0.7 |
| 2026-11-08 | Repair | Targeted repair approved (re-ask only bad / missing words); fix the per-line word index bound |
| 2026-11-08 | Labeler models after the 20-receipt runs | Replace nemotron-3-nano:4b (labeler_b) with llama3.1:8b and llama3.2:3b (labeler_c) with gemma4:12b (installed), both 8k tags; user pulled llama3.1:8b. Rerun the same 20 receipts |
| 2026-11-08 | qwen identical numbers | Rerun, not reused (33 fresh calls; identical because temperature 0 and identical inputs) |
| 2026-11-08 | "other" in the labeler prompt | Only for text outside the order (store name, address, date, cashier, footer); item names, counts, prices, subtotals and totals always get a real label, low confidence when unsure. Applies from Milestone 5's first run |
| 2026-11-08 | Headline labeling metric | Agreement with v2 over CORD-matched words, counting "other" on a matched word as wrong. Excl-other stays secondary. Flag any labeler under 75% at M5's first run |
| 2026-11-08 | Run reports | Every run report shows all-agree -> % matches v2 and majority -> % matches v2 |
| 2026-11-08 | qwen-only rerun | Not done (user: no rerun); --labelers option kept, unused |
| 2026-11-08 | M5 plan | Approved. Q1: cluster disagreement spans (consecutive words, same vote pattern) by normalized text. Q2: re-split with embeddings when >50% of groups are tiny (<3). Q3: develop on train_0000-0049, one 100-receipt run for acceptance. Q4: OCR text for work batch and gold set. Q5: verify every GUIDELINE_GAP patch, report timings |
| 2026-11-08 | Cluster hold-out | Rule writer sees ~70% of a cluster (examples only from it); verifier measures cluster alpha gain on the held-out ~30%. Clusters too small to split are marked "not held out" |
| 2026-11-08 | Clusters per run | Process only the top N clusters by size (N = 8, config); the rest stay in the review queue |
| 2026-11-08 | Timings | Report per stage (cluster, diagnose, write, verify) after the 50-receipt dev run |
