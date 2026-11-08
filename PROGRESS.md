# Progress

## Done
- SPEC.md saved, then updated for the v1 drop (v2 only, labeling from scratch, no error-detection metric, suspect-v2 list, Pillow + PyYAML allowed).
- Poetry env on Python 3.12. Deps: datasets, pandas, pydantic, typer, rich, pillow, pyyaml (dev: pytest, ruff).
- Inspected CORD v1/v2 (gate findings: `reports/m1_gate_findings.md`). v1 since dropped.
- **Milestone 1 (data)**, ready for review:
  - `config.yaml` + `rulebook/config.py` (`OLLAMA_BASE_URL` overrides the endpoint).
  - `rulebook/data/cord.py`: v2 loader; label-free `Receipt` / `Line` / `Word`; line text in reading order (rows by mean top edge, then x); line box = union of word boxes; `load_image`; `label_space()` (29 names, from the work split).
  - `eval/answer_key.py`: `load_answer_key(split)`, the only per-line label accessor.
  - `eval/data_summary.py` -> `reports/cord_v2_summary.md`.
  - Tests: 9 passing, including leakage tests (no label fields on agent-facing models or loaded data; static scan that nothing in `rulebook/` except `data/cord.py` reads raw ground truth or categories, and nothing imports `eval/`).

- Milestone 1 approved (2026-11-08). Follow-ups done:
  - Gold set: `rulebook/data/gold.py` (`select_gold_receipts`, `gold_labels`); 30 dev receipts (seed 0, all 23 dev classes covered, 340 lines) frozen in `config.yaml`.
  - Leakage tests extended: gold accessor refuses train/test ids and a non-dev gold split; gold.py may only read the dev split; only allowed modules may call `gold_labels`. 16 tests pass.
  - `guidelines/v1.md` drafted (DRAFT, not approved): 15 classes defined, 14 listed by name only.
  - Reading order checked: only 2 lines change text vs CORD's own word order.

## In progress
- Waiting for: guideline v1 approval, model choice, drive decision.

## Next
- Milestone 2: harness core (`model.py`, `tools.py`, `loop.py`, `trace.py`) + calculator toy agent.

## Open questions
- Very rare classes: `void_menu.nm`, `void_menu.price`, `menu.itemsubtotal` have 1 train line each.
- Models: proposal sent, waiting for OK (needed from M2).
- Repo is on an NTFS (fuseblk) partition: keep, or move runtime files (db, runs) / the repo to ext4?
- Two-row lines on tilted receipts read value-first (`2.000,00 Change:`), same as CORD's order. Fix or keep?
