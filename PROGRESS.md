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

## In progress
- Nothing. Waiting for the Milestone 1 review.

## Next
- Milestone 2: harness core (`model.py`, `tools.py`, `loop.py`, `trace.py`) + calculator toy agent.

## Open questions
- Gold set: which 30 of the 100 dev receipts (needed by M5).
- "Confidently disagree with v2" for the suspect-v2 list: exact definition (needed after the first run).
- Label space is derived from category names in the work split's v2 labels (names only, not per-line). OK under the leakage rule?
- Very rare classes: `void_menu.nm`, `void_menu.price`, `menu.itemsubtotal` have 1 train line each.
- None of the spec's default Ollama models are installed (needed from M2).
