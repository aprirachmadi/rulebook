# Progress

## Done
- SPEC.md saved from the pasted spec.
- Poetry env on Python 3.12; Milestone 1 deps: datasets, pandas, pydantic, typer, rich (dev: pytest, ruff).
- Package skeleton: `rulebook/`, `rulebook/data/`, `rulebook/pipeline/`, `eval/`, `tests/`.
- Inspected CORD v1/v2. Receipts match 1:1; words match exactly on (quad, text). See `reports/m1_gate_findings.md`.

## In progress
- Milestone 1 is **paused at the decision gate**: only 18 lines change label between v1 and v2 (gate is ~100).

## Next (after the gate decision)
- `rulebook/data/cord.py`: loaders and line extraction.
- Diff script writing `reports/v1_v2_diff.csv` and its summary.
- Leakage guard (`eval.load_answer_key()`) and a leakage test.

## Open questions
- Gate fallback: what to use as the answer key for label-error detection.
- Should the `menu.sub_*` -> `menu.sub.*` rename count as a label change? (Assumed no in the exploratory numbers.)
- Images are read as raw bytes (Pillow is not on the allowed list). Pillow will be needed by Milestone 4 (PaddleOCR / crops).
- `config.yaml` needs a YAML parser; PyYAML is not on the allowed list (it is already installed as a dependency of `datasets`).
- None of the spec's default Ollama models are installed (needed from Milestone 2/4).
