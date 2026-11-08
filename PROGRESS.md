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

- Guideline v1 approved as a starting draft. Models set in `config.yaml` (labelers by user; orchestrator / vision / toy agent `gemma4:12b` / `qwen3.5:4b` by Claude, left to Claude by user).
- **Milestone 2 (harness core)**, ready for review:
  - `rulebook/harness/model.py`: `ModelClient.chat(messages, tools, json_schema)` over httpx; retries with exponential backoff on network errors / 429 / 5xx (not other 4xx); native tool calls, with fallback parsing of `{"tool": ..., "args": ...}` blocks from text; `reasoning_effort` from config.
  - `rulebook/harness/tools.py`: `@tool` decorator (one pydantic args model), registry, `run()` validates args, catches exceptions, truncates results over `tool_result_max_chars`; `needs_approval` tools are refused until the M3 guard exists.
  - `rulebook/harness/trace.py`: `runs/<run_id>/trace.jsonl`, one record per model call / tool call / final / error; `read_trace()` validates every line.
  - `rulebook/harness/loop.py`: loop until no tool calls, `final_answer`, or `max_steps`; tool allowlist per agent; `output_schema` validation with one repair retry (schema enforced, no tools).
  - `rulebook/harness/agents.py`: `Agent` dataclass only (YAML loading in M3).
  - `rulebook/pipeline/toy.py` + `rulebook toy-agent` CLI: calculator (AST-restricted, no `eval`) on 5 tasks; expected answers computed by Python; a task counts as solved only if the answer is correct AND equals a number the calculator returned.
  - Tests: 37 passing (harness tests use a scripted fake endpoint).
  - Live acceptance on `qwen3.5:4b`: 5 runs, 22/25 tasks solved; 5/5 in 2 runs, 4/5 in 3. All 3 misses are Indonesian number format (`12.500` read as 12.5; `12550` reported as `12.55`); traces valid in every run (20–24 records).

## In progress
- Nothing. Waiting for the Milestone 2 review.

## Next
- Milestone 3: skills, context budget, approval guard; load agents from `agents.yaml`.

## Open questions
- Very rare classes: `void_menu.nm`, `void_menu.price`, `menu.itemsubtotal` have 1 train line each.
- `nemotron-3-nano:4b`, `llama3.2:3b`, `nomic-embed-text` not pulled yet (needed from M4 / M5).

## Revisit at M8
- Temperature: not set, so models use their defaults (qwen3.5: 1.0). Likely cause of the run-to-run variance in the toy agent; consider a `temperature` setting per role (e.g. low for labelers).
- `reasoning_effort: none` (thinking off) for all models. Thinking may help the diagnoser / rule writer; it costs ~50x tokens.
- Indonesian number format confuses small models (toy agent misses). Keep all price parsing in code (`sum_check`, export `value`), never ask a model to normalise numbers.
- Verifier thresholds (`max_gold_accuracy_drop` 0.01, `min_cluster_alpha_gain` 0.05).
- Guideline v1 content, after the first real run.
