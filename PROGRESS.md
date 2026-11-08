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

- Milestone 2 accepted (2026-11-08).
- Temperature per agent (`config.yaml` `temperature`). Toy agent at temperature 0: 5/5 in all 5 runs, and the 5 traces are identical (same tool calls and results), so it is one repeated sample, not 5 independent ones.
- Number rule in SPEC.md and CLAUDE.md. `rulebook/tools/receipt.py`: token parser with both readings, `amount_token`, `sum_check` (1% from config). 26 tests with real CORD strings. Policy comparison: `reports/number_policy_check.md`.

## In progress
- Milestone 3.
- Blocked on a decision: how to resolve ambiguous number tokens per receipt (see open questions). Not built until answered.

## Next
- Milestone 3: skills, context budget, approval guard; load agents from `agents.yaml`.

## Open questions
- **Number resolution policy (blocking the resolver only).** The per-receipt rule makes 93/99 dev receipts "ambiguous". Proposal: per token, lone separator + 3 digits = thousands; sum check then verifies (89/99 dev pass) and flags lines when it fails. See `reports/number_policy_check.md`.
- `ocr_agent` has no temperature in config (not in the instruction); it uses the model default until set.
- Very rare classes: `void_menu.nm`, `void_menu.price`, `menu.itemsubtotal` have 1 train line each.
- `nemotron-3-nano:4b`, `llama3.2:3b`, `nomic-embed-text` not pulled yet (needed from M4 / M5).

## Revisit at M8
- `reasoning_effort: none` (thinking off) for all models. Thinking may help the diagnoser / rule writer; it costs ~50x tokens.
- Verifier thresholds (`max_gold_accuracy_drop` 0.01, `min_cluster_alpha_gain` 0.05).
- Guideline v1 content, after the first real run.
- The toy agent's tasks make the model read `Rp 12.500` itself, which the number rule now forbids for real agents. Fine for a harness test; don't copy that pattern.
