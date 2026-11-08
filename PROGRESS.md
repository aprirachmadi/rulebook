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

- **Milestone 3 (skills, context, guard)**, ready for review:
  - `rulebook/harness/skills.py`: `skills/<name>/SKILL.md` with frontmatter; agents see name + description in the system prompt; built-in `load_skill` returns full text for allowlisted skills only.
  - `rulebook/harness/context.py`: chars/4 estimate; over budget -> keep system + task + last `context_keep_turns` turns, summarize older tool results to one line, then drop oldest turns (tool calls and their results stay paired). The loop keeps the full history and sends the compacted view.
  - `rulebook/harness/guard.py`: `CLIApprover` (rich panel with args and context; approve / edit / reject; edited args re-validated; approver + UTC timestamp). Wired into `ToolRegistry.run`; approvals logged as `approval` trace events.
  - `agents.yaml` + `rulebook/harness/agents.py` `build_agent()`: model = key into config `models`; temperature from config; refuses agents with no system prompt / missing output schema / budget not below the model's context window.
  - Tests: 78 passing.
  - Live on `qwen3.5:4b` (`rulebook m3-check`): skill loaded on demand and used (KOPI7); 51-step run read all 50 pages, correct answer, max context sent 2,987 / budget 3,000 (full history ~18k tokens), Ollama prompt_tokens max 2,380 < 4,096; approval: blocks with no input (killed by timeout, exit 124, tool never ran), approve runs it, reject does not.
  - Bugs found by the live runs and fixed: rich markup swallowed `[a]`/`[e]`/`[r]` in the prompt and would swallow `[...]` in receipt text.
  - Found: Ollama serves only 4,096 tokens of context for `qwen3.5:4b` and `gemma4:12b` (server default) and silently drops the start of longer prompts. `config.yaml` `harness.context_window` records this; budgets are checked against it.

- Milestone 3 accepted (2026-11-08). Follow-ups done:
  - Number money rule (per token) in `receipt.py`: `money_value`, `parse_amount`; `sum_check` uses it. Mixed-style receipts now pass.
  - Contexts: labelers use `-8k` tags (Modelfile `num_ctx 8192`, same weights), gemma agents `gemma4:12b-16k`. Budgets 6000 / 14000 leave room for replies.
  - VRAM guard (`rulebook/harness/memory.py`): before a model's first call, unload every other loaded model, then refuse unless free VRAM >= estimate + 1.5 GB headroom. Estimates in `config.yaml` `memory.estimate_gb`, calculated from `ollama show -v`. Models without an estimate are refused.
  - Labelers preload the guideline into the system prompt; the loop warns (trace `warning` event) when prompt + skills exceed 80% of the budget.
  - `ocr_agent` temperature 0.

- **Milestone 4 (OCR + labelers)**, built; first 20-receipt runs in progress:
  - PaddleOCR 3.7.0 / paddlepaddle 3.3.1 CPU (no CUDA, no torch), PP-OCRv6 small models copied from crumb, offline. ~3.3 s/receipt, ~0.9 GB RAM. Results cached in `runs/ocr_cache/`.
  - `rulebook/tools/ocr.py`: OCR words rebuilt from PaddleOCR tokens; OCR word -> CORD word by centre-in-box then text similarity. On train_0000-0019: 89% of OCR words matched, 86% of CORD words covered; the unmatched OCR text is mostly noise from CORD's masked regions.
  - `rulebook/tools/labels.py`: `LabelDoc` for both modes (gold lines sorted top-to-bottom so CORD's annotation order can't hint at labels), `render()` (line idx, position 0-100, numbered words), per-receipt answer schema: label enum, bounded line/word indices, full non-overlapping coverage validator.
  - Labelers run in structured mode (one call, JSON schema enforced), one model at a time; guideline preloaded from `skills/cord-guideline/SKILL.md` (generated from the current `guidelines/v<N>.md`).
  - `rulebook/data/store.py` (SQLite: runs, docs, votes per word, label_failures), `rulebook/tools/stats.py` (word-level alpha, one-vs-rest alpha per label, agreement pattern), `rulebook run`, `eval/labeling_report.py`.
  - Smoke runs on a 71-line receipt found and fixed: reply cap too small (answers cut off), repair prompts that pushed past the 8k window (now a fresh repair), llama inventing line numbers (now bounded in the schema), and alpha crashing with < 2 labelers.
  - 114 tests passing.

- First 20-receipt runs (train_0000-0019), reports in `reports/run_<id>_labeling.md`:
  - gold 20261108-134252-9c093a: receipts passing validation a 15/20, b 4/20, c 0/20. Agreement with v2 on labeled words: qwen 84.9%, nemotron 17.1%, llama -. Alpha 0.031 (not meaningful: no word had all three votes).
  - ocr 20261108-135346-832f4d: a 13/20, b 3/20, c 1/20. Agreement with v2: qwen 65.3%, nemotron 6.8%, llama 47.8% (23 words). Alpha 0.060 (same caveat). OCR misses 67/624 words.
  - Failures are coverage errors (missed / doubled words, word index past the end of its line), not broken JSON. qwen's failed receipts miss 0-10 words; llama's 3-65.

- Fixes after the first runs (2026-11-08): `other` label, 20-line chunks, reason only below 0.7 confidence, per-line word bounds in the JSON schema (anyOf per line; Ollama enforces it, checked live), targeted repair of only missed/conflicting words. 117 tests.
- Rerun on train_0000-0019 (`reports/run_<id>_labeling.md`):
  - gold 20261108-143209-f4963e: complete receipts a 20, b 20, c 14. Agreement with v2 excl. other: qwen 87.2%, nemotron 24.8%, llama 42.6%. `other` use: 18.5% / 21.6% / 3.3% (in gold mode every word has a v2 label). Alpha 0.181.
  - ocr 20261108-144026-0d2712: complete a 19, b 19, c 7. Agreement excl. other: 79.5% / 10.6% / 36.4%. `other` use: 37.6% / 39.0% / 6.6%. Alpha 0.105.

## In progress
- Waiting for the decision on labeler models (nemotron and llama are far below qwen).

## Next
- Milestone 4: OCR agent + labelers (needs `nemotron-3-nano:4b`, `llama3.2:3b` pulled).

## Open questions
- `nomic-embed-text` is not in `ollama list` yet (needed in M5).
- Very rare classes: `void_menu.nm`, `void_menu.price`, `menu.itemsubtotal` have 1 train line each.

## Revisit at M8
- `reasoning_effort: none` (thinking off) for all models. Thinking may help the diagnoser / rule writer; it costs ~50x tokens.
- Verifier thresholds (`max_gold_accuracy_drop` 0.01, `min_cluster_alpha_gain` 0.05).
- Guideline v1 content, after the first real run.
- The toy agent's tasks make the model read `Rp 12.500` itself, which the number rule now forbids for real agents. Fine for a harness test; don't copy that pattern.
- Agent budgets: 6,000 (labelers) / 14,000 (gemma agents); `context_keep_turns` 4; VRAM headroom 1.5 GB. The chars/4 estimate ran ~25% above Ollama's real count in the 50-step run (safe direction).
- Token estimate: chars/4 (SPEC) undercounts real labeler prompts ~2x (est 1.4k vs real 2.8k). The loop now also checks Ollama's real counts against the window.
- VRAM estimates are conservative: nemotron-3-nano:4b-8k measured 3.0 GB in `ollama ps` (estimate 3.9).
- Strict coverage ("every word exactly once", one full repair) is hard for the small labelers on long receipts. Options if the failure rate is high: targeted repair of only the missing words, or labeling long receipts in line chunks.
