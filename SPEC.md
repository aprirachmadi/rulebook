# Rulebook — Project Spec

An agentic labeling system that improves its own **labeling guideline** from model disagreement.
Built on a **custom agent harness written from scratch** (no LangGraph / LangChain / CrewAI / AutoGen).
Domain: Indonesian receipts (CORD dataset), starting from the **receipt image**.

Purpose: portfolio project targeting Datasaur (private, on‑prem AI; Data Studio labeling).
Everything must run **fully offline on local open‑weights models**.

---

## 0. Ground rules for the coding agent

- Poetry project. Python version = the same one the neighboring projects use (see CLAUDE.md → Environment). Type hints everywhere, `ruff` + `pytest`.
- **No agent frameworks.** Allowed libs: `httpx` or `openai` (client only, pointed at a local server), `pydantic`, `numpy`, `pandas`, `datasets`, `paddleocr`, `krippendorff`, `scikit-learn`, `fastapi`, `uvicorn`, `sqlite3`/`sqlmodel`, `rich`, `typer`, `pillow`, `pyyaml`. Ask before adding anything else.
- Build in the milestone order below. Each milestone ends with passing tests and a short note in `PROGRESS.md`. Stop for my review between milestones.
- Never call external APIs. All models go through an OpenAI‑compatible endpoint (`OLLAMA_BASE_URL`, default `http://localhost:11434/v1`).
- When something in this spec is ambiguous, **ask before deciding** (see CLAUDE.md). Record each answer in `DECISIONS.md`.

### [CONFIRM] items (defaults used until changed)
| Item | Default |
|---|---|
| Hardware | Single GPU with ~12–16 GB VRAM, or CPU fallback (slow) |
| Labeler models | `qwen2.5:7b-instruct`, `llama3.1:8b-instruct`, `gemma2:9b` via Ollama |
| Orchestrator / Diagnoser / Rule writer model | `qwen2.5:14b-instruct` (fallback: `qwen2.5:7b-instruct`) |
| Vision model for OCR re‑reads | `qwen2.5vl:7b` (optional; skip if VRAM too small) |
| Embedding model | `nomic-embed-text` via Ollama |
| UI | FastAPI + minimal React (Vite). CLI first, UI in milestone 7 |

Model names live in `config.yaml` only — never hardcode them.

---

## 1. Data

### Source
- CORD v2: Hugging Face `naver-clova-ix/cord-v2` → the **only** label source, and the **answer key**.
- CORD v1 is not used. It differs from v2 on only 18 lines once v2's label renames are mapped (record: `reports/m1_gate_findings.md`).
- License CC BY 4.0. 1,000 receipts (800 train / 100 dev / 100 test).
- Scope = **labeling from scratch**: agents label blind, from the image / OCR text + the guideline. They never start from existing labels.

### Structure (verify against the actual data before coding on it)
Each sample has an image and a `ground_truth` JSON string. Inside `gt` → `valid_line`: a list of lines, each with
`category` (e.g. `menu.nm`, `menu.price`, `sub_total.subtotal_price`, `total.total_price`), `group_id`, `sub_group_id`,
and `words` (each with `quad` box coordinates, `text`, `is_key`, `row_id`).

### Unit of labeling = one `valid_line`
- `line_text` = words joined with spaces, in reading order: rows top to bottom (by mean top edge of each `row_id`), words left to right within a row. Use the same ordering whenever OCR text is compared against CORD text.
- `line_box` = union of the word quads.
- Label space = the set of `category` values that actually appear in the data. **Derive it from the data**, don't hardcode. Expect ~30.

### Milestone 1 task — v2 data layer
1. Load CORD v2, extract lines (text in reading order, box, words) per receipt.
2. Keep labels out of agent-facing data; expose them only through the answer key (see leakage rules).
3. Output `reports/cord_v2_summary.md`: receipts and lines per split, label space, label distribution.

### Splits and leakage rules (critical)
- `work` set: the receipts the agents process (train split).
- `gold` set: 30 receipts from dev with **v2 labels**, used by the Verifier agent to test rule patches. Picked with a fixed seed, stratified so every dev class appears at least once; ids frozen in `config.yaml`. The Verifier reads them only through `rulebook.data.gold.gold_labels()`, which refuses train or test receipts.
- `eval` set: test split, v2 labels. **No agent, tool, prompt, or skill may ever read v2 labels of the work or eval sets.** Only `eval/` scripts read them, after a run finishes.
- Enforce this in code: the data layer exposes v2 labels only through `eval.load_answer_key()` and the gold‑set tool. Add a test that fails if any agent tool can reach them.

---

## 2. Pipeline (what happens to one batch)

```
receipt image
  → OCR agent: PaddleOCR → lines (text, box, conf); match OCR lines to CORD lines by IoU ≥ 0.5
  → 3 Labeler agents: each labels every line using guideline vN (JSON output, label + confidence 0–1)
  → agreement: Krippendorff's alpha (nominal) per batch, per label; per‑line vote pattern
  → lines where all 3 agree with confidence ≥ 0.7 → accepted
  → disagreements → clustering → Diagnoser classifies each cluster:
        OCR_ERROR | MODEL_ERROR | AMBIGUOUS | GUIDELINE_GAP
  → GUIDELINE_GAP → Rule writer drafts a patch → Verifier dry‑runs it on gold set
  → human approval (CLI prompt first, UI later) → guideline vN+1 → relabel affected lines
  → loop on next batch
```

### Modes
- `--text-source gold`: use CORD's provided text (baseline, isolates labeling quality).
- `--text-source ocr`: use PaddleOCR output (the real demo). Default.

### Diagnosis rules of thumb (Diagnoser skill content)
- OCR_ERROR: OCR conf < 0.6, or text has digit/letter swaps (`O`↔`0`, `l`↔`1`, `S`↔`5`), or receipt sum check fails and the line holds a number.
- MODEL_ERROR: 2 of 3 agree, the guideline text clearly supports the majority, and the dissenter's reasoning contradicts the guideline.
- AMBIGUOUS: the guideline covers the case but the input itself is unclear (e.g. truncated line).
- GUIDELINE_GAP: no guideline rule covers the case, or two rules conflict. This is the only class that produces a patch.

### Clustering (keep it simple first)
1. Key = (sorted set of voted labels, normalized text pattern). Normalized pattern: lowercase, digits → `#`, e.g. `disc #%`.
2. If a key group has > 50 lines or there are too many tiny groups, sub‑cluster with embeddings (`nomic-embed-text`) + agglomerative clustering (cosine, distance threshold 0.25).
3. Only clusters with ≥ 3 lines go to the Rule writer. Smaller ones go to the review queue as single items.

### Number rule (hard rule)
- All Indonesian number parsing ("12.500" → 12500, "2.000,00" → 2000, "Rp") happens in code (`rulebook/tools/receipt.py`). Tools return parsed numbers.
- Models never parse prices or do arithmetic.
- CORD mixes formats ("53.000", "2.000,00", "38,000", "Rp 194,726"). The parser handles both. Where the reading is unclear it returns "ambiguous" instead of guessing.
- How unclear readings are resolved per receipt: **pending decision** (see DECISIONS.md / PROGRESS.md).

### Receipt sum check (tool)
Parse prices (Indonesian format: `.` thousands separator, `,` decimal; strip `Rp`). Check sum(menu prices) − discounts + tax/service ≈ subtotal/total within 1%. Return which numeric lines are suspicious.

---

## 3. The harness (`rulebook/harness/`)

Write from scratch. Target ~1,000 lines total.

| File | Responsibility |
|---|---|
| `model.py` | `ModelClient.chat(messages, tools=None, json_schema=None)` against the OpenAI‑compatible endpoint. Retries with backoff. Supports native tool calls; **fallback**: if the model doesn't emit valid tool calls, parse a JSON block `{"tool": ..., "args": ...}` from the text. |
| `tools.py` | `@tool(name, description, needs_approval=False)` decorator; args schema from a pydantic model; registry; `run()` validates args, catches exceptions, returns a compact string/JSON result (truncate > 2,000 chars with a note). |
| `skills.py` | Skills = folders in `skills/<name>/SKILL.md` with frontmatter `name`, `description`. Agents always see name + description. A built‑in `load_skill(name)` tool returns the full text. Only skills in the agent's allowlist are visible/loadable. |
| `context.py` | Token budget per agent (estimate tokens as chars/4). When over budget: keep system prompt + task + last N turns, replace older tool results with one‑line summaries. |
| `agents.py` | `Agent(name, system_prompt, model, tools: list[str], skills: list[str], max_steps, budget, output_schema=None)`. Loaded from `agents.yaml`. |
| `loop.py` | `run(agent, task) -> AgentResult`: model call → tool calls → results appended → repeat until no tool calls, `final_answer` tool, or `max_steps`. Validates final output against `output_schema` if set (one repair retry). |
| `guard.py` | Approval gate for `needs_approval` tools. `CLIApprover` (rich prompt: shows patch, impact, sample lines; approve / edit / reject) and later `QueueApprover` (writes a pending item to DB, waits for UI decision). Records approver + timestamp. |
| `orchestrator.py` | Orchestrator is itself an `Agent` with a `delegate(agent_name, task)` tool that runs a sub‑agent via `loop.run` and returns its result. Sub‑agents cannot delegate (no recursion). |
| `trace.py` | Append every step to `runs/<run_id>/trace.jsonl`: run_id, agent, step, messages hash, tool name, args, result (truncated), latency, tokens. This **is** the audit log. |

### Agents (`agents.yaml`)
| Agent | Tools | Skills |
|---|---|---|
| `ocr_agent` | `run_paddleocr`, `crop_line`, `vlm_read_crop`, `sum_check` | `indonesian-receipt-format` |
| `labeler_a/b/c` (3 different models) | `get_receipt_context` | `cord-guideline` (current version) |
| `diagnoser` | `get_cluster`, `get_votes`, `get_ocr_conf`, `sum_check`, `search_similar_lines` | `disagreement-taxonomy`, `cord-guideline` |
| `rule_writer` | `get_cluster`, `search_similar_lines`, `get_guideline` | `guideline-style`, `cord-guideline` |
| `verifier` | `simulate_patch`, `compute_alpha`, `run_gold_set`, `diff_labels` | `stats-checks` |
| `orchestrator` | `delegate`, `list_clusters`, `request_human`, `apply_patch` (**needs_approval**), `relabel` (**needs_approval**) | — |

Labelers should use `output_schema` (JSON: `[{line_idx, label, confidence, reason}]`) and may run without tool calls for speed. Batch ~1 receipt per call.

---

## 4. Guideline

- File: `guidelines/v1.md`. One section per label class: definition (1–2 lines) + 1–2 examples.
- v1 is **not** written from train labels: definitions come from the class names, examples only from non-gold dev receipts (never train, never test).
- **Intentionally incomplete**: don't cover discounts vs subtotal discounts, service charge, item modifiers/sub‑items, void lines, or "etc" fields. The agent should discover these.
- Each patch = a numbered rule (`7.2`) with: rule text, 3–5 example lines (from work set only), rationale, affected line count. Stored as `guidelines/patches/<n>.json` and rendered into `guidelines/v<N>.md`.
- Revert rule: if the Verifier finds gold‑set accuracy drops or alpha on the cluster doesn't improve by ≥ 0.05, the patch is auto‑rejected before reaching the human.

---

## 5. Storage

SQLite `rulebook.db`: `receipts`, `lines` (text, box, ocr_conf, text_source), `votes` (line, agent, guideline_version, label, confidence, reason), `final_labels` (line, label, status: accepted|relabeled|ocr_corrected|ambiguous|human, rule_id), `clusters`, `patches` (status: proposed|verified|approved|rejected, approver, timestamps), `review_queue`.

---

## 6. Outputs

1. `exports/receipts/<id>.json` per receipt:
```json
{"receipt_id": "0412", "guideline_version": "v4",
 "lines": [{"text": "DISC 10%", "value": -5000, "label": "menu.discountprice",
            "status": "relabeled", "rule": "7.2",
            "votes": {"labeler_a": "...", "labeler_b": "...", "labeler_c": "..."}},
           {"text": "SUBTOTAL", "value": 53000, "label": "sub_total.subtotal_price",
            "status": "ocr_corrected", "ocr_raw": "SUBT0TAL 53.0O0", "ocr_conf": 0.41}],
 "checks": {"sum_matches_total": true}}
```
2. `guidelines/v<N>.md` final guideline + changelog.
3. `reports/run_<id>.md` + charts (PNG) with the metrics below.

---

## 7. Evaluation (`eval/`, runs after the agents finish)

- Krippendorff's alpha per guideline version (nominal, 3 labelers).
- % of lines that reached a human; # human decisions vs # disagreeing lines.
- Label accuracy vs v2 on the eval set, for guideline v1 vs final: overall accuracy + macro‑F1. Per‑class scores flag classes with < 5 eval lines as "too few to score".
- No label‑error‑detection metric for now (v1 dropped, no noise injection).
- **Suspect v2 lines**: after a run, list the top 50 lines where all 3 labelers confidently disagree with v2, for a hand check of whether v2 itself is wrong. "Confidently disagree" = all 3 labelers at confidence ≥ 0.7 and none matches v2; ranked by average confidence. Output `reports/run_<id>_suspect_v2.csv`.
- OCR: character error rate vs CORD text; labeling accuracy with gold text vs OCR text.
- Diagnosis accuracy: on clusters, was OCR_ERROR predicted when OCR text really differed from CORD text?
- Ablation: single generalist agent (all tools) vs specialized agents — accuracy, tokens, steps, wall time.

---

## 8. Milestones (acceptance criteria)

Dev runs use 20–50 receipts. The full 800‑receipt work set is only for the final report.

1. **Data**: v2 loader, line extraction, answer key, leakage test. ✅ when `reports/cord_v2_summary.md` exists and tests pass.
2. **Harness core**: `model.py`, `tools.py`, `loop.py`, `trace.py`. ✅ when a toy agent with a `calculator` tool solves 5 tasks on a local model and the trace file is valid.
3. **Skills + context + guard**: ✅ when an agent loads a skill on demand, survives a 50‑step run without exceeding budget, and a `needs_approval` tool blocks until CLI approval.
4. **OCR agent + labelers**: ✅ when 20 receipts are OCR'd, matched to CORD lines, labeled by 3 models, and alpha is reported, in both text‑source modes.
5. **Diagnoser + clustering + rule writer + verifier**: ✅ when at least one GUIDELINE_GAP cluster yields a verified patch on a 100‑receipt run.
6. **Orchestrator end‑to‑end loop**: ✅ when a full run on the work set goes v1 → vN with CLI approvals and exports are written.
7. **UI**: FastAPI + React with 3 screens — project dashboard (metrics, alpha‑by‑version chart, changelog), review queue (receipt + patch with approve/edit/reject, OCR "can't read" items, ambiguous items), run trace viewer (per‑agent steps and tool calls). ✅ when approvals from the UI drive the loop.
8. **Eval + report + ablation**: ✅ when `reports/run_<id>.md` contains every metric in section 7.

---

## 9. Repo layout

```
rulebook/
  harness/        model.py tools.py skills.py context.py agents.py loop.py guard.py orchestrator.py trace.py
  tools/          ocr.py receipt.py labels.py clusters.py guideline.py stats.py
  data/           cord.py store.py
  pipeline/       run.py (CLI entry: `rulebook run --batch-size 20 --text-source ocr`)
  api/            FastAPI app
skills/           <name>/SKILL.md
guidelines/       v1.md, patches/
eval/             metrics.py, ablation.py, report.py
web/              React app (milestone 7)
agents.yaml  config.yaml  DECISIONS.md  PROGRESS.md  README.md
tests/
```
