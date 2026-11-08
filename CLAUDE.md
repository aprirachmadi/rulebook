# CLAUDE.md

Rulebook: an agentic labeling system on a from-scratch agent harness, run on local models, using CORD receipts.
**SPEC.md is the source of truth.** Read it before any task. This file says how to work.

## How to work (every task)
1. **Read**: the relevant SPEC section, `PROGRESS.md`, and the files you'll touch.
2. **Plan**: write a short plan (files, steps, how you'll verify). For anything bigger than a small fix, show me the plan and wait for "go."
3. **Build** in small steps. Keep each change focused. No drive-by refactors.
4. **Verify**: run `ruff check .` and `pytest`, then actually run the code you changed. Show the real output.
5. **Report**: what changed, how you verified it, and any open questions. Update `PROGRESS.md`.

## You decide alone (just do it)
- Code structure inside a file, naming, helpers, type hints, docstrings
- Writing and fixing tests
- Fixing bugs where the intended behavior is clear from SPEC.md
- Formatting, lint fixes, small refactors of code you just wrote

## You ask me first (stop and wait)
- Any **logic or business decision**: thresholds, label rules, guideline content, diagnosis rules, clustering method, metrics, data splits, what counts as "done"
- Anything SPEC.md doesn't cover, is ambiguous about, or that you think is wrong
- Changing a value or behavior that's already in SPEC.md
- Adding a dependency, picking or switching a model, changing the repo layout
- Deleting or overwriting data, runs, guidelines, or anyone else's files
- Making scope bigger or smaller than the current milestone

**How to ask:** one short message with the question, 2–3 options with trade-offs, and your recommendation. Then **stop**. Don't keep building on a guess. If the question blocks only part of the work, finish the parts that don't depend on it and say so.

Write every decision I make into `DECISIONS.md` (date, question, answer).

## Hard rules
- **No agent frameworks** (LangGraph, LangChain, CrewAI, AutoGen, LlamaIndex agents). The harness is the project.
- **Offline only.** Models go through the local OpenAI-compatible endpoint in `config.yaml`. No external APIs.
- **No leakage.** Agents, tools, prompts and skills must never read CORD v2 labels for the work or eval sets. Only `eval/` reads them, after a run.
- **No fake results.** Never hardcode, mock or estimate numbers that should come from a real run. If something can't run, say so.
- **No hardcoded models or thresholds.** They live in `config.yaml`.
- **Git:** commit freely, in small commits with clear messages, after tests pass. Don't push unless I ask.

## Environment (Poetry)
- Use **Poetry** for everything (`poetry add`, `poetry run`). No bare `pip install`.
- **Before creating the env**, check the neighboring projects (sibling folders of this repo):
  1. Look for `pyproject.toml`, `poetry.lock`, `.python-version`, `requirements*.txt`, and existing `.venv`s.
  2. Find which **Python version** they use and where that interpreter lives (`poetry env info`, `which python3.x`, pyenv).
  3. Find which **torch version and build** they use (CPU / CUDA, e.g. `2.x.x+cu121`, and its package source).
- Report what you found and propose the Python + torch versions. Wait for my "go" before creating the env.
- Then: `poetry env use <that existing interpreter>`. Don't install a new Python.
- If torch is needed, pin the **exact same version, build and source** as the neighbor project, so the cached wheel gets reused instead of downloaded again.

## When stuck
If the same approach fails twice, stop. Tell me what you tried, what happened, and what you suggest next. Don't loop.

## Commands
```bash
poetry run ruff check . && poetry run pytest     # before saying anything is done
poetry run rulebook run --batch-size 20 --text-source ocr
ollama list                       # check which local models are available
```

## Keep context clean
- Long outputs: summarize, and save the full text to `runs/` or `reports/`.
- At the end of each session, update `PROGRESS.md`: done, in progress, next, open questions.
