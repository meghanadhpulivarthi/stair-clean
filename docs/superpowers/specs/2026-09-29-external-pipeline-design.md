# STAIR external-facing pipeline — design spec

Date: 2026-09-29
Status: approved (in-chat design), pending spec review

## Purpose

Turn this repo into something shareable with an outside user (not on the team):
a modular, config-driven pipeline covering parse -> prepare data -> train ->
report -> eval -> report, wrapping the existing internal code. The end user
never reads or edits Python — they run `stair` CLI commands and edit YAML
config files.

## Current state (what exists today, found during exploration)

- `benchmark/extract_pdf/`: PDF parsing, chunking, token stats, NQ-specific
  prep. Standalone `uv` project, driven by ad hoc shell scripts pointing at
  internal absolute paths (`~/data/ki_data`, IBM-internal corpora like
  `systemz`/`sz`).
- `stair/` (nested repo, currently untracked in the outer repo): the actual
  STAIR method — fine-tune an LLM (via LoRA/SFT) to select relevant
  table-of-contents section titles for a query, given a book/manual title +
  TOC. Contains:
  - `experiments/rag_eda_benchmark/`: synthetic QA generation, indexing,
    zero-shot ("OOB") retrieval eval via vLLM/RITS, metrics (recall@K today).
  - `external/silt` (git submodule, private `github.ibm.com` repo, now
    initialized locally): the real training framework.
    - `src/training_fsdp_trainer.py`: the actual training entrypoint,
      launched via `torchrun`. Subclasses HF TRL's `SFTTrainer`
      (`src/trainers.py`) with LoRA. Standard HF `Trainer`, so it writes
      `trainer_state.json` (with `log_history`) to each checkpoint dir
      regardless of `report_to`.
    - `src/arguments.py`: pydantic-based config objects; YAML configs use
      `${var}` interpolation + a `defaults:` block.
    - `scripts/sft/*.sh`: LSF/`bsub`-based job submission
      (`train_ccc_separate_args.sh` calls `torchrun --nnodes=$NNODES ...`
      with node/rank values derived from LSF env vars — but `torchrun` runs
      fine standalone with `--nnodes=1` on a single machine).
    - `experiments/example/`: an already-scaffolded "example" experiment
      (OpenROAD docs) with config files but no bundled data (paths point to
      `/dccstor/...`).
  - `.env` file present — must never be copied/exposed.
- No example dataset is currently bundled anywhere in the repo. All real
  corpora referenced are either internal-only (systemz, sz) or scaffolded but
  data-less (OpenROAD example).

User confirmed: do not change the core training logic (TRL `SFTTrainer`
subclass, LoRA config handling, arguments parsing) — only add
wrappers/adapters around it.

## Goals

1. One `stair` CLI, subcommands only — no code editing required for normal use.
2. Config-driven: one `configs/base.yaml` with all defaults; user changes go
   in a small override YAML they author themselves.
3. Modular so a new corpus can be dropped in: parse -> prepare train/val/test
   -> train -> eval, without touching pipeline code.
4. Terminal reports: after training, a report on loss/eval curves with
   diagnostics (overfitting, plateaus, LR issues, early-stopping) and
   recommendations. After eval, a report with retrieval metrics.
5. Ships with exactly one example corpus, self-contained (no private/internal
   dependencies, no internal paths, no LSF assumptions).
6. Runs standalone: single GPU by default, CPU fallback documented. No
   multi-node, no wandb/tensorboard requirement, no lm-eval-harness/model_eval
   dependency (both explicitly rejected — model_eval is IBM-internal and
   would break for an outside user).

## Non-goals (explicitly out of scope for this pass)

- Distributed / multi-node training.
- wandb/tensorboard dashboards (reports are terminal-only).
- lm-evaluation-harness / model_eval integration.
- Supporting more than one bundled example corpus (structure must allow
  adding more later; only one ships now).
- Preserving the LSF/`bsub` job-submission scripts — dropped, not adapted.

## Architecture

```
stair/                          <- repo root after cleanup
  cli.py                        # `stair` entrypoint, argparse subcommands
  configs/
    base.yaml                   # every tunable, defaults, commented sections
    examples/book_example.yaml  # sample override showing the pattern
  core/
    data/                       # parse, chunk, TOC-build, QA-pair generation, split
    train/                      # vendored SiLT src/ (trainers.py, training_fsdp_trainer.py,
                                 # arguments.py, data/) — logic untouched, import paths flattened
    eval/                       # retrieval metrics: recall@K, hit rate, NDCG, MRR, hallucination rate
    report/                     # NEW: reads trainer_state.json + eval output,
                                 # renders terminal curves & diagnostics
  data/example_book/            # bundled example corpus + generated splits
  README.md
```

Everything internal-only is deleted outright during the port, not merely
excluded: LSF/`bsub` scripts, `/dccstor` paths, `.env`, wandb org config,
model_eval/lm-eval-harness references, IBM-internal corpora
(`systemz`/`sz`/`nq`-internal paths).

### Config system

- `configs/base.yaml`: the only place with full defaults (model name, LoRA
  rank/alpha/dropout, optimizer/lr, epochs, batch size, eval strategy/Ks,
  data paths, etc.), grouped into commented sections per this project's
  code-style rules (config block at top, full-word keys).
- A user override file contains only the keys they want to change.
- The CLI deep-merges override onto base (override wins) and translates the
  resolved dict into the shapes SiLT's `arguments.py` pydantic models expect.
  This translation is the adapter layer — `arguments.py` itself is untouched.
- No `${var}`/bash-style templating exposed to the end user; that mechanism
  stays internal to the translation step if still needed to talk to vendored
  SiLT code.

### CLI commands

- `stair prepare-data --corpus <path> --out <dir>`
  Parse -> chunk -> build TOC -> generate synthetic QA pairs -> split
  train/val/test. Prints item counts and every output file path as it goes
  (per this project's traceability rules): loaded doc count, chunk count,
  QA pair count, train/val/test sizes, output paths.
- `stair train --config <override.yaml> --out <run_dir>`
  Merges config, writes `config.json` into `run_dir` (traceability), invokes
  the vendored trainer locally via
  `torchrun --nnodes=1 --nproc_per_node=<num_gpus>` (`num_gpus` defaults to 1;
  works, slowly, on CPU too).
- `stair report train --run <run_dir>`
  Reads `trainer_state.json`'s `log_history` from the run's checkpoint dir
  (no wandb/tensorboard needed since HF `Trainer` always writes this).
  Renders terminal loss/eval curves, then a rule-based diagnostics section:
  - overfitting (train loss falling while eval loss rises)
  - plateaued loss (no meaningful improvement over N eval steps)
  - LR likely too high (loss spikes/divergence) or too low (near-flat curve)
  - early stopping triggered, and how early relative to configured epochs
  Each flagged issue includes a concrete recommendation (e.g. suggested LR
  change, more data, fewer epochs).
- `stair eval --run <run_dir> --config <override.yaml>`
  Runs inference on the test split using the trained checkpoint, computes
  Recall@K, Hit Rate, NDCG, MRR, and hallucination rate (extends the
  existing `compute_metrics` logic from `indexing.py`/`utils.py`).
- `stair report eval --run <run_dir>`
  Terminal table of the above metrics across configured Ks, with weak spots
  flagged.

### Data pipeline

Unifies `benchmark/extract_pdf` (PDF -> docs -> chunks) with the TOC-building
and synthetic QA generation currently in `rag_eda_benchmark`
(`create_synthetic_data.py`) into one deterministic `prepare-data` flow. Output
schema matches what the vendored SiLT data loader expects: `question`,
`answer`, `reference` (section id/title) JSONL files, plus a `toc.json`.

### Reporting

New code (`core/report/`) — nothing today reads `trainer_state.json` or
renders curves. Terminal rendering via a lightweight plotting library (no
GUI dependency). Diagnostics are rule-based thresholds over the log history,
not a model — thresholds documented as config-top variables (per code-style
rules) so they're easy to see and tune.

### Example corpus

One public-domain book from Project Gutenberg, with existing chapter/section
structure standing in for the TOC. Small enough to fine-tune quickly on a
single GPU (or slowly on CPU). Exact title chosen during implementation
(short, clearly-structured nonfiction preferred); the pipeline works with
any single book meeting that shape, so swapping later is just re-running
`prepare-data` on a different source.

## Data flow summary

```
raw corpus (PDF/text)
  -> stair prepare-data --corpus ... --out data/example_book/
       -> docs.jsonl -> chunks.jsonl -> toc.json -> synthetic QA pairs
       -> train.jsonl / val.jsonl / test.jsonl
  -> stair train --config configs/examples/book_example.yaml --out runs/<name>/
       -> checkpoints/, trainer_state.json
  -> stair report train --run runs/<name>/
       -> terminal curves + diagnostics
  -> stair eval --run runs/<name>/ --config configs/examples/book_example.yaml
       -> eval/results.jsonl (per-query retrieval + metrics)
  -> stair report eval --run runs/<name>/
       -> terminal metrics table
```

## Testing approach

- Data pipeline: run `prepare-data` on the bundled example corpus, assert
  expected file counts/shapes.
- Train: a short/debug config (few steps, tiny LoRA rank) run on CPU in CI
  as a smoke test, confirming `trainer_state.json` is produced and
  `report train` can parse it.
- Eval + report eval: run against the smoke-test checkpoint, confirm the
  metrics table renders without needing a fully converged model.
- No GPU-dependent test is required to pass in CI; GPU path is exercised
  manually.

## Open implementation details (left to the plan, not blocking spec approval)

- Exact Project Gutenberg title for the example corpus.
- Exact terminal-plotting library choice.
- Exact translation mapping from the flat user config to SiLT's
  `model_config`/`data_config` pydantic shapes.
