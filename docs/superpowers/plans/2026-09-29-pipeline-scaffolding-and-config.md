# STAIR Pipeline Scaffolding & Config System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the installable `stair` package (src-layout), a working `stair` CLI with stubbed subcommands, and the base+override YAML config system that every later plan (data pipeline, train wrapper, reporting, eval) builds on.

**Architecture:** A `uv`-managed Python package at the repo root (`src/stair/`), with a single `stair` console-script entrypoint dispatching to subcommands via `argparse`. Config is two YAML files — `configs/base.yaml` (every default, the only place a value is allowed to be undocumented) and an optional user override YAML — deep-merged at runtime into one plain `dict`, no templating, no env-var substitution exposed to the end user.

**Tech Stack:** Python 3.12, `uv` for env/deps (matches `benchmark/extract_pdf` and `stair/external/silt` conventions already in this repo), `PyYAML` for config loading, `pytest` for tests, `argparse` for the CLI (stdlib, no extra dependency).

**Spec:** `docs/superpowers/specs/2026-09-29-external-pipeline-design.md`

## Global Constraints

- End user never edits Python; all customization goes through the override YAML (spec Goal 2).
- `configs/base.yaml` must contain every tunable with an inline comment explaining non-obvious defaults (code-style.md: config at top of file, comments explain "why").
- No type hints in project code (code-style.md).
- No one-liners / no unpacked comprehensions-as-one-liners (code-style.md).
- No helper-function abstraction unless the same code appears 3+ times (code-style.md).
- Every script prints a run header (timestamp, script name, resolved config) at start (traceability.md).
- No absolute paths anywhere in checked-in code or configs (code-style.md).
- 4-space indentation.
- This plan only adds the new `stair` package; it does not delete or modify `benchmark/extract_pdf/` or the nested `stair/` (legacy, will be vendored from and then removed in a later plan) — leave both untouched.

## Review Focus

- **Missing override file:** `stair` commands must work with base config alone (no `--config` flag) — a user's first run before they've written any override.
- **Override YAML with an unknown key:** a typo'd key in the user's override (e.g. `lroa_rank` instead of `lora_rank`) should be caught loudly, not silently ignored — spec Goal 2 promises config-driven behavior, and a silently-ignored typo breaks that promise invisibly.
- **Override YAML that is empty or only comments:** `yaml.safe_load` returns `None` for an empty file; the merge step must treat that as "no overrides," not crash.
- **Nested override (e.g. overriding one key under `lora:` without repeating the whole block):** the merge must be a deep merge, not a shallow dict `.update()`, or the user loses every other key in that block.
- **CLI invoked with no subcommand:** `stair` alone should print usage/help and exit non-zero, not stack-trace.

---

## File Structure

```
pyproject.toml                  # new: package metadata, uv-managed, console-script `stair`
src/
  stair/
    __init__.py
    cli.py                      # argparse entrypoint, subcommand dispatch
    config.py                   # load_base_config, load_override_config, merge_configs, resolve_config
configs/
  base.yaml                     # every default, commented
tests/
  test_cli.py
  test_config.py
```

---

### Task 1: Package scaffolding and `stair` CLI skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/stair/__init__.py`
- Create: `src/stair/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `stair.cli.build_parser() -> argparse.ArgumentParser` — used directly by tests and by `main()`.
- Produces: `stair.cli.main(argv=None) -> int` — console-script entrypoint, returns process exit code.
- Produces console script `stair` (via `pyproject.toml` `[project.scripts]`), so `uv run stair ...` and (after install) `stair ...` both work.

- [ ] **Step 1: Write the failing CLI test**

Create `tests/test_cli.py`:

```python
import subprocess
import sys


def run_stair(*args):
    command = [sys.executable, "-m", "stair.cli"] + list(args)
    result = subprocess.run(command, capture_output=True, text=True)
    return result


def test_no_subcommand_prints_usage_and_exits_nonzero():
    result = run_stair()
    assert result.returncode != 0
    assert "usage" in result.stderr.lower() or "usage" in result.stdout.lower()


def test_help_lists_all_subcommands():
    result = run_stair("--help")
    assert result.returncode == 0
    assert "prepare-data" in result.stdout
    assert "train" in result.stdout
    assert "report-train" in result.stdout
    assert "eval" in result.stdout
    assert "report-eval" in result.stdout


def test_prepare_data_subcommand_help_runs():
    result = run_stair("prepare-data", "--help")
    assert result.returncode == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stair'` (package does not exist yet).

- [ ] **Step 3: Write `pyproject.toml`**

Create `pyproject.toml`:

```toml
[project]
name = "stair-pipeline"
version = "0.1.0"
description = "Parse, train, and evaluate STAIR retrieval models on your own corpus"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "pyyaml>=6.0",
]

[project.scripts]
stair = "stair.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[dependency-groups]
dev = [
    "pytest>=8.0",
]
```

- [ ] **Step 4: Write the CLI skeleton**

Create `src/stair/__init__.py` (empty file).

Create `src/stair/cli.py`:

```python
import argparse
import sys


def add_prepare_data_subcommand(subparsers):
    parser = subparsers.add_parser(
        "prepare-data",
        help="Parse a raw corpus and build train/val/test splits plus a table of contents",
    )
    parser.add_argument(
        "--corpus", required=True,
        help="Path to the raw corpus (a PDF file with bookmarks/table of contents)",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output directory for docs.jsonl, toc.json, train.jsonl, val.jsonl, test.jsonl",
    )
    parser.add_argument(
        "--config", default=None,
        help="Optional override YAML (see configs/base.yaml for all defaults)",
    )
    return parser


def add_train_subcommand(subparsers):
    parser = subparsers.add_parser(
        "train",
        help="Fine-tune a model on prepared train/val data",
    )
    parser.add_argument(
        "--config", default=None,
        help="Optional override YAML (see configs/base.yaml for all defaults)",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output run directory for checkpoints and logs",
    )
    return parser


def add_report_train_subcommand(subparsers):
    parser = subparsers.add_parser(
        "report-train",
        help="Print a terminal report of training curves and diagnostics for a run",
    )
    parser.add_argument("--run", required=True, help="Run directory produced by `stair train`")
    return parser


def add_eval_subcommand(subparsers):
    parser = subparsers.add_parser(
        "eval",
        help="Run the trained model on the test split and compute retrieval metrics",
    )
    parser.add_argument("--run", required=True, help="Run directory produced by `stair train`")
    parser.add_argument(
        "--config", default=None,
        help="Optional override YAML (see configs/base.yaml for all defaults)",
    )
    return parser


def add_report_eval_subcommand(subparsers):
    parser = subparsers.add_parser(
        "report-eval",
        help="Print a terminal report of retrieval metrics for a run",
    )
    parser.add_argument("--run", required=True, help="Run directory produced by `stair eval`")
    return parser


def build_parser():
    parser = argparse.ArgumentParser(
        prog="stair",
        description="Parse a corpus, train a STAIR retrieval model, and report on it.",
    )
    subparsers = parser.add_subparsers(dest="subcommand")
    add_prepare_data_subcommand(subparsers)
    add_train_subcommand(subparsers)
    add_report_train_subcommand(subparsers)
    add_eval_subcommand(subparsers)
    add_report_eval_subcommand(subparsers)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand is None:
        parser.print_usage(sys.stderr)
        return 1

    print(f"stair {args.subcommand}: not implemented yet", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/stair/__init__.py src/stair/cli.py tests/test_cli.py
git commit -m "feat: add stair package scaffolding and CLI skeleton"
```

---

### Task 2: Config system — `configs/base.yaml` and deep-merge loader

**Files:**
- Create: `configs/base.yaml`
- Create: `src/stair/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `stair.config.load_yaml_file(path) -> dict` — raises `FileNotFoundError` if the path does not exist, returns `{}` if the file is empty.
- Produces: `stair.config.deep_merge(base_dict, override_dict) -> dict` — returns a new dict; nested dicts are merged key-by-key recursively, non-dict values in `override_dict` replace the value in `base_dict`.
- Produces: `stair.config.resolve_config(override_path=None) -> dict` — loads `configs/base.yaml` from the repo root, deep-merges the override file (if given) on top, validates every top-level and nested key in the override exists in the base config (raises `ValueError` listing the unknown keys otherwise), and returns the merged dict. Later plans (train, eval) call this function directly.

- [ ] **Step 1: Write the failing config tests**

Create `tests/test_config.py`:

```python
import textwrap

import pytest

from stair.config import deep_merge, load_yaml_file, resolve_config


def test_load_yaml_file_reads_nested_structure(tmp_path):
    config_path = tmp_path / "sample.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            model:
              name: some-model
              lora:
                rank: 16
            """
        )
    )
    loaded = load_yaml_file(config_path)
    assert loaded["model"]["name"] == "some-model"
    assert loaded["model"]["lora"]["rank"] == 16


def test_load_yaml_file_missing_path_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_yaml_file(missing_path)


def test_load_yaml_file_empty_file_returns_empty_dict(tmp_path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("")
    assert load_yaml_file(config_path) == {}


def test_deep_merge_overrides_nested_key_without_dropping_siblings():
    base = {
        "model": {"name": "base-model", "lora": {"rank": 8, "alpha": 16}},
        "training": {"epochs": 3},
    }
    override = {"model": {"lora": {"rank": 32}}}

    merged = deep_merge(base, override)

    assert merged["model"]["lora"]["rank"] == 32
    assert merged["model"]["lora"]["alpha"] == 16
    assert merged["model"]["name"] == "base-model"
    assert merged["training"]["epochs"] == 3


def test_deep_merge_does_not_mutate_inputs():
    base = {"model": {"lora": {"rank": 8}}}
    override = {"model": {"lora": {"rank": 32}}}

    deep_merge(base, override)

    assert base["model"]["lora"]["rank"] == 8
    assert override["model"]["lora"]["rank"] == 32


def test_resolve_config_with_no_override_returns_base_config():
    resolved = resolve_config(override_path=None)
    assert "model" in resolved
    assert "training" in resolved
    assert "data" in resolved


def test_resolve_config_applies_override(tmp_path):
    override_path = tmp_path / "override.yaml"
    override_path.write_text(
        textwrap.dedent(
            """
            model:
              lora:
                rank: 64
            """
        )
    )
    resolved = resolve_config(override_path=override_path)
    assert resolved["model"]["lora"]["rank"] == 64


def test_resolve_config_empty_override_file_is_a_noop(tmp_path):
    override_path = tmp_path / "override.yaml"
    override_path.write_text("# no changes yet\n")
    resolved = resolve_config(override_path=override_path)
    base_only = resolve_config(override_path=None)
    assert resolved == base_only


def test_resolve_config_rejects_unknown_override_key(tmp_path):
    override_path = tmp_path / "override.yaml"
    override_path.write_text(
        textwrap.dedent(
            """
            model:
              lroa_rank: 64
            """
        )
    )
    with pytest.raises(ValueError, match="lroa_rank"):
        resolve_config(override_path=override_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stair.config'`

- [ ] **Step 3: Write `configs/base.yaml`**

Create `configs/base.yaml`:

```yaml
# STAIR base config — every tunable the pipeline understands lives here with
# its default. Never edit this file for a specific run: write a small
# override YAML with only the keys you want to change and pass it as
# --config to any `stair` command.

data:
  # how many tokens per chunk when a section of the corpus is too long to
  # fit in the table of contents prompt as a single leaf
  chunk_max_tokens: 2048
  chunk_overlap: 32
  # sections shorter than this many words are dropped as noise (headers,
  # page numbers that slipped past the PDF parser)
  min_words_in_a_section: 5
  # synthetic question/answer pairs generated per chunk during `prepare-data`
  qa_pairs_per_chunk: 3
  train_fraction: 0.8
  val_fraction: 0.1
  test_fraction: 0.1
  # random seed for the train/val/test split, so re-running prepare-data on
  # the same corpus reproduces the same split
  split_seed: 42

model:
  # small, ungated, chat-template-equipped model so the bundled example
  # trains on a single modest GPU or (slowly) on CPU without needing a
  # Hugging Face access token
  name: Qwen/Qwen2.5-0.5B-Instruct
  lora:
    rank: 16
    alpha: 32
    dropout: 0.1
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, down_proj, up_proj]

training:
  learning_rate: 1.0e-4
  lr_schedule: constant_with_warmup
  warmup_steps: 20
  num_train_epochs: 3
  batch_size_per_gpu: 4
  gradient_accumulation_steps: 4
  max_seq_length: 2048
  eval_strategy: steps
  eval_interval: 30
  save_interval: 30
  save_total_limit: 3
  # stop early if eval loss hasn't improved for this many evals, so a small
  # example corpus doesn't overfit for the full epoch budget by default
  early_stopping: true
  early_stopping_patience: 5
  metric_for_best_model: eval_loss
  greater_is_better: false
  # 1 = single GPU, no multi-node. Raise to use more local GPUs.
  num_gpus: 1

eval:
  # recall/precision/NDCG/MRR are reported at each of these cutoffs
  ks: [1, 3, 5, 10]
  top_k: 10

report:
  # a run is flagged as "overfitting" once eval loss has been rising for
  # this many consecutive logged evals while train loss keeps falling
  overfitting_patience_evals: 3
  # a run is flagged as "plateaued" once the best eval loss hasn't improved
  # by at least this relative fraction over the last `plateau_patience_evals`
  plateau_relative_improvement: 0.01
  plateau_patience_evals: 5
```

- [ ] **Step 4: Write `src/stair/config.py`**

```python
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"


def load_yaml_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as config_file:
        loaded = yaml.safe_load(config_file)
    if loaded is None:
        return {}
    return loaded


def deep_merge(base_dict, override_dict):
    merged = dict(base_dict)
    for key, override_value in override_dict.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = deep_merge(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


def find_unknown_keys(base_dict, override_dict, path_prefix=""):
    unknown_keys = []
    for key, override_value in override_dict.items():
        full_path = f"{path_prefix}.{key}" if path_prefix else key
        if key not in base_dict:
            unknown_keys.append(full_path)
            continue
        base_value = base_dict[key]
        if isinstance(override_value, dict) and isinstance(base_value, dict):
            unknown_keys.extend(find_unknown_keys(base_value, override_value, full_path))
    return unknown_keys


def resolve_config(override_path=None):
    base_config = load_yaml_file(BASE_CONFIG_PATH)

    if override_path is None:
        return base_config

    override_config = load_yaml_file(override_path)

    unknown_keys = find_unknown_keys(base_config, override_config)
    if unknown_keys:
        joined_keys = ", ".join(unknown_keys)
        raise ValueError(
            f"Override config has keys not present in base config: {joined_keys}. "
            f"Check configs/base.yaml for valid keys."
        )

    return deep_merge(base_config, override_config)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (all 8 tests)

- [ ] **Step 6: Commit**

```bash
git add configs/base.yaml src/stair/config.py tests/test_config.py
git commit -m "feat: add base config and deep-merge config loader"
```

---

### Task 3: Wire config resolution into the CLI and add the run header

**Files:**
- Modify: `src/stair/cli.py`
- Create: `src/stair/run_header.py`
- Test: `tests/test_cli.py` (extend)

**Interfaces:**
- Consumes: `stair.config.resolve_config(override_path=None) -> dict` from Task 2.
- Produces: `stair.run_header.print_run_header(script_name, config) -> None` — prints the traceability header (timestamp, script name, resolved config) to stdout. Used by every subsequent plan's commands (train, prepare-data, eval).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
def test_prepare_data_with_bad_override_key_fails_loudly(tmp_path):
    override_path = tmp_path / "bad_override.yaml"
    override_path.write_text("model:\n  lroa_rank: 64\n")
    corpus_path = tmp_path / "corpus.pdf"
    corpus_path.write_text("not a real pdf, just needs to exist for this test")
    out_dir = tmp_path / "out"

    result = run_stair(
        "prepare-data",
        "--corpus", str(corpus_path),
        "--out", str(out_dir),
        "--config", str(override_path),
    )

    assert result.returncode != 0
    assert "lroa_rank" in result.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — current `main()` always prints "not implemented yet" regardless of config, so the unknown-key error never surfaces.

- [ ] **Step 3: Write `src/stair/run_header.py`**

```python
import datetime
import json


def print_run_header(script_name, config):
    now = datetime.datetime.now()
    print("=" * 60)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {script_name}")
    print(f"Config      : {json.dumps(config, indent=2)}")
    print("=" * 60)
```

- [ ] **Step 4: Wire config resolution into `main()`**

In `src/stair/cli.py`, replace the body of `main()`:

```python
def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand is None:
        parser.print_usage(sys.stderr)
        return 1

    if hasattr(args, "config"):
        try:
            resolve_config(override_path=args.config)
        except (FileNotFoundError, ValueError) as error:
            print(f"stair {args.subcommand}: {error}", file=sys.stderr)
            return 1

    print(f"stair {args.subcommand}: not implemented yet", file=sys.stderr)
    return 1
```

Add the import at the top of `src/stair/cli.py`:

```python
from stair.config import resolve_config
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS (all tests from Task 1, Task 2, and Task 3)

- [ ] **Step 7: Commit**

```bash
git add src/stair/cli.py src/stair/run_header.py tests/test_cli.py
git commit -m "feat: validate config in CLI dispatch and add run-header helper"
```

---

## Self-Review Notes

- **Spec coverage:** This plan covers spec Goal 2 (config-driven, base + override YAML) in full and lays the CLI skeleton for spec Goal 1 (single `stair` entrypoint, subcommands only). It does not cover Goals 1 (full command behavior), 3 (modularity across corpora — depends on the data pipeline plan), 4 (reporting), 5 (bundled example), or 6 (runs standalone) — those are the subject of the data-pipeline, train-wrapper, report, and eval plans that follow this one.
- **Placeholder scan:** no TBD/TODO; the "not implemented yet" string in `main()` is an intentional, tested stub for subcommands whose real behavior is built in later plans, not a placeholder left in this plan's own scope.
- **Type consistency:** `resolve_config(override_path=None) -> dict` and `print_run_header(script_name, config) -> None` are the only two functions later plans need to call; both signatures are fixed here and used as-is.
- **Review Focus:** all five items above have a task and a test (Task 2 for the empty/unknown-key/nested-merge cases, Task 3 for the no-subcommand and bad-override-through-the-CLI cases).

## Next Plans

This is the first of four plans implementing the spec:
1. **This plan** — scaffolding + config system.
2. Data pipeline (`stair prepare-data`): vendor the PDF/TOC extraction from `benchmark/extract_pdf`, add synthetic QA-pair generation and train/val/test splitting, choose and bundle the example corpus.
3. Training wrapper (`stair train`) + `stair report-train`: vendor `stair/external/silt/src/` (training logic untouched), add a local single-machine `torchrun` launcher, and the terminal training-curve/diagnostics report.
4. Eval (`stair eval`) + `stair report-eval`: local inference against a trained checkpoint, retrieval metrics (recall/precision/F1/MRR/NDCG@K, hallucination rate — reusing the existing `compute_metrics` logic), and the terminal eval report.

Each will be written as its own plan document once the prior plan is implemented, since exact interfaces (e.g. the data pipeline's output schema) are locked in during implementation, not guessed in advance.
