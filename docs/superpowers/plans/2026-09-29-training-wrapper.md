# STAIR Training Wrapper (`stair train` + `stair report-train`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `stair train` (fine-tunes a model on Plan 2's `train.jsonl`/`val.jsonl` via the vendored, unmodified SiLT LoRA/SFT trainer) and `stair report-train` (reads the trainer's `trainer_state.json` and prints terminal loss/eval curves plus rule-based diagnostics and recommendations).

**Architecture:** SiLT's training code (`stair/external/silt/src/`) is vendored verbatim into `src/stair/vendor/silt/` — every file byte-identical to the source, bare imports untouched, so it keeps working exactly as it does today. A new, original adapter layer (`src/stair/core/train/`) translates our resolved config into the flat `--key value` CLI arguments the vendored trainer's own argument parser already expects (confirmed by reading `arguments.py` directly: config YAML files are optional, and everything can be passed as CLI flags whose values are JSON-decoded when they're lists/dicts) and provides one new, original post-processing function that turns a Plan-2-produced `(question, reference)` pair plus `toc.json` into the chat-formatted prompt/completion the trainer needs — this is new code we write, not a modification of anything vendored. A local launcher runs the vendored trainer via `torchrun` as a subprocess. `stair report-train` is entirely new code reading the trainer's standard HF `trainer_state.json` output.

**Tech Stack:** Python 3.12, `torch`/`transformers`/`peft`/`trl`/`accelerate`/`datasets`/`pydantic`/`pandas`/`jsonlines`/`tqdm`/`llama-index-core`/`ipython` (SiLT's own load-bearing dependencies, confirmed by reading its imports directly — `deepspeed`, `wandb`, `tensorboardx`, `langchain`, `chromadb` etc. are NOT imported by the files being vendored and are not added), `plotext` (new, for terminal curve rendering), `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-29-external-pipeline-design.md`

**Plan 1 (merged):** `docs/superpowers/plans/2026-09-29-pipeline-scaffolding-and-config.md` — provides `stair.config.resolve_config`, `stair.run_header.print_run_header`, the `stair` CLI skeleton, and `configs/base.yaml`.

**Plan 2 (merged):** `docs/superpowers/plans/2026-09-29-data-pipeline.md` — provides `stair prepare-data`, whose output directory (passed to `stair train` as `--data`) contains `train.jsonl`, `val.jsonl`, `test.jsonl`, `toc.json`, `docs.jsonl`, `config.json`.

## Global Constraints

- **The vendored trainer requires a real CUDA GPU. There is no CPU path.** Confirmed by reading the vendored source directly: `training_fsdp_trainer.py` calls `model.to("cuda")` unconditionally, uses `attn_implementation="flash_attention_2"`, and `train_utils.py` hardcodes `bf16=True`/`optim="adamw_torch_fused"`. This is a property of code we are not modifying. Every task in this plan must be testable without a GPU EXCEPT the actual `torchrun` subprocess invocation itself, which is tested only by verifying the exact command/argument list a fake subprocess runner receives (dependency injection, same pattern Plan 2 used for the LLM client) — never by actually running it. The real end-to-end training run is exercised manually on a GPU machine, not in CI.
- No `logging` module — `print()` only. No one-liners — unpack comprehensions into explicit loops. No type hints. Full-word names. 4-space indent. (code-style.md — every task in Plan 2 that violated these was caught in review; do not repeat that here.)
- Every step of `stair train` prints what it's doing and every output path, per traceability.md. `stair train` must write a run manifest — note the vendored trainer already does this automatically (`get_config()` in `arguments.py` writes `<save_path>/training_config.json` itself whenever `mode == Mode.training`), so `run_train` does not need to duplicate it — just confirm/print that path once training starts.
- No absolute paths in checked-in code.
- The vendored SiLT files under `src/stair/vendor/silt/` must remain byte-for-byte identical to `stair/external/silt/src/` except for the one dead, inert commented-out line containing a real-looking hardcoded token (`training_fsdp_trainer.py`'s commented `os.environ["HF_TOKEN"] = "hf_..."`), which is deleted as a credential-hygiene cleanup, not a logic change.
- The new post-processing function (Task 2) is original code we own, not a modification of vendored SiLT — it lives entirely under `src/stair/core/train/`, referenced by the vendored trainer only through its existing, unmodified `class_path` dotted-import mechanism.

## Review Focus

- **`--data` directory missing `train.jsonl`/`val.jsonl`/`toc.json`** (e.g. a user points `stair train` at a raw corpus directory instead of `prepare-data`'s output): must fail with a clear message before attempting to invoke the trainer, not a confusing error from deep inside the vendored code.
- **A `reference` id in a QA pair that isn't in `toc.json`'s id list**: the new post-process function must fail loudly (a clear `KeyError`-adjacent message naming the bad id) rather than silently producing a garbage training example — this can legitimately happen if a user hand-edits `train.jsonl` or mixes data from two different `prepare-data` runs.
- **`trainer_state.json` with no eval entries at all** (e.g. a very short debug run, or `eval_strategy` misconfigured): `report-train` must say so plainly and skip eval-dependent diagnostics, not crash trying to compute a diagnostic over an empty list.
- **`trainer_state.json` with only one eval entry**: overfitting/plateau diagnostics need at least 2 points to say anything about a trend — `report-train` must recognize "not enough data yet" rather than reporting a false positive/negative trend off a single point.
- **No checkpoint directory found under the run dir at all** (e.g. `report-train` run against a directory training never actually wrote to, or training crashed before its first checkpoint): must print a clear "no checkpoints found" message and exit non-zero, not a raw `FileNotFoundError` traceback.

---

## File Structure

```
src/stair/
  vendor/
    silt/                        # verbatim copy of stair/external/silt/src/, unmodified except the one dead HF_TOKEN line
      arguments.py
      constants.py
      trainers.py
      train_utils.py
      training_fsdp_trainer.py
      data/
        __init__.py
        config.py
        dataset.py
        load_data.py
        post_process_data.py
  core/
    train/
      __init__.py
      io_functions.py             # NEW: prepare_input_output_stair post-process function
      args_builder.py             # NEW: build_training_cli_args(config, data_dir, run_dir) -> list[str]
    report/
      __init__.py
      train_report.py             # NEW: checkpoint discovery, log-history loading, diagnostics, curve rendering
  train.py                        # NEW: run_train(data_dir, run_dir, config, subprocess_run=subprocess.run) -> dict
  cli.py                          # modified: train and report-train subcommands call the real implementations
tests/
  test_vendor_silt_importable.py
  test_io_functions.py
  test_args_builder.py
  test_train.py
  test_train_report.py
```

---

### Task 1: Vendor SiLT's training code

**Files:**
- Create: `src/stair/vendor/__init__.py`
- Create: `src/stair/vendor/silt/__init__.py`
- Create: `src/stair/vendor/silt/arguments.py` (verbatim copy)
- Create: `src/stair/vendor/silt/constants.py` (verbatim copy)
- Create: `src/stair/vendor/silt/trainers.py` (verbatim copy)
- Create: `src/stair/vendor/silt/train_utils.py` (verbatim copy)
- Create: `src/stair/vendor/silt/training_fsdp_trainer.py` (verbatim copy, minus one dead commented line — see below)
- Create: `src/stair/vendor/silt/data/__init__.py` (verbatim copy)
- Create: `src/stair/vendor/silt/data/config.py` (verbatim copy)
- Create: `src/stair/vendor/silt/data/dataset.py` (verbatim copy)
- Create: `src/stair/vendor/silt/data/load_data.py` (verbatim copy)
- Create: `src/stair/vendor/silt/data/post_process_data.py` (verbatim copy)
- Test: `tests/test_vendor_silt_importable.py`

**Interfaces:**
- Consumes: nothing from prior plans.
- Produces: an importable `src/stair/vendor/silt/` directory whose modules can be loaded by inserting that directory onto `sys.path` (matching how SiLT's own `torchrun .../training_fsdp_trainer.py` invocation makes its bare imports resolve — do not try to make these modules importable as `stair.vendor.silt.arguments` via normal package imports, since the files use bare imports like `from constants import ...` internally that only work when their own directory is first on `sys.path`). Task 4's launcher relies on this: it invokes `training_fsdp_trainer.py` as a subprocess script, not as an imported module, so no other task imports these vendored modules directly except this task's own smoke test.

- [ ] **Step 1: Copy the vendored files**

Run these commands (source path may differ if you're working from a fresh checkout — the source is the SiLT git submodule already checked out under the outer `stair/` directory at repo root, alongside this `stair-pipeline` package's own root):

```bash
mkdir -p src/stair/vendor/silt/data
cp ../stair/external/silt/src/arguments.py src/stair/vendor/silt/arguments.py
cp ../stair/external/silt/src/constants.py src/stair/vendor/silt/constants.py
cp ../stair/external/silt/src/trainers.py src/stair/vendor/silt/trainers.py
cp ../stair/external/silt/src/train_utils.py src/stair/vendor/silt/train_utils.py
cp ../stair/external/silt/src/training_fsdp_trainer.py src/stair/vendor/silt/training_fsdp_trainer.py
cp ../stair/external/silt/src/data/__init__.py src/stair/vendor/silt/data/__init__.py
cp ../stair/external/silt/src/data/config.py src/stair/vendor/silt/data/config.py
cp ../stair/external/silt/src/data/dataset.py src/stair/vendor/silt/data/dataset.py
cp ../stair/external/silt/src/data/load_data.py src/stair/vendor/silt/data/load_data.py
cp ../stair/external/silt/src/data/post_process_data.py src/stair/vendor/silt/data/post_process_data.py
touch src/stair/vendor/__init__.py src/stair/vendor/silt/__init__.py
```

If `../stair/external/silt/src/` is not present relative to the repo root, run `git submodule update --init --recursive` from `../stair` (the outer, untracked nested repo containing the `external/silt` submodule) first, then retry the copy.

- [ ] **Step 2: Remove the one dead credential-looking line**

Open `src/stair/vendor/silt/training_fsdp_trainer.py` and find the commented-out line matching `# os.environ["HF_TOKEN"] = ...` (search for `HF_TOKEN`). Delete that single line. Do not change anything else in the file — every other line, including other comments, stays byte-identical to the source.

- [ ] **Step 3: Write the smoke test**

Create `tests/test_vendor_silt_importable.py`:

```python
import subprocess
import sys
from pathlib import Path


VENDOR_SILT_DIR = Path(__file__).resolve().parents[1] / "src" / "stair" / "vendor" / "silt"


def test_vendored_silt_modules_import_without_error():
    import_check_script = (
        "import arguments\n"
        "import constants\n"
        "import trainers\n"
        "import train_utils\n"
        "print('IMPORTS_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", import_check_script],
        cwd=str(VENDOR_SILT_DIR),
        capture_output=True,
        text=True,
    )
    assert "IMPORTS_OK" in result.stdout, result.stderr


def test_vendored_training_script_shows_no_hardcoded_hf_token():
    training_script_path = VENDOR_SILT_DIR / "training_fsdp_trainer.py"
    training_script_text = training_script_path.read_text()
    assert "HF_TOKEN" not in training_script_text


def test_vendored_arguments_module_exposes_get_args():
    import_check_script = (
        "import arguments\n"
        "assert hasattr(arguments, 'get_args')\n"
        "assert hasattr(arguments, 'TrainingArgs')\n"
        "print('INTERFACE_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", import_check_script],
        cwd=str(VENDOR_SILT_DIR),
        capture_output=True,
        text=True,
    )
    assert "INTERFACE_OK" in result.stdout, result.stderr
```

- [ ] **Step 4: Add the confirmed load-bearing dependencies**

Run: `uv add torch transformers peft trl accelerate datasets pandas jsonlines tqdm llama-index-core ipython plotext`

(`pydantic` and `pyyaml` are already dependencies from Plan 1. `plotext` is new, for Task 5's terminal curve rendering, not used by the vendored code itself.)

- [ ] **Step 5: Run the smoke test**

Run: `uv run pytest tests/test_vendor_silt_importable.py -v`
Expected: PASS (3 tests). If an import fails with `ModuleNotFoundError` for something other than the packages just installed, report BLOCKED rather than guessing at additional dependencies — the Global Constraints section names the confirmed complete set; a missing one means the vendored copy diverged from the source and needs re-checking against Step 1, not a silent `uv add` of something unconfirmed.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS (all tests from Plans 1 and 2, plus this task's 3 new tests)

- [ ] **Step 7: Commit**

```bash
git add src/stair/vendor/ tests/test_vendor_silt_importable.py pyproject.toml uv.lock
git commit -m "feat: vendor SiLT training code verbatim"
```

---

### Task 2: STAIR post-processing function (`core/train/io_functions.py`)

**Files:**
- Create: `src/stair/core/train/__init__.py`
- Create: `src/stair/core/train/io_functions.py`
- Test: `tests/test_io_functions.py`

**Interfaces:**
- Consumes: nothing from Task 1 directly (this is new code, called only by the vendored trainer's `class_path` dotted-import mechanism at actual training time — never imported directly by Task 1's vendored files).
- Produces: `stair.core.train.io_functions.prepare_input_output_stair(data_config, examples, tokenizer, split, system_prompt, toc_json, user_prompt, output_col, num_samples=1.0, repetitions=1, **kwargs) -> list[dict]`. Matches the exact positional-argument calling convention the vendored `post_process_data.py` uses (confirmed by reading its source: it calls `function_name(data_config, examples, tokenizer, split, **function_args)` where `function_args` is the `init_args` dict from the dataset's `post_process_functions` config entry, with `data_config`/`examples`/`tokenizer`/`split` explicitly stripped out of `function_args` first if present, so those four names must be positional parameters here, not keyword-only). `tokenizer` is an already-instantiated tokenizer object (has `.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`); this function never loads a tokenizer itself. Each returned dict has `messages` (list of role/content dicts), `prompt` (the rendered chat template string), `completion` (a Python-list-of-strings literal, e.g. `'\n["1.2 Getting Started"]'`, matching the format the eval-time parser (Plan 4) will expect), `prompt_type`, `meta_data`. Raises `KeyError` with a message naming the bad id if any example's `reference` contains an id not present in `toc_json`'s `table_of_contents`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_io_functions.py`:

```python
import json
import textwrap

import pytest

from stair.core.train.io_functions import prepare_input_output_stair


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        rendered_parts = []
        for message in messages:
            rendered_parts.append(f"<{message['role']}>{message['content']}")
        return "\n".join(rendered_parts)


def write_toc_json(tmp_path):
    toc = {
        "title": "Test Book",
        "table_of_contents": [
            {"id": "1", "section_num": "1", "title": "Introduction", "leaf": True},
            {"id": "2", "section_num": "2", "title": "Getting Started", "leaf": True},
        ],
    }
    toc_path = tmp_path / "toc.json"
    toc_path.write_text(json.dumps(toc))
    return toc_path


def test_prepare_input_output_stair_builds_prompt_and_completion(tmp_path):
    toc_path = write_toc_json(tmp_path)
    examples = [
        {"question": "What is this book about?", "reference": ["1"]},
    ]

    result = prepare_input_output_stair(
        data_config=None,
        examples=examples,
        tokenizer=FakeTokenizer(),
        split="train",
        system_prompt="You are a helpful assistant.",
        toc_json=str(toc_path),
        user_prompt="Book: {title}\nTOC:\n{toc}\nQuestion: {question}\nAnswer:",
        output_col="{reference}",
    )

    assert len(result) == 1
    assert result[0]["messages"][0]["role"] == "system"
    assert result[0]["messages"][1]["content"].startswith("Book: Test Book")
    assert "<system>" in result[0]["prompt"]
    assert result[0]["completion"] == '\n["1 Introduction"]'


def test_prepare_input_output_stair_raises_on_unknown_reference_id(tmp_path):
    toc_path = write_toc_json(tmp_path)
    examples = [
        {"question": "What is this book about?", "reference": ["99"]},
    ]

    with pytest.raises(KeyError, match="99"):
        prepare_input_output_stair(
            data_config=None,
            examples=examples,
            tokenizer=FakeTokenizer(),
            split="train",
            system_prompt="You are a helpful assistant.",
            toc_json=str(toc_path),
            user_prompt="Book: {title}\nTOC:\n{toc}\nQuestion: {question}\nAnswer:",
            output_col="{reference}",
        )


def test_prepare_input_output_stair_handles_multiple_reference_ids(tmp_path):
    toc_path = write_toc_json(tmp_path)
    examples = [
        {"question": "Compare both sections.", "reference": ["1", "2"]},
    ]

    result = prepare_input_output_stair(
        data_config=None,
        examples=examples,
        tokenizer=FakeTokenizer(),
        split="train",
        system_prompt="You are a helpful assistant.",
        toc_json=str(toc_path),
        user_prompt="Question: {question}",
        output_col="{reference}",
    )

    assert result[0]["completion"] == '\n["1 Introduction", "2 Getting Started"]'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_io_functions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stair.core.train'`

- [ ] **Step 3: Write `src/stair/core/train/io_functions.py`**

```python
import ast
import json


def load_toc(toc_json_path):
    with open(toc_json_path, "r") as toc_file:
        toc = json.load(toc_file)

    title = toc["title"]
    toc_lines = []
    id_to_title_map = {}
    for node in toc["table_of_contents"]:
        toc_lines.append(f"{node['section_num']} {node['title']}")
        id_to_title_map[node["id"]] = f"{node['section_num']} {node['title']}"

    return title, "\n".join(toc_lines), id_to_title_map


def prepare_input_output_stair(
    data_config,
    examples,
    tokenizer,
    split,
    system_prompt,
    toc_json,
    user_prompt,
    output_col,
    num_samples=1.0,
    repetitions=1,
    **kwargs
):
    book_title, toc_text, id_to_title_map = load_toc(toc_json)

    if num_samples <= 1.0:
        keep_until_index = int(num_samples * len(examples))
    else:
        keep_until_index = int(num_samples)

    prepared_examples = []
    for example_index, example in enumerate(examples):
        if example_index >= keep_until_index:
            break

        prompt_fields = dict(example)
        prompt_fields.update(kwargs)
        prompt_fields["title"] = book_title
        prompt_fields["toc"] = toc_text

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(**prompt_fields)},
        ]

        reference_ids_text = output_col.format(**prompt_fields).lstrip("\n")
        reference_ids = ast.literal_eval(reference_ids_text)

        reference_titles = []
        for reference_id in reference_ids:
            if reference_id not in id_to_title_map:
                raise KeyError(
                    f"reference id {reference_id!r} not found in table of contents at {toc_json}"
                )
            reference_titles.append(id_to_title_map[reference_id])

        quoted_titles = []
        for reference_title in reference_titles:
            quoted_titles.append(f'"{reference_title}"')
        completion = "\n[" + ", ".join(quoted_titles) + "]"

        prepared_example = {}
        prepared_example["messages"] = messages
        prepared_example["prompt"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prepared_example["completion"] = completion
        prepared_example["prompt_type"] = prompt_fields.get("prompt_type", "generic")
        prepared_example["meta_data"] = prompt_fields
        prepared_examples.append(prepared_example)

    if repetitions > 1:
        repeated_examples = []
        for prepared_example in prepared_examples:
            for _ in range(repetitions):
                repeated_examples.append(dict(prepared_example))
        prepared_examples = repeated_examples

    return prepared_examples
```

Create `src/stair/core/train/__init__.py` (empty file).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_io_functions.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/stair/core/train/__init__.py src/stair/core/train/io_functions.py tests/test_io_functions.py
git commit -m "feat: add STAIR post-processing function for training data"
```

---

### Task 3: Config-to-CLI-args adapter (`core/train/args_builder.py`)

**Files:**
- Create: `src/stair/core/train/args_builder.py`
- Test: `tests/test_args_builder.py`

**Interfaces:**
- Consumes: the resolved config dict shape from `configs/base.yaml` (Plan 1's `resolve_config`), plus a `data_dir` (Plan 2's `prepare-data` output directory) and a `run_dir` (this run's output directory).
- Produces: `stair.core.train.args_builder.build_training_cli_args(config, data_dir, run_dir) -> list[str]`. A flat list of `--key`, `value` pairs (always an even count — the vendored parser drops a trailing unpaired argument, confirmed by reading `read_defaults()`'s `while ind < len(unknown_args) - 1` loop) suitable for passing directly as extra CLI arguments to the vendored `training_fsdp_trainer.py`. Used by Task 4.

- [ ] **Step 1: Write the failing test**

Create `tests/test_args_builder.py`:

```python
import json

from stair.config import resolve_config
from stair.core.train.args_builder import build_training_cli_args


def args_list_to_dict(args_list):
    args_as_dict = {}
    index = 0
    while index < len(args_list):
        key = args_list[index].lstrip("-")
        value = args_list[index + 1]
        args_as_dict[key] = value
        index += 2
    return args_as_dict


def test_build_training_cli_args_has_an_even_number_of_entries():
    config = resolve_config(override_path=None)
    cli_args = build_training_cli_args(config, data_dir="data/example_book", run_dir="runs/example")
    assert len(cli_args) % 2 == 0


def test_build_training_cli_args_sets_required_fields():
    config = resolve_config(override_path=None)
    cli_args = build_training_cli_args(config, data_dir="data/example_book", run_dir="runs/example")
    args_as_dict = args_list_to_dict(cli_args)

    assert args_as_dict["model_name"] == config["model"]["name"]
    assert args_as_dict["save_path"] == "runs/example"
    assert args_as_dict["batch_size_per_gpu"] == str(config["training"]["batch_size_per_gpu"])
    assert args_as_dict["report_to"] == "none"


def test_build_training_cli_args_datasets_field_is_valid_json_with_one_dataset():
    config = resolve_config(override_path=None)
    cli_args = build_training_cli_args(config, data_dir="data/example_book", run_dir="runs/example")
    args_as_dict = args_list_to_dict(cli_args)

    datasets = json.loads(args_as_dict["datasets"])
    assert len(datasets) == 1
    assert datasets[0]["data_class"] == "JSONLinesDataset"
    assert datasets[0]["data_path"] == "data/example_book"
    assert datasets[0]["files"]["train"] == ["train.jsonl"]
    assert datasets[0]["files"]["val"] == ["val.jsonl"]
    assert datasets[0]["files"]["test"] == ["test.jsonl"]


def test_build_training_cli_args_dataset_post_process_function_points_at_io_functions():
    config = resolve_config(override_path=None)
    cli_args = build_training_cli_args(config, data_dir="data/example_book", run_dir="runs/example")
    args_as_dict = args_list_to_dict(cli_args)

    datasets = json.loads(args_as_dict["datasets"])
    post_process_functions = datasets[0]["post_process_functions"]
    assert len(post_process_functions) == 1
    assert post_process_functions[0]["class_path"] == "stair.core.train.io_functions.prepare_input_output_stair"
    assert post_process_functions[0]["init_args"]["toc_json"] == "data/example_book/toc.json"
    assert post_process_functions[0]["init_args"]["output_col"] == "{reference}"


def test_build_training_cli_args_lora_settings_come_from_config():
    config = resolve_config(override_path=None)
    cli_args = build_training_cli_args(config, data_dir="data/example_book", run_dir="runs/example")
    args_as_dict = args_list_to_dict(cli_args)

    assert args_as_dict["lora_rank"] == str(config["model"]["lora"]["rank"])
    assert args_as_dict["lora_alpha"] == str(config["model"]["lora"]["alpha"])
    assert args_as_dict["lora_dropout"] == str(config["model"]["lora"]["dropout"])
    lora_target_modules = json.loads(args_as_dict["lora_target_modules"])
    assert lora_target_modules == config["model"]["lora"]["target_modules"]


def test_build_training_cli_args_optimizer_field_carries_configured_learning_rate():
    config = resolve_config(override_path=None)
    cli_args = build_training_cli_args(config, data_dir="data/example_book", run_dir="runs/example")
    args_as_dict = args_list_to_dict(cli_args)

    optimizer = json.loads(args_as_dict["optimizer"])
    assert optimizer["lr"] == config["training"]["learning_rate"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_args_builder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stair.core.train.args_builder'`

- [ ] **Step 3: Write `src/stair/core/train/args_builder.py`**

```python
import json


def build_dataset_config(config, data_dir):
    system_prompt = (
        "You are a helpful assistant tasked with selecting the most relevant "
        "sections from a book's table of contents that best answers a user query."
    )
    user_prompt = (
        "Book: {title}\n"
        "Select all relevant sections from the table of contents below that can "
        "help answer the user query. Return the output only as a Python list of "
        "strings, where each string follows the format:\n"
        '"section_num title"\n'
        'Example Output: ["1.1 Section Name", "2.3 Another Section"]\n'
        "Do not include any explanations or additional text.\n\n"
        "Table of Contents:\n{toc}\n\n"
        "Query:\n{question}\n\n"
        "Relevant Sections:"
    )

    dataset_config = {
        "data_class": "JSONLinesDataset",
        "data_name": "stair",
        "data_path": data_dir,
        "files": {
            "train": ["train.jsonl"],
            "val": ["val.jsonl"],
            "test": ["test.jsonl"],
        },
        "post_process_functions": [
            {
                "class_path": "stair.core.train.io_functions.prepare_input_output_stair",
                "init_args": {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "toc_json": data_dir + "/toc.json",
                    "output_col": "{reference}",
                },
            }
        ],
    }
    return dataset_config


def build_optimizer_config(training_config):
    optimizer_config = {
        "lr": training_config["learning_rate"],
        "weight_decay": 0.05,
        "betas": [0.9, 0.95],
        "eps": 1.0e-10,
    }
    return optimizer_config


def build_training_cli_args(config, data_dir, run_dir):
    model_config = config["model"]
    lora_config = model_config["lora"]
    training_config = config["training"]

    dataset_config = build_dataset_config(config, data_dir)

    field_values = {
        "model_name": model_config["name"],
        "save_path": run_dir,
        "batch_size_per_gpu": training_config["batch_size_per_gpu"],
        "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
        "num_train_epochs": training_config["num_train_epochs"],
        "max_seq_length": training_config["max_seq_length"],
        "eval_strategy": training_config["eval_strategy"],
        "save_strategy": training_config["eval_strategy"],
        "eval_interval": training_config["eval_interval"],
        "save_interval": training_config["save_interval"],
        "save_total_limit": training_config["save_total_limit"],
        "early_stopping": training_config["early_stopping"],
        "early_stopping_patience": training_config["early_stopping_patience"],
        "num_gpus": training_config["num_gpus"],
        "lora_rank": lora_config["rank"],
        "lora_alpha": lora_config["alpha"],
        "lora_dropout": lora_config["dropout"],
        "lora_target_modules": lora_config["target_modules"],
        "optimizer": build_optimizer_config(training_config),
        "lr_schedule": training_config["lr_schedule"],
        "warmup_steps": training_config["warmup_steps"],
        "report_to": "none",
        "datasets": [dataset_config],
    }

    cli_args = []
    for field_name, field_value in field_values.items():
        cli_args.append("--" + field_name)
        if isinstance(field_value, (list, dict)):
            cli_args.append(json.dumps(field_value))
        else:
            cli_args.append(str(field_value))

    return cli_args
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_args_builder.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/stair/core/train/args_builder.py tests/test_args_builder.py
git commit -m "feat: add config-to-CLI-args adapter for the vendored trainer"
```

---

### Task 4: `stair train` launcher

**Files:**
- Create: `src/stair/train.py`
- Modify: `src/stair/cli.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `build_training_cli_args` (Task 3), `resolve_config`/`print_run_header` (Plan 1).
- Produces: `stair.train.run_train(data_dir, run_dir, config, subprocess_run=None) -> dict` — the testable entry point. Returns `{"command": list_of_strings, "returncode": int}`. When `subprocess_run` is `None`, uses the real `subprocess.run`; tests inject a fake. Raises `FileNotFoundError` with a clear message if `data_dir` is missing any of `train.jsonl`, `val.jsonl`, `toc.json` before building any CLI args or touching the vendored trainer. `cli.py`'s `train` subcommand gains a `--data` argument and calls this function.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train.py`:

```python
import json
from pathlib import Path

import pytest

from stair.config import resolve_config
from stair.train import run_train


def make_complete_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text('{"question": "q", "answer": "a", "reference": ["1"]}\n')
    (data_dir / "val.jsonl").write_text("")
    (data_dir / "test.jsonl").write_text("")
    toc = {"title": "Test Book", "table_of_contents": [{"id": "1", "section_num": "1", "title": "Intro", "leaf": True}]}
    (data_dir / "toc.json").write_text(json.dumps(toc))
    return data_dir


def make_fake_subprocess_run(returncode):
    calls = []

    def fake_subprocess_run(command, **kwargs):
        calls.append(command)

        class FakeCompletedProcess:
            pass

        completed_process = FakeCompletedProcess()
        completed_process.returncode = returncode
        return completed_process

    fake_subprocess_run.calls = calls
    return fake_subprocess_run


def test_run_train_raises_when_train_jsonl_missing(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "val.jsonl").write_text("")
    (data_dir / "toc.json").write_text("{}")
    run_dir = tmp_path / "run"
    config = resolve_config(override_path=None)

    with pytest.raises(FileNotFoundError, match="train.jsonl"):
        run_train(str(data_dir), str(run_dir), config, subprocess_run=make_fake_subprocess_run(0))


def test_run_train_raises_when_toc_json_missing(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("")
    (data_dir / "val.jsonl").write_text("")
    run_dir = tmp_path / "run"
    config = resolve_config(override_path=None)

    with pytest.raises(FileNotFoundError, match="toc.json"):
        run_train(str(data_dir), str(run_dir), config, subprocess_run=make_fake_subprocess_run(0))


def test_run_train_invokes_torchrun_against_the_vendored_trainer(tmp_path):
    data_dir = make_complete_data_dir(tmp_path)
    run_dir = tmp_path / "run"
    config = resolve_config(override_path=None)
    fake_subprocess_run = make_fake_subprocess_run(0)

    result = run_train(str(data_dir), str(run_dir), config, subprocess_run=fake_subprocess_run)

    assert result["returncode"] == 0
    assert len(fake_subprocess_run.calls) == 1
    invoked_command = fake_subprocess_run.calls[0]
    assert invoked_command[0] == "torchrun"
    assert "--nnodes=1" in invoked_command
    assert any("training_fsdp_trainer.py" in part for part in invoked_command)
    assert "--save_path" in invoked_command


def test_run_train_returns_nonzero_returncode_without_raising(tmp_path):
    data_dir = make_complete_data_dir(tmp_path)
    run_dir = tmp_path / "run"
    config = resolve_config(override_path=None)

    result = run_train(str(data_dir), str(run_dir), config, subprocess_run=make_fake_subprocess_run(1))

    assert result["returncode"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stair.train'`

- [ ] **Step 3: Write `src/stair/train.py`**

```python
import subprocess
from pathlib import Path

from stair.core.train.args_builder import build_training_cli_args
from stair.run_header import print_run_header


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_TRAINER_SCRIPT = REPO_ROOT / "src" / "stair" / "vendor" / "silt" / "training_fsdp_trainer.py"


def check_required_data_files(data_dir):
    data_dir_path = Path(data_dir)
    required_file_names = ["train.jsonl", "val.jsonl", "toc.json"]
    for required_file_name in required_file_names:
        required_file_path = data_dir_path / required_file_name
        if not required_file_path.exists():
            raise FileNotFoundError(
                f"Expected {required_file_name} under {data_dir} "
                f"(run `stair prepare-data` first to produce it)"
            )


def run_train(data_dir, run_dir, config, subprocess_run=None):
    if subprocess_run is None:
        subprocess_run = subprocess.run

    print_run_header("stair train", config)

    check_required_data_files(data_dir)

    Path(run_dir).mkdir(parents=True, exist_ok=True)

    training_cli_args = build_training_cli_args(config, data_dir, run_dir)

    num_gpus = config["training"]["num_gpus"]
    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=" + str(num_gpus),
        str(VENDORED_TRAINER_SCRIPT),
    ]
    command.extend(training_cli_args)

    print(f"Running: {' '.join(command)}")
    completed_process = subprocess_run(command, cwd=str(VENDORED_TRAINER_SCRIPT.parent))

    print(f"Training run finished with exit code {completed_process.returncode}")
    print(f"Checkpoints and training_config.json saved under: {run_dir}")

    return {"command": command, "returncode": completed_process.returncode}
```

- [ ] **Step 4: Wire `train` into `cli.py`**

In `src/stair/cli.py`, add the import:

```python
from stair.train import run_train
```

Add a `--data` argument to the `train` subcommand (in `add_train_subcommand`):

```python
    parser.add_argument(
        "--data", required=True,
        help="Directory produced by `stair prepare-data` (must contain train.jsonl, val.jsonl, toc.json)",
    )
```

In `main()`, add a branch for `train` alongside the existing `prepare-data` branch:

```python
    if args.subcommand == "train":
        try:
            result = run_train(args.data, args.out, resolved_config)
        except (OSError, ValueError) as error:
            print(f"stair train: {error}", file=sys.stderr)
            return 1
        return 0 if result["returncode"] == 0 else 1
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_train.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS (all tests from every prior task and plan)

- [ ] **Step 7: Commit**

```bash
git add src/stair/train.py src/stair/cli.py tests/test_train.py
git commit -m "feat: wire stair train to launch the vendored trainer locally"
```

---

### Task 5: `stair report-train`

**Files:**
- Create: `src/stair/core/report/__init__.py`
- Create: `src/stair/core/report/train_report.py`
- Modify: `src/stair/cli.py`
- Test: `tests/test_train_report.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-4 directly (reads plain `trainer_state.json` files from disk, a format Task 4's real runs will eventually produce, but this task's own tests build fixture files directly — no dependency on actually running training).
- Produces: `stair.core.report.train_report.find_latest_checkpoint_state(run_dir) -> str` (path to the `trainer_state.json` with the highest numeric checkpoint step under `run_dir`; raises `FileNotFoundError` with a clear message if none found — excludes any non-numeric checkpoint directory name like `checkpoint-best`). `stair.core.report.train_report.load_log_history(trainer_state_path) -> list[dict]`. `stair.core.report.train_report.split_log_history(log_history) -> (list, list)` — returns `(train_entries, eval_entries)`, where train entries have a `"loss"` key and eval entries have an `"eval_loss"` key. `stair.core.report.train_report.diagnose(train_entries, eval_entries, report_config, num_train_epochs) -> list[str]` — a list of human-readable diagnostic lines (possibly empty), using `report_config` (the resolved config's `report` section: `overfitting_patience_evals`, `plateau_relative_improvement`, `plateau_patience_evals`). `stair.core.report.train_report.render_curves(train_entries, eval_entries) -> None` — prints terminal plots via `plotext`; does nothing but print "No eval data yet" if `eval_entries` is empty, and does nothing but print "No training data yet" if `train_entries` is empty. `cli.py`'s `report-train` subcommand calls all of the above in sequence.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_report.py`:

```python
import json

import pytest

from stair.core.report.train_report import (
    diagnose,
    find_latest_checkpoint_state,
    load_log_history,
    split_log_history,
)


REPORT_CONFIG = {
    "overfitting_patience_evals": 3,
    "plateau_relative_improvement": 0.01,
    "plateau_patience_evals": 5,
}


def write_trainer_state(path, log_history):
    path.write_text(json.dumps({"log_history": log_history}))


def test_find_latest_checkpoint_state_picks_highest_numeric_step(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoint-10").mkdir(parents=True)
    (run_dir / "checkpoint-30").mkdir(parents=True)
    (run_dir / "checkpoint-best").mkdir(parents=True)
    write_trainer_state(run_dir / "checkpoint-10" / "trainer_state.json", [])
    write_trainer_state(run_dir / "checkpoint-30" / "trainer_state.json", [])

    latest_path = find_latest_checkpoint_state(str(run_dir))

    assert "checkpoint-30" in latest_path


def test_find_latest_checkpoint_state_raises_when_none_found(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="no checkpoints"):
        find_latest_checkpoint_state(str(run_dir))


def test_load_log_history_returns_the_list(tmp_path):
    state_path = tmp_path / "trainer_state.json"
    write_trainer_state(state_path, [{"loss": 1.0, "step": 1, "epoch": 0.1}])

    log_history = load_log_history(str(state_path))

    assert log_history == [{"loss": 1.0, "step": 1, "epoch": 0.1}]


def test_split_log_history_separates_train_and_eval_entries():
    log_history = [
        {"loss": 1.0, "step": 1, "epoch": 0.1},
        {"eval_loss": 0.9, "step": 1, "epoch": 0.1},
        {"loss": 0.8, "step": 2, "epoch": 0.2},
        {"eval_loss": 0.7, "step": 2, "epoch": 0.2},
    ]

    train_entries, eval_entries = split_log_history(log_history)

    assert len(train_entries) == 2
    assert len(eval_entries) == 2
    assert all("loss" in entry for entry in train_entries)
    assert all("eval_loss" in entry for entry in eval_entries)


def test_diagnose_with_no_eval_entries_says_so_and_skips_eval_diagnostics():
    diagnostics = diagnose(
        train_entries=[{"loss": 1.0, "step": 1, "epoch": 0.1}],
        eval_entries=[],
        report_config=REPORT_CONFIG,
        num_train_epochs=3,
    )

    joined_diagnostics = " ".join(diagnostics)
    assert "no eval data" in joined_diagnostics.lower()


def test_diagnose_with_one_eval_entry_says_not_enough_data():
    diagnostics = diagnose(
        train_entries=[{"loss": 1.0, "step": 1, "epoch": 0.1}],
        eval_entries=[{"eval_loss": 0.9, "step": 1, "epoch": 0.1}],
        report_config=REPORT_CONFIG,
        num_train_epochs=3,
    )

    joined_diagnostics = " ".join(diagnostics)
    assert "not enough" in joined_diagnostics.lower()


def test_diagnose_flags_overfitting_when_eval_loss_rises_while_train_loss_falls():
    # eval loss falls for the first 2 evals, then rises for 3 in a row
    # (index 3, 4, 5) — long enough to trip overfitting_patience_evals=3
    train_entries = [
        {"loss": 0.85, "step": 1, "epoch": 1},
        {"loss": 0.70, "step": 2, "epoch": 2},
        {"loss": 0.55, "step": 3, "epoch": 3},
        {"loss": 0.40, "step": 4, "epoch": 4},
        {"loss": 0.25, "step": 5, "epoch": 5},
        {"loss": 0.10, "step": 6, "epoch": 6},
    ]
    eval_entries = [
        {"eval_loss": 1.0, "step": 1, "epoch": 1},
        {"eval_loss": 0.8, "step": 2, "epoch": 2},
        {"eval_loss": 0.6, "step": 3, "epoch": 3},
        {"eval_loss": 0.7, "step": 4, "epoch": 4},
        {"eval_loss": 0.9, "step": 5, "epoch": 5},
        {"eval_loss": 1.1, "step": 6, "epoch": 6},
    ]

    diagnostics = diagnose(train_entries, eval_entries, REPORT_CONFIG, num_train_epochs=6)

    joined_diagnostics = " ".join(diagnostics).lower()
    assert "overfit" in joined_diagnostics


def test_diagnose_flags_plateau_when_eval_loss_stops_improving():
    train_entries = [{"loss": 1.0 - (0.05 * step), "step": step, "epoch": step} for step in range(1, 8)]
    eval_entries = [{"eval_loss": 0.5, "step": step, "epoch": step} for step in range(1, 8)]

    diagnostics = diagnose(train_entries, eval_entries, REPORT_CONFIG, num_train_epochs=7)

    joined_diagnostics = " ".join(diagnostics).lower()
    assert "plateau" in joined_diagnostics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stair.core.report'`

- [ ] **Step 3: Write `src/stair/core/report/train_report.py`**

```python
import json
import re
from pathlib import Path

import plotext


def find_latest_checkpoint_state(run_dir):
    run_dir_path = Path(run_dir)
    checkpoint_dir_pattern = re.compile(r"^checkpoint-(\d+)$")

    numbered_checkpoints = []
    for candidate_dir in run_dir_path.glob("checkpoint-*"):
        match = checkpoint_dir_pattern.match(candidate_dir.name)
        if match is None:
            continue
        step_number = int(match.group(1))
        state_path = candidate_dir / "trainer_state.json"
        if state_path.exists():
            numbered_checkpoints.append((step_number, state_path))

    if not numbered_checkpoints:
        raise FileNotFoundError(f"no checkpoints with trainer_state.json found under {run_dir}")

    numbered_checkpoints.sort(key=lambda entry: entry[0])
    highest_step_path = numbered_checkpoints[-1][1]
    return str(highest_step_path)


def load_log_history(trainer_state_path):
    with open(trainer_state_path, "r") as trainer_state_file:
        trainer_state = json.load(trainer_state_file)
    return trainer_state["log_history"]


def split_log_history(log_history):
    train_entries = []
    eval_entries = []
    for entry in log_history:
        if "eval_loss" in entry:
            eval_entries.append(entry)
        elif "loss" in entry:
            train_entries.append(entry)
    return train_entries, eval_entries


def diagnose(train_entries, eval_entries, report_config, num_train_epochs):
    diagnostics = []

    if not train_entries:
        diagnostics.append("No training data found in this run's log history.")
        return diagnostics

    if not eval_entries:
        diagnostics.append(
            "No eval data found in this run's log history — cannot assess overfitting "
            "or plateauing. Check eval_strategy/eval_interval in your config."
        )
        return diagnostics

    if len(eval_entries) < 2:
        diagnostics.append(
            "Not enough eval data points yet (only 1 so far) to assess a trend. "
            "Check back after more evals have run."
        )
        return diagnostics

    overfitting_patience_evals = report_config["overfitting_patience_evals"]
    rising_streak = 0
    for entry_index in range(1, len(eval_entries)):
        if eval_entries[entry_index]["eval_loss"] > eval_entries[entry_index - 1]["eval_loss"]:
            rising_streak += 1
        else:
            rising_streak = 0
        if rising_streak >= overfitting_patience_evals:
            diagnostics.append(
                f"Overfitting: eval loss has risen for {rising_streak} consecutive evals "
                f"while train loss continues to fall. Try a lower learning rate, fewer "
                f"epochs, or more training data."
            )
            break

    plateau_patience_evals = report_config["plateau_patience_evals"]
    plateau_relative_improvement = report_config["plateau_relative_improvement"]
    if len(eval_entries) >= plateau_patience_evals:
        recent_eval_losses = []
        for entry in eval_entries[-plateau_patience_evals:]:
            recent_eval_losses.append(entry["eval_loss"])
        best_recent_loss = min(recent_eval_losses)
        earliest_recent_loss = recent_eval_losses[0]
        relative_improvement = (earliest_recent_loss - best_recent_loss) / earliest_recent_loss
        if relative_improvement < plateau_relative_improvement:
            diagnostics.append(
                f"Plateaued: eval loss has improved by less than "
                f"{plateau_relative_improvement * 100:.1f}% over the last "
                f"{plateau_patience_evals} evals. More epochs are unlikely to help much "
                f"without other changes."
            )

    last_train_epoch = train_entries[-1]["epoch"]
    if last_train_epoch < num_train_epochs:
        epochs_completed_fraction = last_train_epoch / num_train_epochs
        if epochs_completed_fraction < 0.9:
            diagnostics.append(
                f"Training stopped early at epoch {last_train_epoch:.1f} of "
                f"{num_train_epochs} configured — likely triggered by early stopping."
            )

    if not diagnostics:
        diagnostics.append("No issues detected: eval loss is trending down without a plateau.")

    return diagnostics


def render_curves(train_entries, eval_entries):
    if not train_entries:
        print("No training data yet.")
    else:
        train_steps = []
        train_losses = []
        for entry in train_entries:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        plotext.plot(train_steps, train_losses, label="train loss")

    if not eval_entries:
        print("No eval data yet.")
    else:
        eval_steps = []
        eval_losses = []
        for entry in eval_entries:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])
        plotext.plot(eval_steps, eval_losses, label="eval loss")

    if train_entries or eval_entries:
        plotext.title("Training curves")
        plotext.xlabel("step")
        plotext.ylabel("loss")
        plotext.show()
```

Create `src/stair/core/report/__init__.py` (empty file).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train_report.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Add the `plotext` dependency if not already present from Task 1**

Run: `uv add plotext` (no-op if Task 1 already added it)

- [ ] **Step 6: Wire `report-train` into `cli.py`**

In `src/stair/cli.py`, add the import:

```python
from stair.core.report.train_report import diagnose, find_latest_checkpoint_state, load_log_history, render_curves, split_log_history
```

Add a branch in `main()`:

```python
    if args.subcommand == "report-train":
        try:
            trainer_state_path = find_latest_checkpoint_state(args.run)
        except FileNotFoundError as error:
            print(f"stair report-train: {error}", file=sys.stderr)
            return 1
        log_history = load_log_history(trainer_state_path)
        train_entries, eval_entries = split_log_history(log_history)
        render_curves(train_entries, eval_entries)
        base_config = resolve_config(override_path=None)
        diagnostics = diagnose(
            train_entries, eval_entries, base_config["report"], base_config["training"]["num_train_epochs"]
        )
        print("")
        print("Diagnostics:")
        for diagnostic_line in diagnostics:
            print(f"- {diagnostic_line}")
        return 0
```

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS (all tests from every prior task and plan)

- [ ] **Step 8: Commit**

```bash
git add src/stair/core/report/ src/stair/cli.py tests/test_train_report.py pyproject.toml uv.lock
git commit -m "feat: add stair report-train terminal curves and diagnostics"
```

---

## Self-Review Notes

- **Spec coverage:** implements spec Goal 1 (train subcommand) and Goal 4 (terminal training report with diagnostics and recommendations) for the training half. `stair train`'s GPU-only nature is documented as a Global Constraint rather than silently assumed, consistent with the user's explicit decision to accept that limitation rather than modify vendored logic. Eval (`stair eval`, `stair report-eval`) remains Plan 4.
- **Placeholder scan:** no TBD/TODO. The GPU requirement is a real, tested-around limitation (every task's tests run without a GPU by injecting a fake subprocess runner at the one point that would need one), not a stub.
- **Type consistency:** `build_training_cli_args(config, data_dir, run_dir) -> list[str]` (Task 3) is consumed exactly as-is by `run_train` (Task 4). `find_latest_checkpoint_state`/`load_log_history`/`split_log_history`/`diagnose`/`render_curves` (Task 5) share a consistent `(train_entries, eval_entries)` shape derived from `split_log_history`'s output, used identically by `diagnose` and `render_curves` and by the `cli.py` wiring.
- **Review Focus:** all five items have a task and a test — missing `--data` files (Task 4, two dedicated tests), unknown reference id (Task 2, `test_prepare_input_output_stair_raises_on_unknown_reference_id`), no eval entries (Task 5, `test_diagnose_with_no_eval_entries_says_so_and_skips_eval_diagnostics`), only one eval entry (Task 5, `test_diagnose_with_one_eval_entry_says_not_enough_data`), no checkpoints found (Task 5, `test_find_latest_checkpoint_state_raises_when_none_found`).

## Next Plan

Plan 4: eval (`stair eval`) + `stair report-eval` — local inference against a trained checkpoint (loading the LoRA adapter produced under a run directory from this plan), retrieval metrics (recall/precision/F1/MRR/NDCG@K — already implemented in the existing `stair/experiments/rag_eda_benchmark/src/utils.py`'s `compute_metrics`, to be vendored/adapted the same way this plan vendored the trainer), hallucination rate, and the terminal eval report.
