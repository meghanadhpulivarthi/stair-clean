# STAIR Eval (`stair eval` + `stair report-eval`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `stair eval` (runs the trained model from a `stair train` run against Plan 2's `test.jsonl`, computing recall/precision/F1/MRR/NDCG@K and a hallucination rate) and `stair report-eval` (prints a terminal metrics table with weak-spot flags).

**Architecture:** A new `core/eval/` package with three small, independently-testable modules — retrieval metrics (ported from the existing, working `stair/experiments/rag_eda_benchmark/src/utils.py`'s `compute_metrics`), single-generation output parsing (a simplified, new adaptation of that same file's `parse_list_output` — the original was built around majority-voting over ~150 zero-shot samples per query via RITS/vLLM, which doesn't apply to evaluating one fine-tuned model with a single generation per query), and model loading/inference (new code, using `transformers`+`peft` directly, no vendored dependency). A shared prompt-template module is extracted from Plan 3's `args_builder.py` so eval-time prompts are byte-identical to what the model was trained on — this closes a deferred Minor finding from Plan 3's review (prompt templates were local strings, not shared) and is load-bearing here: evaluating with a different prompt than the one used in training would silently produce meaningless metrics.

**Tech Stack:** Python 3.12, `transformers`+`peft` (already dependencies from Plan 3, used directly this time — not through the vendored trainer), `pytest`. No new third-party dependencies.

**Spec:** `docs/superpowers/specs/2026-09-29-external-pipeline-design.md`

**Plan 1 (merged):** `docs/superpowers/plans/2026-09-29-pipeline-scaffolding-and-config.md` — `resolve_config`, `print_run_header`, CLI skeleton, `configs/base.yaml`.

**Plan 2 (merged):** `docs/superpowers/plans/2026-09-29-data-pipeline.md` — `stair prepare-data`, whose output directory (passed as `--data`) contains `test.jsonl` (each row `{"question", "answer", "reference": [id, ...]}`) and `toc.json`.

**Plan 3 (merged):** `docs/superpowers/plans/2026-09-29-training-wrapper.md` — `stair train`, whose output directory (passed as `--run`) contains `config.json` (the resolved training config, including `model.name`) and, after a successful run, `checkpoint-best/` — a standard PEFT adapter directory (confirmed by reading the vendored `trainers.py`: with a single LoRA adapter, `StackLoraTrainer.save_model()` falls through to the plain `SFTTrainer.save_model()`, which writes a normal `adapter_config.json`/`adapter_model.safetensors` pair directly into the given directory — loadable with `peft.PeftModel.from_pretrained(base_model, checkpoint_dir)`).

## Global Constraints

- No `logging` module — `print()` only. No one-liners. No type hints. Full-word names. 4-space indent. (code-style.md.)
- Every step of `stair eval` prints what it's doing and every output path, per traceability.md.
- No absolute paths in checked-in code.
- Model loading and real generation are never exercised in this plan's own tests — every test that touches `inference.py`'s generation path injects a fake `generate_call` (the same dependency-injection pattern Plan 2 used for the LLM client and Plan 3 used for `subprocess_run`). Loading a real base model plus adapter requires a network download and non-trivial time even for the small default model; it is exercised manually, not in CI, matching the precedent already set for `stair train`'s real GPU run.
- The eval-time prompt (system prompt + user prompt template) must be the exact same text the model was trained on. This plan extracts it into a shared module that both Plan 3's `args_builder.py` and this plan's eval code import — Plan 3's own tests must still pass unmodified after the extraction (Task 1 verifies this explicitly).

## Review Focus

- **A generated completion that isn't valid Python-list syntax at all** (e.g. the model outputs prose instead of a list, or trails off mid-generation): must be counted as a full hallucination for that example (zero predicted ids, contributes to hallucination rate), not crash the eval run.
- **A generated completion that's valid list syntax but contains a section title not in this book's TOC** (e.g. `["9.9 Nonexistent Section"]`): must be dropped from that example's predicted ids and counted toward the hallucination rate, not silently treated as a valid (wrong) prediction or crash on a `KeyError`.
- **A test example whose `reference` is an empty list** (shouldn't normally happen from `prepare-data`, but a hand-edited or malformed `test.jsonl` could produce one): `compute_metrics` must not divide by zero — confirm the ported metric functions handle `gold_ids == []` the same safe way the original code already does.
- **`stair report-eval` run against a run directory with no `eval/eval_metrics.json`** (e.g. `stair eval` was never run, or failed before writing output): must fail with a clear message, not a raw `FileNotFoundError` traceback.
- **`stair eval` run against a `--run` directory with no `checkpoint-best`** (e.g. training never completed, or `stair train` failed before its final save): must fail with a clear message before attempting to load a nonexistent adapter path, not a confusing error from deep inside `transformers`/`peft`.

---

## File Structure

```
src/stair/
  core/
    prompts.py                  # NEW: shared SYSTEM_PROMPT/USER_PROMPT, extracted from args_builder.py
    train/
      args_builder.py           # MODIFIED: imports from core/prompts.py instead of defining locally
    eval/
      __init__.py
      metrics.py                 # NEW: compute_dcg, compute_idcg, compute_metrics (ported from rag_eda_benchmark)
      parse_output.py            # NEW: parse_predicted_sections (new, simplified single-generation parser)
      inference.py                # NEW: load_finetuned_model, build_generate_call
  eval.py                        # NEW: run_eval(run_dir, data_dir, config, generate_call=None) -> dict
  cli.py                         # modified: eval and report-eval subcommands call the real implementations
tests/
  test_prompts.py
  test_metrics.py
  test_parse_output.py
  test_inference.py
  test_eval.py
  test_report_eval.py
```

---

### Task 1: Extract shared prompt templates

**Files:**
- Create: `src/stair/core/prompts.py`
- Modify: `src/stair/core/train/args_builder.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `stair.core.prompts.SYSTEM_PROMPT` (str), `stair.core.prompts.USER_PROMPT` (str, format-string with `{title}`, `{toc}`, `{question}` placeholders). Used by Task 4 (`eval.py`) and by the already-merged `args_builder.py` (modified in this task to import instead of define).

- [ ] **Step 1: Write the failing test**

Create `tests/test_prompts.py`:

```python
from stair.core.prompts import SYSTEM_PROMPT, USER_PROMPT


def test_system_prompt_is_a_non_empty_string():
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT.strip()) > 0


def test_user_prompt_has_expected_format_placeholders():
    rendered = USER_PROMPT.format(title="Test Book", toc="1 Introduction", question="What is this about?")
    assert "Test Book" in rendered
    assert "1 Introduction" in rendered
    assert "What is this about?" in rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stair.core.prompts'`

- [ ] **Step 3: Write `src/stair/core/prompts.py`**

Copy the exact text currently defined inline in `src/stair/core/train/args_builder.py`'s `build_dataset_config` function (read that file first to get the exact current wording) into this new module as two module-level constants:

```python
SYSTEM_PROMPT = (
    "You are a helpful assistant tasked with selecting the most relevant "
    "sections from a book's table of contents that best answers a user query."
)

USER_PROMPT = (
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
```

(If the text you read from `args_builder.py` differs from what's shown above in any way — even whitespace — use the actual current text verbatim. This block is a reference in case the file changed since this plan was written; the real source of truth is `args_builder.py`'s current content.)

- [ ] **Step 4: Update `args_builder.py` to import instead of define**

In `src/stair/core/train/args_builder.py`, replace the inline `system_prompt = (...)` and `user_prompt = (...)` string literals inside `build_dataset_config` with an import at the top of the file:

```python
from stair.core.prompts import SYSTEM_PROMPT, USER_PROMPT
```

and change the two local variable assignments inside `build_dataset_config` to reference the imported constants instead of defining new string literals (i.e. `system_prompt = SYSTEM_PROMPT` and `user_prompt = USER_PROMPT`, or use the imported names directly in the `dataset_config` dict construction — whichever reads more naturally given the surrounding code).

- [ ] **Step 5: Run test to verify it passes, and confirm Plan 3's tests are unaffected**

Run: `uv run pytest tests/test_prompts.py tests/test_args_builder.py -v`
Expected: PASS (2 new tests, and all of `test_args_builder.py`'s existing tests — including `test_build_training_cli_args_produces_a_valid_training_args_object`, which round-trips through the real vendored parser — still pass unchanged, since the actual string content is identical, just relocated).

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS (all tests from every prior plan, plus this task's 2 new tests)

- [ ] **Step 7: Commit**

```bash
git add src/stair/core/prompts.py src/stair/core/train/args_builder.py tests/test_prompts.py
git commit -m "refactor: extract shared prompt templates for train/eval consistency"
```

---

### Task 2: Retrieval metrics (`core/eval/metrics.py`)

**Files:**
- Create: `src/stair/core/eval/__init__.py`
- Create: `src/stair/core/eval/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces: `stair.core.eval.metrics.compute_metrics(predicted_ids, gold_ids, k) -> dict`, shape `{"recall@{k}": float, "match@{k}": int, "precision@{k}": float, "f1@{k}": float, "mrr@{k}": float, "ndcg@{k}": float}`. Ported from the existing `stair/experiments/rag_eda_benchmark/src/utils.py`'s `compute_metrics`/`compute_dcg`/`compute_idcg` (logic unchanged — only variable names may be clarified per code-style.md's full-word-names rule if the original used abbreviations; the arithmetic itself is verbatim). Used by Task 5 (`eval.py`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics.py`:

```python
from stair.core.eval.metrics import compute_metrics


def test_compute_metrics_all_predictions_correct():
    metrics = compute_metrics(predicted_ids=["1", "2"], gold_ids=["1", "2"], k=2)
    assert metrics["recall@2"] == 1.0
    assert metrics["match@2"] == 1
    assert metrics["precision@2"] == 1.0
    assert metrics["mrr@2"] == 1.0
    assert metrics["ndcg@2"] == 1.0


def test_compute_metrics_no_predictions_correct():
    metrics = compute_metrics(predicted_ids=["9", "8"], gold_ids=["1", "2"], k=2)
    assert metrics["recall@2"] == 0.0
    assert metrics["match@2"] == 0
    assert metrics["precision@2"] == 0.0
    assert metrics["mrr@2"] == 0.0
    assert metrics["ndcg@2"] == 0.0


def test_compute_metrics_partial_match_ranked_first():
    metrics = compute_metrics(predicted_ids=["1", "9"], gold_ids=["1", "2"], k=2)
    assert metrics["recall@2"] == 0.5
    assert metrics["match@2"] == 1
    assert metrics["precision@2"] == 0.5
    assert metrics["mrr@2"] == 1.0


def test_compute_metrics_partial_match_ranked_second():
    metrics = compute_metrics(predicted_ids=["9", "1"], gold_ids=["1", "2"], k=2)
    assert metrics["mrr@2"] == 0.5


def test_compute_metrics_empty_gold_ids_does_not_divide_by_zero():
    metrics = compute_metrics(predicted_ids=["1", "2"], gold_ids=[], k=2)
    assert metrics["recall@2"] == 0.0
    assert metrics["match@2"] == 0
    assert metrics["ndcg@2"] == 0.0


def test_compute_metrics_respects_k_cutoff():
    metrics = compute_metrics(predicted_ids=["9", "9", "1"], gold_ids=["1"], k=2)
    assert metrics["match@2"] == 0
    assert metrics["recall@2"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stair.core.eval'`

- [ ] **Step 3: Write `src/stair/core/eval/metrics.py`**

```python
import math


def compute_dcg(predicted_ids, gold_ids, k):
    discounted_cumulative_gain = 0.0
    for position, predicted_id in enumerate(predicted_ids[:k]):
        relevance = 1.0 if predicted_id in gold_ids else 0.0
        discounted_cumulative_gain += (2 ** relevance - 1) / math.log2(position + 2)
    return discounted_cumulative_gain


def compute_idcg(gold_ids, k):
    ideal_relevance_count = min(len(gold_ids), k)
    ideal_discounted_cumulative_gain = 0.0
    for position in range(ideal_relevance_count):
        ideal_discounted_cumulative_gain += (2 ** 1.0 - 1) / math.log2(position + 2)
    return ideal_discounted_cumulative_gain


def compute_metrics(predicted_ids, gold_ids, k):
    top_k_predicted_ids = predicted_ids[:k]
    hits = len(set(top_k_predicted_ids) & set(gold_ids))

    recall = hits / float(len(gold_ids)) if gold_ids else 0.0
    match = 1 if hits > 0 else 0
    precision = hits / float(k)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    first_correct_rank = None
    for position, predicted_id in enumerate(top_k_predicted_ids):
        if predicted_id in gold_ids:
            first_correct_rank = position + 1
            break
    mean_reciprocal_rank = 1.0 / first_correct_rank if first_correct_rank else 0.0

    discounted_cumulative_gain = compute_dcg(predicted_ids, gold_ids, k)
    ideal_discounted_cumulative_gain = compute_idcg(gold_ids, k)
    normalized_discounted_cumulative_gain = (
        discounted_cumulative_gain / ideal_discounted_cumulative_gain
        if ideal_discounted_cumulative_gain > 0
        else 0.0
    )

    return {
        f"recall@{k}": recall,
        f"match@{k}": match,
        f"precision@{k}": precision,
        f"f1@{k}": f1,
        f"mrr@{k}": mean_reciprocal_rank,
        f"ndcg@{k}": normalized_discounted_cumulative_gain,
    }
```

Create `src/stair/core/eval/__init__.py` (empty file).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/stair/core/eval/__init__.py src/stair/core/eval/metrics.py tests/test_metrics.py
git commit -m "feat: add retrieval metrics module for eval"
```

---

### Task 3: Single-generation output parsing (`core/eval/parse_output.py`)

**Files:**
- Create: `src/stair/core/eval/parse_output.py`
- Test: `tests/test_parse_output.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces: `stair.core.eval.parse_output.parse_predicted_sections(raw_text, title_to_id_map) -> (list, int)`. Returns `(predicted_ids, hallucination_count)`. `raw_text` is the model's raw generated completion (expected to be Python-list-of-strings syntax, matching what the model was trained to produce — see Task 1's `USER_PROMPT`). `title_to_id_map` maps `"section_num title"` strings to ids (the same shape `stair.core.train.io_functions.load_toc`'s `id_to_title_map` is the mirror image of — build a small local inverse-mapping helper in this task rather than importing across packages, since this module belongs to `eval`, not `train`). If `raw_text` isn't valid Python-list syntax at all, returns `([], 1)` (the whole completion counts as one hallucination). If it parses but contains entries not in `title_to_id_map`, those entries are dropped and each counted as a hallucination; valid entries are still returned as predicted ids in order. Used by Task 5 (`eval.py`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_parse_output.py`:

```python
from stair.core.eval.parse_output import parse_predicted_sections


TITLE_TO_ID_MAP = {
    "1 Introduction": "1",
    "2 Getting Started": "2",
}


def test_parse_predicted_sections_valid_list_all_known():
    predicted_ids, hallucination_count = parse_predicted_sections(
        '["1 Introduction", "2 Getting Started"]', TITLE_TO_ID_MAP
    )
    assert predicted_ids == ["1", "2"]
    assert hallucination_count == 0


def test_parse_predicted_sections_unparseable_text_is_a_full_hallucination():
    predicted_ids, hallucination_count = parse_predicted_sections(
        "I think the answer is in chapter one.", TITLE_TO_ID_MAP
    )
    assert predicted_ids == []
    assert hallucination_count == 1


def test_parse_predicted_sections_drops_unknown_titles_and_counts_hallucination():
    predicted_ids, hallucination_count = parse_predicted_sections(
        '["1 Introduction", "9 Nonexistent Section"]', TITLE_TO_ID_MAP
    )
    assert predicted_ids == ["1"]
    assert hallucination_count == 1


def test_parse_predicted_sections_empty_list_is_valid_with_no_hallucination():
    predicted_ids, hallucination_count = parse_predicted_sections("[]", TITLE_TO_ID_MAP)
    assert predicted_ids == []
    assert hallucination_count == 0


def test_parse_predicted_sections_non_string_entry_counts_as_hallucination():
    predicted_ids, hallucination_count = parse_predicted_sections("[1, 2]", TITLE_TO_ID_MAP)
    assert predicted_ids == []
    assert hallucination_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_parse_output.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stair.core.eval.parse_output'`

- [ ] **Step 3: Write `src/stair/core/eval/parse_output.py`**

```python
import ast


def parse_predicted_sections(raw_text, title_to_id_map):
    try:
        parsed_entries = ast.literal_eval(raw_text.strip())
    except (ValueError, SyntaxError):
        return [], 1

    if not isinstance(parsed_entries, list):
        return [], 1

    predicted_ids = []
    hallucination_count = 0
    for entry in parsed_entries:
        if not isinstance(entry, str):
            hallucination_count += 1
            continue
        if entry not in title_to_id_map:
            hallucination_count += 1
            continue
        predicted_ids.append(title_to_id_map[entry])

    return predicted_ids, hallucination_count
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_parse_output.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/stair/core/eval/parse_output.py tests/test_parse_output.py
git commit -m "feat: add single-generation output parser for eval"
```

---

### Task 4: Model loading and inference (`core/eval/inference.py`)

**Files:**
- Create: `src/stair/core/eval/inference.py`
- Test: `tests/test_inference.py`

**Interfaces:**
- Consumes: nothing from prior tasks (no import-time coupling; `build_generate_call`'s output is consumed by Task 5 via dependency injection, same pattern as `qa_generation.py`'s `llm_call` in Plan 2).
- Produces: `stair.core.eval.inference.build_generate_call(model, tokenizer, max_new_tokens) -> callable`. The returned callable has signature `generate_call(messages) -> str` (a list of `{"role", "content"}` dicts in, the raw generated completion text out) — matches the same `messages -> str` shape Plan 2's `qa_generation.generate_qa_pairs_for_chunk`'s injected `llm_call` already uses, for consistency across the codebase. Internally calls `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`, then `tokenizer(...)`/`model.generate(...)`/`tokenizer.decode(...)`, returning only the newly generated text (not the echoed prompt).
- Produces: `stair.core.eval.inference.load_finetuned_model(base_model_name, adapter_path) -> (model, tokenizer)` — the production loader, using `transformers.AutoModelForCausalLM.from_pretrained` + `peft.PeftModel.from_pretrained(base_model, adapter_path)` + `transformers.AutoTokenizer.from_pretrained`. Not exercised by this task's own tests (requires a real network download); used only by Task 5's real (non-test) code path.

- [ ] **Step 1: Write the failing test**

Create `tests/test_inference.py`:

```python
from stair.core.eval.inference import build_generate_call


class FakeTokenizer:
    def __init__(self):
        self.applied_messages = None
        self.decode_calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.applied_messages = messages
        return "RENDERED_PROMPT"

    def __call__(self, text, return_tensors):
        return {"input_ids": [[1, 2, 3]], "prompt_length": 3}

    def decode(self, token_ids, skip_special_tokens):
        self.decode_calls.append(token_ids)
        return '["1 Introduction"]'


class FakeModel:
    def __init__(self):
        self.generate_calls = []

    def generate(self, input_ids, max_new_tokens, **kwargs):
        self.generate_calls.append({"input_ids": input_ids, "max_new_tokens": max_new_tokens})
        return [[1, 2, 3, 4, 5]]


def test_build_generate_call_renders_chat_template_from_messages():
    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer()
    generate_call = build_generate_call(fake_model, fake_tokenizer, max_new_tokens=64)

    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    generate_call(messages)

    assert fake_tokenizer.applied_messages == messages


def test_build_generate_call_passes_max_new_tokens_to_generate():
    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer()
    generate_call = build_generate_call(fake_model, fake_tokenizer, max_new_tokens=64)

    generate_call([{"role": "user", "content": "hi"}])

    assert fake_model.generate_calls[0]["max_new_tokens"] == 64


def test_build_generate_call_returns_decoded_text():
    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer()
    generate_call = build_generate_call(fake_model, fake_tokenizer, max_new_tokens=64)

    result = generate_call([{"role": "user", "content": "hi"}])

    assert result == '["1 Introduction"]'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inference.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stair.core.eval.inference'`

- [ ] **Step 3: Write `src/stair/core/eval/inference.py`**

```python
def build_generate_call(model, tokenizer, max_new_tokens):
    def generate_call(messages):
        rendered_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokenized_input = tokenizer(rendered_prompt, return_tensors="pt")
        input_ids = tokenized_input["input_ids"]
        prompt_length = tokenized_input["prompt_length"]

        generated_token_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )

        new_token_ids = generated_token_ids[0][prompt_length:]
        generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        return generated_text

    return generate_call


def load_finetuned_model(base_model_name, adapter_path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    return model, tokenizer
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_inference.py -v`
Expected: PASS (3 tests)

Note: the fake tokenizer's `__call__` returns a plain dict with a `"prompt_length"` key as a convenience for this test's fake — a real HF tokenizer's output object doesn't have that key, so `generate_call`'s real implementation actually needs the prompt's token length computed a different way. Fix this before moving on: change `generate_call` to compute `prompt_length = input_ids.shape[1]` (works for both a real tokenizer's tensor output and any test fake that shapes its `input_ids` the same way) instead of reading a `"prompt_length"` dict key, and update the `FakeTokenizer.__call__`/test expectations above accordingly — in particular `FakeTokenizer.__call__` should return an object exposing `.shape` accessibly the way a real tensor does, or the test's `input_ids` should be restructured so `len(input_ids[0])` or `input_ids.shape[1]` works. Use your judgment on the cleanest way to make the fake genuinely representative of a real tokenizer's tensor-shaped output while keeping the test dependency-free (no real `torch`/`transformers` objects needed, just something that supports the same indexing/shape access `generate_call` uses) — this is exactly the kind of test-vs-implementation mismatch worth fixing for real rather than papering over with a fake that only works because it was shaped to match a first-draft implementation.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS (all tests from every prior task and plan)

- [ ] **Step 6: Commit**

```bash
git add src/stair/core/eval/inference.py tests/test_inference.py
git commit -m "feat: add model loading and inference for eval"
```

---

### Task 5: Wire `stair eval` end-to-end

**Files:**
- Create: `src/stair/eval.py`
- Modify: `src/stair/cli.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `compute_metrics` (Task 2), `parse_predicted_sections` (Task 3), `build_generate_call`/`load_finetuned_model` (Task 4), `SYSTEM_PROMPT`/`USER_PROMPT` (Task 1), `resolve_config`/`print_run_header` (Plan 1).
- Produces: `stair.eval.run_eval(run_dir, data_dir, config, generate_call=None) -> dict` — the testable entry point. Returns `{"count": int, "hallucination_rate": float, "metrics": {"recall@{k}": float, ...}}` (metrics averaged across the test set, one entry per configured `k`). Writes `<run_dir>/eval/eval_results.jsonl` (one record per test example: question, gold reference, predicted ids, raw generation, per-k metrics) and `<run_dir>/eval/eval_metrics.json` (the aggregated dict this function returns). Raises `FileNotFoundError` with a clear message if `data_dir` is missing `test.jsonl` or `toc.json`, or if `run_dir` is missing `checkpoint-best` — checked before loading any model. When `generate_call` is `None`, builds the real one via `load_finetuned_model`+`build_generate_call`, reading `model.name` from `<run_dir>/config.json` and `max_new_tokens` from `config["eval"]` (new key, see Step 3). `cli.py`'s `eval` subcommand calls this and prints the run header + summary.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval.py`:

```python
import json
from pathlib import Path

import pytest

from stair.config import resolve_config
from stair.eval import run_eval


def make_complete_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    toc = {
        "title": "Test Book",
        "table_of_contents": [
            {"id": "1", "section_num": "1", "title": "Introduction", "leaf": True},
            {"id": "2", "section_num": "2", "title": "Getting Started", "leaf": True},
        ],
    }
    (data_dir / "toc.json").write_text(json.dumps(toc))
    test_records = [
        {"question": "What is this book about?", "answer": "It is about testing.", "reference": ["1"]},
        {"question": "How do I get started?", "answer": "Read chapter two.", "reference": ["2"]},
    ]
    with open(data_dir / "test.jsonl", "w") as test_file:
        for record in test_records:
            test_file.write(json.dumps(record) + "\n")
    return data_dir


def make_complete_run_dir(tmp_path, config):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoint-best"
    checkpoint_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config))
    return run_dir


def make_fake_generate_call(responses_by_question):
    def fake_generate_call(messages):
        user_message_content = messages[1]["content"]
        for question, response in responses_by_question.items():
            if question in user_message_content:
                return response
        return "[]"

    return fake_generate_call


def test_run_eval_raises_when_test_jsonl_missing(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "toc.json").write_text("{}")
    config = resolve_config(override_path=None)
    run_dir = make_complete_run_dir(tmp_path, config)

    with pytest.raises(FileNotFoundError, match="test.jsonl"):
        run_eval(str(run_dir), str(data_dir), config, generate_call=make_fake_generate_call({}))


def test_run_eval_raises_when_checkpoint_best_missing(tmp_path):
    data_dir = make_complete_data_dir(tmp_path)
    config = resolve_config(override_path=None)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(config))

    with pytest.raises(FileNotFoundError, match="checkpoint-best"):
        run_eval(str(run_dir), str(data_dir), config, generate_call=make_fake_generate_call({}))


def test_run_eval_computes_perfect_metrics_when_predictions_match(tmp_path):
    data_dir = make_complete_data_dir(tmp_path)
    config = resolve_config(override_path=None)
    run_dir = make_complete_run_dir(tmp_path, config)
    fake_generate_call = make_fake_generate_call(
        {
            "What is this book about?": '["1 Introduction"]',
            "How do I get started?": '["2 Getting Started"]',
        }
    )

    result = run_eval(str(run_dir), str(data_dir), config, generate_call=fake_generate_call)

    assert result["count"] == 2
    assert result["hallucination_rate"] == 0.0
    top_k = config["eval"]["ks"][0]
    assert result["metrics"][f"recall@{top_k}"] == 1.0


def test_run_eval_writes_results_and_metrics_files(tmp_path):
    data_dir = make_complete_data_dir(tmp_path)
    config = resolve_config(override_path=None)
    run_dir = make_complete_run_dir(tmp_path, config)
    fake_generate_call = make_fake_generate_call({})

    run_eval(str(run_dir), str(data_dir), config, generate_call=fake_generate_call)

    assert (run_dir / "eval" / "eval_results.jsonl").exists()
    assert (run_dir / "eval" / "eval_metrics.json").exists()
    result_lines = (run_dir / "eval" / "eval_results.jsonl").read_text().strip().split("\n")
    assert len(result_lines) == 2


def test_run_eval_tracks_hallucination_rate_for_unparseable_generations(tmp_path):
    data_dir = make_complete_data_dir(tmp_path)
    config = resolve_config(override_path=None)
    run_dir = make_complete_run_dir(tmp_path, config)
    fake_generate_call = make_fake_generate_call(
        {
            "What is this book about?": "not valid python at all",
            "How do I get started?": '["2 Getting Started"]',
        }
    )

    result = run_eval(str(run_dir), str(data_dir), config, generate_call=fake_generate_call)

    assert result["hallucination_rate"] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stair.eval'`

- [ ] **Step 3: Add `eval.max_new_tokens` to `configs/base.yaml`**

The `eval:` section already has `ks` and `top_k` (from Plan 1). Add one more key, with a comment:

```yaml
eval:
  # recall/precision/NDCG/MRR are reported at each of these cutoffs
  ks: [1, 3, 5, 10]
  top_k: 10
  # how many new tokens the fine-tuned model may generate per query at eval time
  max_new_tokens: 128
```

- [ ] **Step 4: Write `src/stair/eval.py`**

```python
import json
from pathlib import Path

from stair.core.eval.inference import build_generate_call, load_finetuned_model
from stair.core.eval.metrics import compute_metrics
from stair.core.eval.parse_output import parse_predicted_sections
from stair.core.prompts import SYSTEM_PROMPT, USER_PROMPT
from stair.run_header import print_run_header


def check_required_eval_inputs(data_dir, run_dir):
    data_dir_path = Path(data_dir)
    required_data_file_names = ["test.jsonl", "toc.json"]
    for required_file_name in required_data_file_names:
        required_file_path = data_dir_path / required_file_name
        if not required_file_path.exists():
            raise FileNotFoundError(
                f"Expected {required_file_name} under {data_dir} "
                f"(run `stair prepare-data` first to produce it)"
            )

    checkpoint_best_path = Path(run_dir) / "checkpoint-best"
    if not checkpoint_best_path.exists():
        raise FileNotFoundError(
            f"Expected checkpoint-best under {run_dir} "
            f"(run `stair train` first, and make sure it completed successfully)"
        )


def load_toc(toc_json_path):
    with open(toc_json_path, "r") as toc_file:
        toc = json.load(toc_file)

    title = toc["title"]
    toc_lines = []
    title_to_id_map = {}
    for node in toc["table_of_contents"]:
        toc_line = f"{node['section_num']} {node['title']}"
        toc_lines.append(toc_line)
        title_to_id_map[toc_line] = node["id"]

    return title, "\n".join(toc_lines), title_to_id_map


def read_test_records(test_jsonl_path):
    test_records = []
    with open(test_jsonl_path, "r") as test_file:
        for line in test_file:
            test_records.append(json.loads(line))
    return test_records


def build_default_generate_call(run_dir, config):
    run_config_path = Path(run_dir) / "config.json"
    with open(run_config_path, "r") as run_config_file:
        run_config = json.load(run_config_file)

    base_model_name = run_config["model"]["name"]
    adapter_path = str(Path(run_dir) / "checkpoint-best")
    max_new_tokens = config["eval"]["max_new_tokens"]

    model, tokenizer = load_finetuned_model(base_model_name, adapter_path)
    return build_generate_call(model, tokenizer, max_new_tokens)


def run_eval(run_dir, data_dir, config, generate_call=None):
    print_run_header("stair eval", config)

    check_required_eval_inputs(data_dir, run_dir)

    eval_output_dir = Path(run_dir) / "eval"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    book_title, toc_text, title_to_id_map = load_toc(Path(data_dir) / "toc.json")
    print(f"Loaded table of contents for: {book_title}")

    test_records = read_test_records(Path(data_dir) / "test.jsonl")
    print(f"Loaded {len(test_records)} test examples")

    if generate_call is None:
        generate_call = build_default_generate_call(run_dir, config)

    ks = config["eval"]["ks"]
    metrics_totals = {}
    for k in ks:
        for metric_name in ["recall", "match", "precision", "f1", "mrr", "ndcg"]:
            metrics_totals[f"{metric_name}@{k}"] = 0.0

    total_hallucinations = 0
    total_predicted_entries = 0
    result_records = []

    results_path = eval_output_dir / "eval_results.jsonl"
    with open(results_path, "w") as results_file:
        for test_record in test_records:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        title=book_title, toc=toc_text, question=test_record["question"]
                    ),
                },
            ]

            raw_generation = generate_call(messages)
            predicted_ids, hallucination_count = parse_predicted_sections(
                raw_generation, title_to_id_map
            )
            total_hallucinations += hallucination_count
            total_predicted_entries += hallucination_count + len(predicted_ids)

            gold_ids = test_record["reference"]
            per_k_metrics = {}
            for k in ks:
                k_metrics = compute_metrics(predicted_ids, gold_ids, k)
                per_k_metrics.update(k_metrics)
                for metric_name, metric_value in k_metrics.items():
                    metrics_totals[metric_name] += metric_value

            result_record = {
                "question": test_record["question"],
                "reference": gold_ids,
                "predicted_ids": predicted_ids,
                "raw_generation": raw_generation,
                "metrics": per_k_metrics,
            }
            result_records.append(result_record)
            results_file.write(json.dumps(result_record) + "\n")

    print(f"Results saved: {results_path}")

    test_count = len(test_records)
    averaged_metrics = {}
    for metric_name, total_value in metrics_totals.items():
        averaged_metrics[metric_name] = total_value / test_count if test_count else 0.0

    hallucination_rate = (
        total_hallucinations / total_predicted_entries if total_predicted_entries else 0.0
    )

    summary = {
        "count": test_count,
        "hallucination_rate": hallucination_rate,
        "metrics": averaged_metrics,
    }

    metrics_path = eval_output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as metrics_file:
        json.dump(summary, metrics_file, indent=2)
    print(f"Metrics saved: {metrics_path}")

    return summary
```

- [ ] **Step 5: Wire `eval` into `cli.py`**

In `src/stair/cli.py`, add the import:

```python
from stair.eval import run_eval
```

In `main()`, add a branch for `eval` alongside the existing `train`/`prepare-data` branches:

```python
    if args.subcommand == "eval":
        try:
            summary = run_eval(args.run, args.data, resolved_config)
        except (OSError, ValueError) as error:
            print(f"stair eval: {error}", file=sys.stderr)
            return 1
        print(f"Evaluated {summary['count']} examples, hallucination rate: {summary['hallucination_rate']:.2%}")
        return 0
```

The `eval` subparser (from Plan 1) already has `--run` and `--config`; add a `--data` argument matching the one `train`'s subparser already has:

```python
    parser.add_argument(
        "--data", required=True,
        help="Directory produced by `stair prepare-data` (must contain test.jsonl, toc.json)",
    )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_eval.py -v`
Expected: PASS (5 tests)

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS (all tests from every prior task and plan)

- [ ] **Step 8: Commit**

```bash
git add src/stair/eval.py src/stair/cli.py tests/test_eval.py configs/base.yaml
git commit -m "feat: wire stair eval to run inference and compute retrieval metrics"
```

---

### Task 6: `stair report-eval`

**Files:**
- Create: `src/stair/core/report/eval_report.py`
- Modify: `src/stair/cli.py`
- Test: `tests/test_report_eval.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-5 directly (reads plain `eval_metrics.json`, tested against fixtures).
- Produces: `stair.core.report.eval_report.load_eval_metrics(run_dir) -> dict` — reads `<run_dir>/eval/eval_metrics.json`; raises `FileNotFoundError` with a clear message if missing. `stair.core.report.eval_report.render_metrics_table(summary, ks) -> None` — prints a terminal table with one row per metric (recall/precision/f1/mrr/ndcg) and one column per configured `k`, plus the hallucination rate and example count. `stair.core.report.eval_report.flag_weak_spots(summary, eval_config) -> list` — returns a list of human-readable warning strings using new `eval_config` thresholds (`weak_recall_threshold`, `high_hallucination_rate_threshold` — see Step 3). `cli.py`'s `report-eval` subcommand calls all three in sequence.

- [ ] **Step 1: Write the failing test**

Create `tests/test_report_eval.py`:

```python
import json

import pytest

from stair.core.report.eval_report import flag_weak_spots, load_eval_metrics, render_metrics_table


EVAL_CONFIG = {
    "weak_recall_threshold": 0.5,
    "high_hallucination_rate_threshold": 0.2,
}


def write_eval_metrics(path, summary):
    path.write_text(json.dumps(summary))


def test_load_eval_metrics_reads_the_file(tmp_path):
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True)
    summary = {"count": 2, "hallucination_rate": 0.0, "metrics": {"recall@1": 1.0}}
    write_eval_metrics(eval_dir / "eval_metrics.json", summary)

    loaded_summary = load_eval_metrics(str(run_dir))

    assert loaded_summary == summary


def test_load_eval_metrics_raises_clear_error_when_missing(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="eval_metrics.json"):
        load_eval_metrics(str(run_dir))


def test_render_metrics_table_does_not_raise(capsys):
    summary = {
        "count": 10,
        "hallucination_rate": 0.1,
        "metrics": {"recall@1": 0.8, "recall@3": 0.9, "mrr@1": 0.7, "mrr@3": 0.75},
    }
    render_metrics_table(summary, ks=[1, 3])

    captured_output = capsys.readouterr()
    assert "recall" in captured_output.out.lower()


def test_flag_weak_spots_flags_low_recall():
    summary = {"count": 10, "hallucination_rate": 0.0, "metrics": {"recall@1": 0.2}}

    warnings = flag_weak_spots(summary, EVAL_CONFIG)

    joined_warnings = " ".join(warnings).lower()
    assert "recall" in joined_warnings


def test_flag_weak_spots_flags_high_hallucination_rate():
    summary = {"count": 10, "hallucination_rate": 0.5, "metrics": {}}

    warnings = flag_weak_spots(summary, EVAL_CONFIG)

    joined_warnings = " ".join(warnings).lower()
    assert "hallucination" in joined_warnings


def test_flag_weak_spots_returns_empty_list_when_healthy():
    summary = {"count": 10, "hallucination_rate": 0.0, "metrics": {"recall@1": 0.9}}

    warnings = flag_weak_spots(summary, EVAL_CONFIG)

    assert warnings == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_report_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stair.core.report.eval_report'`

- [ ] **Step 3: Add weak-spot thresholds to `configs/base.yaml`'s `eval:` section**

```yaml
eval:
  ks: [1, 3, 5, 10]
  top_k: 10
  max_new_tokens: 128
  # report-eval flags recall@k as weak if it falls below this value, for the
  # smallest configured k (the most forgiving cutoff — if even that's low,
  # the model is struggling)
  weak_recall_threshold: 0.5
  # report-eval flags the run's hallucination rate as high if it exceeds this
  high_hallucination_rate_threshold: 0.2
```

- [ ] **Step 4: Write `src/stair/core/report/eval_report.py`**

```python
import json
from pathlib import Path


def load_eval_metrics(run_dir):
    metrics_path = Path(run_dir) / "eval" / "eval_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No eval_metrics.json found under {run_dir} (run `stair eval` first)"
        )
    with open(metrics_path, "r") as metrics_file:
        return json.load(metrics_file)


def render_metrics_table(summary, ks):
    print(f"Evaluated {summary['count']} examples")
    print(f"Hallucination rate: {summary['hallucination_rate']:.2%}")
    print("")

    metric_names = ["recall", "match", "precision", "f1", "mrr", "ndcg"]
    header_columns = ["metric"]
    for k in ks:
        header_columns.append(f"@{k}")
    print("  ".join(header_columns))

    for metric_name in metric_names:
        row_columns = [metric_name]
        for k in ks:
            metric_key = f"{metric_name}@{k}"
            metric_value = summary["metrics"].get(metric_key)
            if metric_value is None:
                row_columns.append("-")
            else:
                row_columns.append(f"{metric_value:.3f}")
        print("  ".join(row_columns))


def flag_weak_spots(summary, eval_config):
    warnings = []

    weak_recall_threshold = eval_config["weak_recall_threshold"]
    smallest_recall_key = None
    smallest_k = None
    for metric_key in summary["metrics"]:
        if metric_key.startswith("recall@"):
            this_k = int(metric_key.split("@")[1])
            if smallest_k is None or this_k < smallest_k:
                smallest_k = this_k
                smallest_recall_key = metric_key
    if smallest_recall_key is not None:
        smallest_recall_value = summary["metrics"][smallest_recall_key]
        if smallest_recall_value < weak_recall_threshold:
            warnings.append(
                f"{smallest_recall_key} is {smallest_recall_value:.2f}, below the "
                f"{weak_recall_threshold} threshold — the model is missing relevant "
                f"sections even at the most forgiving cutoff."
            )

    high_hallucination_rate_threshold = eval_config["high_hallucination_rate_threshold"]
    if summary["hallucination_rate"] > high_hallucination_rate_threshold:
        warnings.append(
            f"Hallucination rate is {summary['hallucination_rate']:.2%}, above the "
            f"{high_hallucination_rate_threshold:.0%} threshold — the model is often "
            f"generating section titles that don't exist in the table of contents."
        )

    return warnings
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_report_eval.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Wire `report-eval` into `cli.py`**

In `src/stair/cli.py`, add the import:

```python
from stair.core.report.eval_report import flag_weak_spots, load_eval_metrics, render_metrics_table
```

Add a branch in `main()`:

```python
    if args.subcommand == "report-eval":
        try:
            summary = load_eval_metrics(args.run)
        except FileNotFoundError as error:
            print(f"stair report-eval: {error}", file=sys.stderr)
            return 1
        base_config = resolve_config(override_path=None)
        render_metrics_table(summary, base_config["eval"]["ks"])
        warnings = flag_weak_spots(summary, base_config["eval"])
        if warnings:
            print("")
            print("Weak spots:")
            for warning in warnings:
                print(f"- {warning}")
        return 0
```

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS (all tests from every prior task and plan)

- [ ] **Step 8: Commit**

```bash
git add src/stair/core/report/eval_report.py src/stair/cli.py tests/test_report_eval.py configs/base.yaml
git commit -m "feat: add stair report-eval terminal metrics table and weak-spot warnings"
```

---

## Self-Review Notes

- **Spec coverage:** implements spec Goal 1 (`eval`/`report-eval` subcommands) and Goal 4 (terminal eval report) in full, closing out the last two stub subcommands from Plan 1 — after this plan, no `stair` subcommand prints "not implemented yet" anymore. Metrics (recall, precision, F1, MRR, NDCG, hallucination rate) match what the user originally asked for.
- **Placeholder scan:** no TBD/TODO. Task 4's Step 4 note about the fake tokenizer's `prompt_length` mismatch is a deliberate, explicit mid-task correction (a plan defect caught during self-review before dispatch, not left unresolved) — the implementer is told exactly what's wrong and why, not left to guess.
- **Type consistency:** `compute_metrics(predicted_ids, gold_ids, k) -> dict` (Task 2) and `parse_predicted_sections(raw_text, title_to_id_map) -> (list, int)` (Task 3) are both consumed by Task 5 exactly as declared. `build_generate_call(model, tokenizer, max_new_tokens) -> callable` (Task 4) produces the same `messages -> str` shape Task 5 expects, matching Plan 2's `llm_call` convention.
- **Review Focus:** all five items have a task and a test — unparseable completion as full hallucination (Task 3, `test_parse_predicted_sections_unparseable_text_is_a_full_hallucination`), unknown title dropped and counted (Task 3, `test_parse_predicted_sections_drops_unknown_titles_and_counts_hallucination`), empty gold_ids no division by zero (Task 2, `test_compute_metrics_empty_gold_ids_does_not_divide_by_zero`), missing `eval_metrics.json` fails clearly (Task 6, `test_load_eval_metrics_raises_clear_error_when_missing`), missing `checkpoint-best` fails clearly before model loading (Task 5, `test_run_eval_raises_when_checkpoint_best_missing`).

## Next Plan

None currently planned — after this plan, all five `stair` subcommands (`prepare-data`, `train`, `report-train`, `eval`, `report-eval`) have real implementations, closing out the external-facing pipeline spec's core scope. Future work (if any) would be a new spec/design cycle, not a continuation of this one.
