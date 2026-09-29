# STAIR

STAIR fine-tunes a language model to find the right section of a document
for a user's question — given a book or manual's table of contents, the
model learns to pick which section(s) answer a given query. This repo is
the pipeline for that: parse your own document into training data, train a
model on it, and see reports on how well it worked. You never need to read
or write Python — everything is driven by a `stair` command and small YAML
config files.

## Status

This repo is being built in stages. Right now the foundation is in place:
the `stair` command-line tool and its config system. The actual
parse/train/evaluate steps are not implemented yet — running them today
prints `not implemented yet` and exits. See `docs/superpowers/plans/` for
what's built and what's next.

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/) — this project uses `uv` to manage its
  Python environment and dependencies, the same way the other projects in
  this repo do. If you don't have it: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Getting started

1. Clone this repository and move into it.
2. Run any `stair` command with `uv run` — `uv` will set up the environment
   the first time you run it, automatically:

   ```
   uv run stair --help
   ```

That's it. You do not `pip install` this project — you always run it with
`uv run` from inside a clone of this repo.

## The `stair` commands

```
uv run stair prepare-data --corpus <path-to-pdf> --out <output-dir>
uv run stair train --out <run-dir>
uv run stair report-train --run <run-dir>
uv run stair eval --run <run-dir>
uv run stair report-eval --run <run-dir>
```

- `prepare-data` turns a raw document into training/validation/test data.
- `train` fine-tunes a model on that data.
- `report-train` prints a terminal report on how training went: loss
  curves, whether it overfit, and recommendations.
- `eval` runs the trained model against the held-out test data.
- `report-eval` prints a terminal report of the evaluation metrics.

Run any command with `--help` to see its exact options, e.g.
`uv run stair train --help`.

`stair train` requires a local CUDA GPU — the vendored trainer has no CPU
fallback.

## Configuring a run

Every setting the pipeline uses — which model to train, how many epochs,
learning rate, and so on — has a default in `configs/base.yaml`. You should
never edit that file. Instead, write your own small YAML file with only the
settings you want to change, and pass it with `--config`:

```yaml
# my_run.yaml
model:
  lora:
    rank: 32
training:
  num_train_epochs: 5
```

```
uv run stair train --out runs/my-run --config my_run.yaml
```

Anything you leave out of your file keeps its default from
`configs/base.yaml`. If you misspell a setting, `stair` will tell you
immediately instead of silently ignoring it — open `configs/base.yaml` to
see every valid setting and what it does (each one has a comment
explaining it).

### Environment variables for `prepare-data`

`prepare-data`'s synthetic QA-generation step calls an OpenAI-compatible
LLM endpoint, so you need to set three environment variables before running
it:

- `STAIR_LLM_API_BASE` — the endpoint's base URL
- `STAIR_LLM_API_KEY` — your API key for that endpoint
- `STAIR_LLM_MODEL` — the model name to use

Once those are set, you can try `prepare-data` against the bundled example
corpus at `data/example_book/sourdough_bread_guide.pdf`:

```
uv run stair prepare-data --corpus data/example_book/sourdough_bread_guide.pdf --out out/example-run
```

## Running the tests

```
uv run pytest
```
