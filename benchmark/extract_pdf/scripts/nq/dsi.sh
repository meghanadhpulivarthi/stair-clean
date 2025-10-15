#!/bin/bash

DATA_DIR="${HOME}/data/ki_data/raw/nq"
TRAIN_JSONL="${HOME}/data/ki_data/dsi/nq-1k/train.jsonl"
QS_JSONL="${HOME}/data/ki_data/dsi/nq-1k/qs.jsonl"

uv run src/nq/dsi.py \
  --in_jsonl ${DATA_DIR}/v1.0-simplified_simplified-nq-train.jsonl \
  --train_jsonl ${TRAIN_JSONL} \
  --qs_jsonl ${QS_JSONL} \
  --max_docs 1000