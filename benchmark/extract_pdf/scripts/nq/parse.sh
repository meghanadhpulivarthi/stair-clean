#!/bin/bash

DATA_DIR="${HOME}/data/ki_data/raw/nq"
SECTIONS_JSONL="${HOME}/data/ki_data/cpt/nq-10k/sections.jsonl"
QS_JSONL="${HOME}/data/ki_data/qs/nq-10k/qs.jsonl"

uv run src/nq/parse.py \
  --in_jsonl ${DATA_DIR}/v1.0-simplified_simplified-nq-train.jsonl \
  --sections_jsonl ${SECTIONS_JSONL} \
  --qs_jsonl ${QS_JSONL} \
  --max_docs 10000

  