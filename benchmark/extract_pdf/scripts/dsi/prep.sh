#!/bin/sh
DATASET_ID="${1}"
CHUNKS_JSONL="${HOME}/data/ki_data/cpt/${DATASET_ID}/chunks.2048.jsonl"
TRAIN_JSONL="${HOME}/data/ki_data/dsi/${DATASET_ID}/train.jsonl"

uv run src/dsi/prep.py \
  --chunks_jsonl ${CHUNKS_JSONL} \
  --train_jsonl ${TRAIN_JSONL}