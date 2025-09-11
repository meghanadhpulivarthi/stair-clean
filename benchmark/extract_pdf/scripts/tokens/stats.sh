#!/bin/sh
DATA_DIR="${HOME}/data/ki_data"
DATASET_ID="${1}"
SECTIONS_JSONL="${DATA_DIR}/cpt/${DATASET_ID}/sections.jsonl"
OUT_DIR="${DATA_DIR}/cpt/${DATASET_ID}/"

uv run src/tokens/stats.py \
  --input_jsonl "${SECTIONS_JSONL}" \
  --out_dir "${OUT_DIR}" \
  --model_id "mistralai/Mistral-7B-v0.3"