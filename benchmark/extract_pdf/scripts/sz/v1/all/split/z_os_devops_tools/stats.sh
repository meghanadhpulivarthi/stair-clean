#!/bin/bash
DATASET_ID="sz/v1/all/split/z_os_devops_tools"
DATA_DIR="${HOME}/data/ki_data"
DOCS_JSONL="${DATA_DIR}/cpt/${DATASET_ID}/docs.jsonl"
OUT_DIR="${DATA_DIR}/cpt/${DATASET_ID}/"

uv run src/tokens/stats.py \
  --input_jsonl "${DOCS_JSONL}" \
  --out_dir "${OUT_DIR}" \
  --model_id "mistralai/Mistral-7B-v0.3"
