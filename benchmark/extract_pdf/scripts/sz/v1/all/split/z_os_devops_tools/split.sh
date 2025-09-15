#!/bin/sh
DATA_DIR="${HOME}/data/ki_data"
DATASET_ID="sz/v1/all/split/z_os_devops_tools"
TOKEN_TO_CHAR="3.85"
MAX_TOKENS="2048"
DOCS_JSONL="${DATA_DIR}/cpt/${DATASET_ID}/docs.jsonl"


OUT_JSONL="${DATA_DIR}/cpt/${DATASET_ID}/chunks.${MAX_TOKENS}.jsonl"
uv run src/chunk/split.py \
  --input_jsonl "${DOCS_JSONL}" \
  --output_jsonl "${OUT_JSONL}" \
  --max_tokens ${MAX_TOKENS} \
  --token_to_char ${TOKEN_TO_CHAR} \
  --chunk_overlap 32