#!/bin/sh
DATA_DIR="${HOME}/data/ki_data"
DATASET_ID="systemz/all/v30"
DOCS_JSONL="${DATA_DIR}/cpt/${DATASET_ID}/docs.jsonl"
OUT_DIR="${DATA_DIR}/cpt/${DATASET_ID}"

# uv run src/tokens/stats.py \
#   --input_jsonl "${DOCS_JSONL}" \
#   --out_dir "${OUT_DIR}" \
#   --model_id "ibm-granite/granite-3.3-8b-base"

TOKEN_TO_CHAR="4.0"
MAX_TOKENS="2048"

OUT_JSONL="${DATA_DIR}/cpt/${DATASET_ID}/chunks.${MAX_TOKENS}.jsonl"
uv run src/chunk/split.py \
  --input_jsonl "${DOCS_JSONL}" \
  --output_jsonl "${OUT_JSONL}" \
  --max_tokens ${MAX_TOKENS} \
  --token_to_char ${TOKEN_TO_CHAR} \
  --chunk_overlap 32