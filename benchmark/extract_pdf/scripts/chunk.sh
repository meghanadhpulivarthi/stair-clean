#!/bin/sh
DATASET_ID="${1}"

DATA_DIR="${HOME}/data/ki_data"
DOCS_JSONL="${DATA_DIR}/cpt/${DATASET_ID}/sections.jsonl"

OUT_DIR="${DATA_DIR}/cpt/${DATASET_ID}"
MODEL="mistralai/Mistral-7B-v0.3"

# uv run src/tokens/stats.py \
#   --input_jsonl "${DOCS_JSONL}" \
#   --out_dir "${OUT_DIR}" \
#   --model_id "${MODEL}"

TOKEN_TO_CHAR="3.5"
MAX_TOKENS="2048"

OUT_JSONL="${DATA_DIR}/cpt/${DATASET_ID}/chunks.${MAX_TOKENS}.jsonl"
uv run src/chunk/split.py \
  --input_jsonl "${DOCS_JSONL}" \
  --output_jsonl "${OUT_JSONL}" \
  --max_tokens ${MAX_TOKENS} \
  --token_to_char ${TOKEN_TO_CHAR} \
  --chunk_overlap 32 \
  --add_prefix \
  --col_prefix "doc_id"