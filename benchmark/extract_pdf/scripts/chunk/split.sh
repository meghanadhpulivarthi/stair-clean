#!/bin/sh
DATA_DIR="${HOME}/data/ki_data"
DATASET_ID="${1}"
TOKEN_TO_CHAR="${2}"
MAX_TOKENS="${3}"
SECTIONS_JSONL="${DATA_DIR}/cpt/${DATASET_ID}/sections.jsonl"


SINGLE_OUT_JSONL="${DATA_DIR}/cpt/${DATASET_ID}/chunks.all.prefix.jsonl"
echo "Processing ${MAX_TOKENS} tokens with ${CHUNK_OVERLAP} chunk overlap"
OUT_JSONL="${DATA_DIR}/cpt/${DATASET_ID}/chunks.${MAX_TOKENS}.prefix.jsonl"
uv run src/chunk/split.py \
  --input_jsonl "${SECTIONS_JSONL}" \
  --output_jsonl "${OUT_JSONL}" \
  --max_tokens ${MAX_TOKENS} \
  --token_to_char ${TOKEN_TO_CHAR} \
  --chunk_overlap 32 \
  --add_prefix
cat "${OUT_JSONL}" >> "${SINGLE_OUT_JSONL}"


SINGLE_OUT_JSONL="${DATA_DIR}/cpt/${DATASET_ID}/chunks.all.jsonl"
echo "Processing ${MAX_TOKENS} tokens with ${CHUNK_OVERLAP} chunk overlap"
OUT_JSONL="${DATA_DIR}/cpt/${DATASET_ID}/chunks.${MAX_TOKENS}.jsonl"
uv run src/chunk/split.py \
  --input_jsonl "${SECTIONS_JSONL}" \
  --output_jsonl "${OUT_JSONL}" \
  --max_tokens ${MAX_TOKENS} \
  --token_to_char ${TOKEN_TO_CHAR} \
  --chunk_overlap 32
cat "${OUT_JSONL}" >> "${SINGLE_OUT_JSONL}"
